"""Corrected gating heads and training loops for DESA.

PARTIALLY SUPERSEDED: `LastTokenGatingHead` and `train_turn_gate_v2` are still
the active turn-gate implementation (used by notebook 10). The X-LoRA training
in `train_xlora_classifier_v2` is superseded by `routing_objectives.py` — it
was never run, and it contains a silent bug: it extracts classifier inputs by
calling the inner model directly, which (verified against peft 0.18.1) applies
ALL adapters at FULL weight instead of suppressing them. Use
`routing_objectives.train_router` instead.

This module is additive — it does not modify the original `gating.py`. It exposes:

- `LastTokenGatingHead`: same MLP architecture as `gating.TurnLevelGatingHead`
  but pools by selecting the last real token instead of mean-pooling.
- `train_turn_gate_v2`: retrains the turn-level gate with last-token pooling,
  longer schedule, label smoothing, and per-epoch alpha-distribution diagnostics
  to detect collapse.
- `train_xlora_classifier_v2`: retrains the X-LoRA classifier with an entropy
  regularizer that pushes per-(token, layer) routing distributions away from
  uniform, plus param-count assertions to catch silent state-dict load failures.

The motivation for these fixes is documented in
`GATING_LITERATURE_AND_EMBEDDINGS.html`: published LoRA-MoE methods (LoRAMoE,
X-LoRA, MoLE) do not mean-pool, and every method that doesn't collapse uses
some form of anti-collapse regularization.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gating import _collate, build_context_prompt, load_gating_records
from paths import OUTPUTS_DIR, SPLITS_DIR


# ---------------------------------------------------------------------------
# Last-token gating head
# ---------------------------------------------------------------------------


class LastTokenGatingHead(nn.Module):
    """Pool the last real token's hidden state and predict adapter weights.

    Architecture is identical to `gating.TurnLevelGatingHead` (4096 -> 1024 ->
    num_adapters with ReLU and dropout). Only the pooling step changes: we
    select h_T (the last real token, accounting for padding) instead of the
    mean across all tokens. In a causal LM, h_T has already attended over the
    full prompt, so it is a context-aware summary without the signal-dilution
    that mean-pooling produces over long emotional dialogues.
    """

    def __init__(self, hidden_dim: int, num_adapters: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_adapters = num_adapters
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_adapters),
        )

    @staticmethod
    def _last_token_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        # Last real-token index per sequence, robust to LEFT or RIGHT padding.
        # `cumsum(mask).argmax` returns the first position holding the running
        # max (the final 1) = the last real token wherever the pads sit. The
        # naive `mask.sum()-1` is right-padding-only; the eval/training
        # tokenizer pads LEFT, so it would read the wrong token in a batch.
        last_idx = attention_mask.long().cumsum(dim=1).argmax(dim=1)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_idx]

    def logits(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self._last_token_pool(hidden_states, attention_mask)
        # Hidden states from a quantized base may be bfloat16/float16; the gate's
        # linear layers are float32. Indexing preserves the input dtype, so cast
        # explicitly to avoid `mat1/mat2 dtype mismatch` in F.linear.
        gate_dtype = next(self.gate.parameters()).dtype
        if pooled.dtype != gate_dtype:
            pooled = pooled.to(gate_dtype)
        return self.gate(pooled)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.softmax(self.logits(hidden_states, attention_mask), dim=-1)


# ---------------------------------------------------------------------------
# Turn-level gate training (v2)
# ---------------------------------------------------------------------------


def _alpha_diagnostics(
    base_model,
    tokenizer,
    gate: LastTokenGatingHead,
    loader: DataLoader,
    device,
) -> dict[str, float]:
    """Return mean/std/entropy stats of the gate's outputs across `loader`."""
    del tokenizer  # unused; collate already tokenized
    gate.eval()
    alphas = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = base_model(**batch, output_hidden_states=True)
            alpha = gate(outputs.hidden_states[-1], batch["attention_mask"])
            alphas.append(alpha.detach().float().cpu())
    if not alphas:
        return {"mean_alpha": float("nan"), "std_alpha": float("nan"), "mean_entropy": float("nan")}
    stacked = torch.cat(alphas, dim=0)
    mean_alpha = stacked.mean(dim=0)
    std_alpha = stacked.std(dim=0)
    clipped = stacked.clamp(min=1e-9)
    entropies = -(clipped * clipped.log()).sum(dim=-1)
    return {
        "mean_alpha": [round(v, 4) for v in mean_alpha.tolist()],
        "std_alpha": [round(v, 4) for v in std_alpha.tolist()],
        "min_std": float(std_alpha.min()),
        "mean_entropy": float(entropies.mean()),
        "max_entropy": float(math.log(stacked.shape[1])),
    }


def evaluate_turn_gate_v2(
    base_model,
    gate: LastTokenGatingHead,
    loader: DataLoader,
    device,
    label_smoothing: float = 0.0,
) -> float:
    """Validation cross-entropy for the v2 gate."""
    gate.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = base_model(**batch, output_hidden_states=True)
            logits = gate.logits(outputs.hidden_states[-1], batch["attention_mask"])
            total += F.cross_entropy(logits, labels, label_smoothing=label_smoothing).item()
            n += 1
    return total / max(1, n)


def train_turn_gate_v2(
    base_model,
    tokenizer,
    output_path: Path = OUTPUTS_DIR / "turn_gate_v2.pt",
    hidden_dim: int = 4096,
    epochs: int = 8,
    batch_size: int = 4,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    max_length: int = 512,
    max_per_cluster: int = 512,
    label_smoothing: float = 0.1,
    collapse_std_threshold: float = 0.02,
) -> LastTokenGatingHead:
    """Train the turn-level gate with last-token pooling and a longer schedule.

    Differences vs `gating.train_turn_gate`:
    - Uses `LastTokenGatingHead` (last-token pool, not mean pool).
    - 8 epochs at lr 5e-4 with cosine schedule (was 2 epochs at lr 1e-4).
    - `cross_entropy(..., label_smoothing=0.1)` to soften overconfident targets.
    - Per-epoch diagnostics: prints mean/std/entropy of alpha across val set.
    - Warns at end of training if `min_std < collapse_std_threshold`.
    """
    device = next(base_model.parameters()).device
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    train_records = load_gating_records("train", max_per_cluster=max_per_cluster)
    val_records = load_gating_records("val", max_per_cluster=max(32, max_per_cluster // 8))
    gate = LastTokenGatingHead(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=weight_decay)

    def collate_fn(items):
        return _collate(items, tokenizer, max_length)

    train_loader = DataLoader(train_records, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_records, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)

    best_val = math.inf
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        gate.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Turn gate v2 epoch {epoch + 1}/{epochs}"):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = base_model(**batch, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
            logits = gate.logits(hidden_states, batch["attention_mask"])
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()

        val_loss = evaluate_turn_gate_v2(base_model, gate, val_loader, device, label_smoothing=label_smoothing)
        diagnostics = _alpha_diagnostics(base_model, tokenizer, gate, val_loader, device)
        print(
            f"Epoch {epoch + 1}: train_loss={running / steps_per_epoch:.4f} "
            f"val_loss={val_loss:.4f} diagnostics={diagnostics}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(gate.state_dict(), output_path)
            print(f"Saved best v2 turn gate to {output_path}")

    gate.load_state_dict(torch.load(output_path, map_location=device))
    final_diag = _alpha_diagnostics(base_model, tokenizer, gate, val_loader, device)
    print(f"Final v2 gate diagnostics: {final_diag}")
    if isinstance(final_diag.get("min_std"), float) and final_diag["min_std"] < collapse_std_threshold:
        print(
            f"WARNING: gate appears collapsed (min_std={final_diag['min_std']:.4f} < {collapse_std_threshold}). "
            "Consider raising lr, more epochs, or different pooling."
        )
    return gate


# ---------------------------------------------------------------------------
# X-LoRA classifier training (v2)
# ---------------------------------------------------------------------------


def _classifier_l2_norm(model) -> float:
    """L2 norm of every parameter whose name contains 'xlora'.

    This is a coarse fingerprint we use to detect (a) silent state-dict load
    failures (norm matches random init), and (b) that the optimizer actually
    moved the weights during training (post-norm differs from pre-norm).
    """
    total = 0.0
    n_tensors = 0
    for name, param in model.named_parameters():
        if "xlora" in name.lower():
            total += float(param.detach().float().pow(2).sum().item())
            n_tensors += 1
    return math.sqrt(total) if n_tensors > 0 else float("nan")


def _xlora_state_dict(model) -> dict:
    """Filter the model state dict to xlora-only keys (matches notebook 03)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "xlora" in k.lower()}


def _load_xlora_records(max_per_cluster: int = 512, total_cap: int = 2048, seed: int = 42) -> list[dict]:
    """Same balanced sampling as notebook 03 cell 5."""
    records: list[dict] = []
    for cid in range(4):
        path = SPLITS_DIR / f"cluster_{cid}_train.jsonl"
        with path.open() as f:
            cluster_records = [json.loads(line) for line in f if line.strip()]
        random.Random(seed + cid).shuffle(cluster_records)
        records.extend(cluster_records[:max_per_cluster])
    random.Random(seed).shuffle(records)
    return records[:total_cap]


def train_xlora_classifier_v2(
    xlora_model,
    tokenizer,
    output_path: Path = OUTPUTS_DIR / "xlora_classifier_v2.pt",
    epochs: int = 4,
    lr: float = 5e-4,
    max_length: int = 512,
    max_per_cluster: int = 512,
    total_cap: int = 2048,
    entropy_lambda: float = 0.05,
    log_every: int = 50,
) -> dict:
    """Retrain the X-LoRA classifier with anti-collapse entropy regularization.

    Differences vs notebook 03 cell 5:
    - 4 epochs (was 1) at lr 5e-4 (was 1e-4).
    - Loss is `CE(pooled_logits, labels) + lambda_H * mean_(t,l) H(softmax(raw_(t,l,:)))`.
      With positive lambda we minimize per-(token, layer) entropy, which fights
      the uniform-softmax fixed point that the original X-LoRA classifier got
      stuck at. CE supplies the directional signal (correct cluster); the
      entropy term supplies peakedness.
    - Logs classifier L2 norm before/after training and asserts it changed,
      catching silent failures.
    - Saves only xlora-prefixed keys, matching the format `inference.load_xlora_model`
      consumes.

    Returns a dict of training stats.
    """
    xlora_core = xlora_model.base_model
    classifier = xlora_core.internal_xlora_classifier.float()
    device = next(xlora_model.parameters()).device

    for param in xlora_model.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = True

    pre_norm = _classifier_l2_norm(xlora_model)
    print(f"Pre-train classifier L2 norm: {pre_norm:.4f}")

    records = _load_xlora_records(max_per_cluster=max_per_cluster, total_cap=total_cap)
    print(f"X-LoRA training records: {len(records)}")

    def collate(batch_records):
        prompts = [build_context_prompt(record, tokenizer) for record in batch_records]
        labels = torch.tensor([int(record["cluster_id"]) for record in batch_records], dtype=torch.long)
        batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batch["labels"] = labels
        return batch

    loader = DataLoader(records, batch_size=1, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)

    xlora_model.train()
    classifier.train()

    step_global = 0
    last_loss = float("nan")
    last_acc = float("nan")
    for epoch in range(epochs):
        epoch_running = 0.0
        epoch_correct = 0
        epoch_total = 0
        for step, batch in enumerate(tqdm(loader, desc=f"X-LoRA classifier v2 epoch {epoch + 1}/{epochs}"), start=1):
            step_global += 1
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = xlora_core.lora_model.model(
                    **batch,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden = outputs.hidden_states[-1].float()
            raw = classifier.layers(hidden)
            raw = raw.reshape(hidden.shape[0], hidden.shape[1], classifier.n_layers, classifier.n_classes)

            mask = batch["attention_mask"].float().unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
            pooled_logits = (raw * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)
            ce = F.cross_entropy(pooled_logits, labels)

            # Per-(token, layer) softmax probabilities and their entropy.
            per_pos_probs = F.softmax(raw, dim=-1)  # (B, T, L, K)
            per_pos_log = per_pos_probs.clamp(min=1e-9).log()
            per_pos_entropy = -(per_pos_probs * per_pos_log).sum(dim=-1)  # (B, T, L)
            entropy_mask = batch["attention_mask"].float().unsqueeze(-1)  # (B, T, 1)
            denom = entropy_mask.sum() * raw.shape[2]
            mean_entropy = (per_pos_entropy * entropy_mask).sum() / denom.clamp(min=1.0)

            loss = ce + entropy_lambda * mean_entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_running += loss.item()
            preds = pooled_logits.argmax(dim=-1)
            epoch_correct += int((preds == labels).sum().item())
            epoch_total += int(labels.numel())

            if step % log_every == 0:
                acc = (preds == labels).float().mean().item()
                last_loss = loss.item()
                last_acc = acc
                print(
                    f"epoch={epoch + 1} step={step} loss={loss.item():.4f} "
                    f"ce={ce.item():.4f} ent={mean_entropy.item():.4f} acc={acc:.3f}"
                )

        epoch_acc = epoch_correct / max(1, epoch_total)
        print(
            f"Epoch {epoch + 1} done: avg_loss={epoch_running / max(1, len(loader)):.4f} "
            f"epoch_acc={epoch_acc:.3f}"
        )

    post_norm = _classifier_l2_norm(xlora_model)
    delta = abs(post_norm - pre_norm) / max(pre_norm, 1e-9)
    print(f"Post-train classifier L2 norm: {post_norm:.4f} (relative delta {delta:.4%})")
    assert delta > 1e-3, (
        f"X-LoRA classifier weights barely moved (delta={delta:.4%}). Check that "
        "`classifier.parameters()` actually had requires_grad=True and the optimizer ran."
    )

    state = _xlora_state_dict(xlora_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    print(f"Saved {len(state)} xlora keys to {output_path}")
    return {
        "pre_l2": pre_norm,
        "post_l2": post_norm,
        "rel_delta": delta,
        "n_keys_saved": len(state),
        "last_step_loss": last_loss,
        "last_step_acc": last_acc,
        "global_steps": step_global,
    }
