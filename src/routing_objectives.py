"""Experiment 3 — supervision granularity vs routing granularity (X-LoRA).

The question
------------
X-LoRA makes one routing decision per (token, layer): with T≈80 tokens and
L=32 layers that is ~10,000 decisions per example. v1 supervised all of them
with ONE pooled cross-entropy label per conversation. The chain rule then
dilutes the gradient reaching any single position by 1/(T·L) ≈ 1/2560, and the
uniform-softmax fixed point wins — which is exactly the collapse v1 observed
(routing entropy = log 4 at machine precision; see XLORA_OBJECTIVE_MISMATCH.html
for the full derivation).

This experiment makes that argument *empirical* by training the same X-LoRA
classifier under three objectives that differ only in supervision granularity:

  pooled_ce    — CE on logits mean-pooled over (token, layer). v1's objective;
                 expected to collapse to uniform.
  per_token_ce — CE applied at EVERY real (token, layer) position against the
                 conversation's cluster label. Same label, no gradient dilution.
  ntp          — next-token prediction through the active mixture: the gold
                 response's LM loss back-propagated to the classifier only.
                 This is what the X-LoRA paper actually trains with; it is the
                 only objective that lets routing earn its keep on the real
                 task rather than on a proxy label.

After training, every router is measured the same way: routing entropy,
gold-cluster mass, routing accuracy (argmax of the mean routing distribution),
and downstream gold-response perplexity.

PEFT-internals notes (verified against peft==0.18.1 source)
-----------------------------------------------------------
* `XLoraLinearLayer.forward(..., scalings=None)` applies ALL adapters at FULL
  weight — it does NOT fall back to the base model. PEFT's own scalings pass
  suppresses the adapters by injecting dummy scalings equal to
  `config.scaling_pass_value` (default 0.0) via forward pre-hooks.
  `_base_hidden_states` below replicates that mechanism; calling the inner
  model directly (as the never-run `gating_v2.py` did) would silently compute
  features under a 4-adapter sum.
* In the real forward, the classifier is invoked OUTSIDE `torch.no_grad()` and
  its scalings are injected into every LoRA layer, so `loss.backward()` on the
  LM loss reaches the classifier parameters. That makes the `ntp` objective a
  standard forward + backward.
* The classifier is created in the base model's dtype (fp16 here). Training
  fp16 master weights with Adam is unstable, so `prepare_classifier` keeps the
  parameters in fp32 and installs a dtype-safe `forward` (a faithful
  reimplementation of `XLoraClassifier.forward` from 0.18.1) that casts hidden
  states up and scalings back down.
"""

from __future__ import annotations

import contextlib
import math
import types
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_core import corpus_ppl, encode_prompt_response, per_example_nll, save_json
from paths import OUTPUTS_DIR
from prompts import build_vanilla_prompt, gold_response

EXPERIMENT_DIR = OUTPUTS_DIR / "experiments" / "routing"
OBJECTIVES = ("pooled_ce", "per_token_ce", "ntp")


def _classifier(xlora_model):
    try:
        return xlora_model.base_model.internal_xlora_classifier
    except AttributeError as exc:
        raise RuntimeError(
            "Could not find internal_xlora_classifier — is this a PEFT X-LoRA model "
            "from inference.load_xlora_model?"
        ) from exc


# ---------------------------------------------------------------------------
# Classifier preparation (fp32 params + dtype-safe forward)
# ---------------------------------------------------------------------------


def prepare_classifier(xlora_model) -> None:
    """Cast the routing classifier to fp32 and make its forward dtype-safe.

    Idempotent. The replacement forward replicates peft 0.18.1's
    `XLoraClassifier.forward` exactly, plus two casts: hidden states up to the
    classifier dtype, produced scalings back down to the base-model dtype so
    they can be injected into fp16 LoRA layers.
    """
    clf = _classifier(xlora_model)
    if getattr(clf, "_desa_prepared", False):
        return
    base_dtype = clf.dtype  # recorded by peft at construction (the base model dtype)
    clf.float()

    def dtype_safe_forward(self, result, input_ids=None, inputs_embeds=None, *args, **kwargs):
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
        else:
            batch_size, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        hidden_state = result.hidden_states[-1]
        logits = self.layers(hidden_state.to(torch.float32))
        if not self.config.layerwise_scalings:
            logits = logits.unsqueeze(2).expand(-1, -1, self.n_layers, -1)
        scalings = logits.reshape(batch_size, seq_len, self.n_layers, self.n_classes)
        if self.config.enable_softmax:
            scalings = self.softmax(scalings)
        if self.scalings_logging:
            self.log_scalings.append(scalings)
        return scalings.to(base_dtype)

    clf.forward = types.MethodType(dtype_safe_forward, clf)
    clf._desa_prepared = True
    print(f"Classifier prepared: params fp32, scalings emitted as {base_dtype}.")


def freeze_all_but_classifier(xlora_model) -> int:
    """Freeze everything except the routing classifier. Returns trainable param count."""
    for param in xlora_model.parameters():
        param.requires_grad = False
    clf = _classifier(xlora_model)
    n = 0
    for param in clf.parameters():
        param.requires_grad = True
        n += param.numel()
    print(f"Trainable classifier parameters: {n:,}")
    return n


def classifier_l2_norm(xlora_model) -> float:
    """Fingerprint to detect silent load failures / optimizers that never stepped."""
    total = sum(
        float(p.detach().float().pow(2).sum()) for n, p in xlora_model.named_parameters() if "xlora" in n.lower()
    )
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# Base hidden states (the scalings pass, replicated faithfully)
# ---------------------------------------------------------------------------


def _inject_scalings_hook(target, args, kwargs, scalings):
    kwargs["scalings"] = scalings
    return args, kwargs


def _base_hidden_states(xlora_model, batch: dict) -> torch.Tensor:
    """Last hidden state of the (adapter-suppressed) base model — what the
    routing classifier is supposed to read.

    Replicates `XLoraModel._enable_peft_forward_hooks`'s scalings pass: inject
    dummy scalings (scaling_pass_value, default 0.0) into every LoRA layer so
    the adapters contribute nothing, then run the inner model once.
    """
    from peft.tuners.lora.layer import LoraLayer

    core = xlora_model.base_model
    clf = _classifier(xlora_model)
    dummy = clf.make_dummy_scalings(input_ids=batch["input_ids"])
    handles = []
    try:
        for module in xlora_model.modules():
            if isinstance(module, LoraLayer):
                handles.append(
                    module.register_forward_pre_hook(
                        partial(_inject_scalings_hook, scalings=dummy), with_kwargs=True
                    )
                )
        with torch.no_grad():
            out = core.lora_model.model(**batch, output_hidden_states=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return out.hidden_states[-1]


# ---------------------------------------------------------------------------
# Training under the three objectives
# ---------------------------------------------------------------------------


def _classifier_logits(clf, hidden: torch.Tensor) -> torch.Tensor:
    """Raw pre-softmax routing logits, shape (B, T, n_layers, n_classes), fp32."""
    logits = clf.layers(hidden.to(torch.float32))
    batch, seq = hidden.shape[0], hidden.shape[1]
    if not clf.config.layerwise_scalings:
        logits = logits.unsqueeze(2).expand(-1, -1, clf.n_layers, -1)
    return logits.reshape(batch, seq, clf.n_layers, clf.n_classes)


def train_router(
    xlora_model,
    tokenizer,
    records: list[dict],
    objective: str,
    epochs: int = 2,
    lr: float = 5e-4,
    max_length: int = 512,
    out_path: Path | None = None,
    log_every: int = 50,
) -> dict:
    """Train the X-LoRA classifier under one objective; save xlora-only weights.

    `records` are conversation dicts carrying `utterances` and `cluster_id`
    (e.g. from `gating.load_gating_records`). Batch size is fixed at 1: the
    `ntp` objective back-propagates through a full 7B forward, and identical
    batching across objectives keeps the comparison clean.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    from tqdm.auto import tqdm

    prepare_classifier(xlora_model)
    freeze_all_but_classifier(xlora_model)
    clf = _classifier(xlora_model)
    device = next(xlora_model.parameters()).device
    optimizer = torch.optim.AdamW((p for p in clf.parameters() if p.requires_grad), lr=lr)

    pre_norm = classifier_l2_norm(xlora_model)
    print(f"[{objective}] pre-train classifier L2 norm: {pre_norm:.4f}")

    xlora_model.train()
    history: list[dict] = []
    step = 0
    for epoch in range(epochs):
        for record in tqdm(records, desc=f"{objective} epoch {epoch + 1}/{epochs}"):
            step += 1
            label = torch.tensor([int(record["cluster_id"])], device=device)
            prompt = build_vanilla_prompt(record, tokenizer)

            if objective == "ntp":
                encoded = encode_prompt_response(
                    tokenizer, prompt, gold_response(record), device, max_length
                )
                if encoded is None:
                    continue
                labels = encoded.pop("labels")
                outputs = xlora_model(**encoded, labels=labels)
                loss = outputs.loss
                logged = {"lm_loss": float(loss.item())}
            else:
                batch = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=max_length
                ).to(device)
                hidden = _base_hidden_states(xlora_model, batch)
                logits = _classifier_logits(clf, hidden)  # (1, T, L, K)
                mask = batch["attention_mask"].bool()  # (1, T)
                if objective == "pooled_ce":
                    # v1's objective: ONE label supervises the (T, L)-mean of the
                    # logits. Each position's gradient is scaled by 1/(T*L).
                    pooled = logits[mask].mean(dim=(0, 1), keepdim=False).unsqueeze(0)  # (1, K)
                    loss = F.cross_entropy(pooled, label)
                else:  # per_token_ce
                    # Same label, but applied independently at every real
                    # (token, layer) position — no dilution.
                    flat = logits[mask].reshape(-1, clf.n_classes)  # (T_real*L, K)
                    loss = F.cross_entropy(flat, label.expand(flat.shape[0]))
                logged = {"ce_loss": float(loss.item())}

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                logged.update(step=step, epoch=epoch + 1)
                history.append(logged)
                print(f"[{objective}] step={step} {logged}")

    post_norm = classifier_l2_norm(xlora_model)
    rel_delta = abs(post_norm - pre_norm) / max(pre_norm, 1e-9)
    print(f"[{objective}] post-train classifier L2 norm: {post_norm:.4f} (rel delta {rel_delta:.4%})")
    assert rel_delta > 1e-4, (
        "Classifier weights barely moved — the optimizer never effectively stepped. "
        "Check requires_grad flags and that the loss actually depends on the classifier."
    )

    if out_path is not None:
        save_router_state(xlora_model, out_path)
    return {
        "objective": objective,
        "pre_l2": pre_norm,
        "post_l2": post_norm,
        "rel_delta": rel_delta,
        "steps": step,
        "history": history,
    }


def save_router_state(xlora_model, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in xlora_model.state_dict().items() if "xlora" in k.lower()}
    torch.save(state, path)
    print(f"Saved {len(state)} xlora keys to {path}")
    return path


def load_router_state(xlora_model, path: Path) -> None:
    """Load saved router weights, failing loudly on key mismatches (the v1
    classifier silently stayed at random init because of strict=False)."""
    prepare_classifier(xlora_model)  # fp32 params before loading fp32 state
    state = torch.load(Path(path), map_location="cpu")
    missing, unexpected = xlora_model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Router state has {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    missing_xlora = [k for k in missing if "xlora" in k.lower()]
    if missing_xlora:
        raise RuntimeError(f"Router state is missing xlora keys, e.g. {missing_xlora[:3]}")
    print(f"Loaded {len(state)} xlora keys from {path}")


# ---------------------------------------------------------------------------
# Router measurement (identical for every objective)
# ---------------------------------------------------------------------------


def evaluate_router(
    xlora_model,
    tokenizer,
    examples: list[dict],
    max_length: int = 512,
    compute_ppl: bool = True,
) -> dict:
    """Routing entropy / alignment / accuracy, and optionally response PPL.

    Reads the actual scalings the model uses at inference via PEFT's scalings
    log (one forward per example over the conversation prompt).
    """
    from tqdm.auto import tqdm

    xlora_model.eval()
    device = next(xlora_model.parameters()).device
    per_example = []
    for example in tqdm(examples, desc="router stats"):
        batch = tokenizer(
            build_vanilla_prompt(example, tokenizer),
            return_tensors="pt", truncation=True, max_length=max_length,
        ).to(device)
        xlora_model.clear_scalings_log()
        xlora_model.enable_scalings_logging()
        with torch.no_grad():
            xlora_model(**batch)
        xlora_model.disable_scalings_logging()
        latest = xlora_model.get_latest_scalings()
        if latest is None:
            continue
        scal = latest.detach().float().cpu().numpy()[0]  # (T, L, K)
        mean_alpha = scal.mean(axis=(0, 1))
        mean_alpha = mean_alpha / mean_alpha.sum()
        per_pos = np.clip(scal, 1e-9, 1.0)
        entropy = float(-(per_pos * np.log(per_pos)).sum(axis=-1).mean())
        per_example.append(
            {
                "example_id": example.get("example_id"),
                "cluster_id": int(example["cluster_id"]),
                "mean_alpha": mean_alpha.tolist(),
                "entropy": entropy,
                "pred_cluster": int(mean_alpha.argmax()),
            }
        )

    entropies = np.array([r["entropy"] for r in per_example])
    gold_mass = np.array([r["mean_alpha"][r["cluster_id"]] for r in per_example])
    accuracy = float(np.mean([r["pred_cluster"] == r["cluster_id"] for r in per_example]))
    out = {
        "mean_entropy": float(entropies.mean()),
        "std_entropy": float(entropies.std()),
        "max_entropy": float(math.log(4)),
        "gold_cluster_mass": float(gold_mass.mean()),
        "routing_accuracy": accuracy,
        "n_examples": len(per_example),
        "per_example": per_example,
    }
    if compute_ppl:
        records = per_example_nll(
            xlora_model, tokenizer, examples,
            prompt_builder=build_vanilla_prompt, max_length=max_length, desc="xlora ppl",
        )
        out["response_ppl"] = corpus_ppl(records)
        out["nll_records"] = records
    return out


# ---------------------------------------------------------------------------
# The uniform-consistency bug check
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def force_uniform_scalings(xlora_model):
    """Make the classifier emit exactly uniform (1/K) scalings.

    With uniform scalings, X-LoRA's effective update is 1/4 · Σ_k ΔW_k — the
    SAME model as `apply_weighted_adapters(model, [0.25]*4)` on the plain
    multi-adapter path. The two implementations must therefore produce nearly
    identical perplexity on the same examples and base precision. v1 reported
    30.7 vs 7272.5 for these two supposedly-identical models, which proves an
    integration bug somewhere; this check localizes it.
    """
    clf = _classifier(xlora_model)
    original = clf.forward

    def uniform_forward(result, *args, **kwargs):
        out = original(result, *args, **kwargs)
        return torch.full_like(out, 1.0 / clf.n_classes)

    clf.forward = uniform_forward
    try:
        yield
    finally:
        clf.forward = original


def compare_uniform_paths(records_blend: list[dict], records_xlora: list[dict]) -> dict:
    """Summarize the consistency check from two aligned per-example NLL runs."""
    ppl_blend = corpus_ppl(records_blend)
    ppl_xlora = corpus_ppl(records_xlora)
    ratio = ppl_xlora / ppl_blend
    verdict = (
        "CONSISTENT — the X-LoRA integration is trustworthy."
        if 0.9 <= ratio <= 1.1
        else "INCONSISTENT — same effective weights, different PPL: the X-LoRA forward "
             "path has a bug (scaling semantics, dtype, or prompt handling). Do not "
             "trust X-LoRA metrics until this is resolved."
    )
    result = {
        "ppl_uniform_blend_set_adapters": ppl_blend,
        "ppl_uniform_xlora": ppl_xlora,
        "ratio": ratio,
        "verdict": verdict,
    }
    print(f"uniform blend (set_adapters): PPL={ppl_blend:.2f}")
    print(f"uniform X-LoRA (forced):      PPL={ppl_xlora:.2f}")
    print(f"ratio={ratio:.3f} -> {verdict}")
    return result


def save_routing_results(results: dict, path: Path = EXPERIMENT_DIR / "routing_objectives.json") -> Path:
    slim = {}
    for objective, payload in results.items():
        slim[objective] = {k: v for k, v in payload.items() if k not in ("per_example", "nll_records")}
    return save_json(slim, path)
