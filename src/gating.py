"""Gating heads and training helpers for DESA."""

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

from paths import CLUSTER_DIR, OUTPUTS_DIR, SPLITS_DIR


class TurnLevelGatingHead(nn.Module):
    """Mean-pool hidden states and predict one adapter-weight vector per turn."""

    def __init__(self, hidden_dim: int, num_adapters: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_adapters = num_adapters
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_adapters),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        logits = self.gate(pooled)
        return F.softmax(logits, dim=-1)

    def logits(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        return self.gate(pooled)


class VADGatingHead(nn.Module):
    """Stretch goal: explicit VAD-conditioned gating with emotional inertia."""

    def __init__(self, num_adapters: int = 4, momentum: float = 0.3):
        super().__init__()
        self.num_adapters = num_adapters
        self.momentum = momentum
        self.adapter_prototypes = nn.Parameter(torch.randn(num_adapters, 3))

    def forward(self, vad_state: torch.Tensor):
        vad_norm = F.normalize(vad_state, dim=-1)
        proto_norm = F.normalize(self.adapter_prototypes, dim=-1)
        return F.softmax(vad_norm @ proto_norm.T, dim=-1)

    def update_vad_state(self, current_vad: torch.Tensor, previous_state: torch.Tensor):
        return (1 - self.momentum) * current_vad + self.momentum * previous_state


def load_cluster_assignments(path: Path = CLUSTER_DIR / "cluster_assignments.json") -> dict[str, int]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {emotion: int(cid) for emotion, cid in data["emotion_to_cluster"].items()}


def load_gating_records(
    split: str = "train",
    max_per_cluster: int = 512,
    seed: int = 42,
) -> list[dict]:
    """Load balanced records for gating training."""
    rng = random.Random(seed)
    records: list[dict] = []
    suffix = "train" if split == "train" else "val"
    for cid in range(4):
        path = SPLITS_DIR / f"cluster_{cid}_{suffix}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run notebook 01 first.")
        with path.open(encoding="utf-8") as f:
            cluster_records = [json.loads(line) for line in f if line.strip()]
        rng.shuffle(cluster_records)
        records.extend(cluster_records[:max_per_cluster])
    rng.shuffle(records)
    return records


def build_context_prompt(example: dict, tokenizer, max_turns: int | None = None) -> str:
    """Format conversation history before the final assistant response."""
    utterances = [str(utt).strip() for utt in example.get("utterances", []) if str(utt).strip()]
    if len(utterances) > 1:
        utterances = utterances[:-1]
    if max_turns:
        utterances = utterances[-max_turns:]

    messages = [
        {"role": "user" if idx % 2 == 0 else "assistant", "content": utterance}
        for idx, utterance in enumerate(utterances)
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        labeled = []
        for idx, utterance in enumerate(utterances):
            role = "User" if idx % 2 == 0 else "Assistant"
            labeled.append(f"{role}: {utterance}")
        return f"<s>[INST] {' '.join(labeled)} [/INST]"


def _collate(records: list[dict], tokenizer, max_length: int):
    prompts = [build_context_prompt(record, tokenizer) for record in records]
    labels = torch.tensor([int(record["cluster_id"]) for record in records], dtype=torch.long)
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    batch["labels"] = labels
    return batch


def train_turn_gate(
    base_model,
    tokenizer,
    output_path: Path = OUTPUTS_DIR / "turn_gate.pt",
    hidden_dim: int = 4096,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_length: int = 512,
    max_per_cluster: int = 512,
) -> TurnLevelGatingHead:
    """Train turn-level routing with CE against the gold VAD cluster."""
    device = next(base_model.parameters()).device
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    train_records = load_gating_records("train", max_per_cluster=max_per_cluster)
    val_records = load_gating_records("val", max_per_cluster=max(32, max_per_cluster // 8))
    gate = TurnLevelGatingHead(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr)

    def collate_fn(items):
        return _collate(items, tokenizer, max_length)

    train_loader = DataLoader(train_records, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_records, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    best_val = math.inf
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        gate.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Turn gate epoch {epoch + 1}/{epochs}"):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = base_model(**batch, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
            logits = gate.logits(hidden_states, batch["attention_mask"])
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        val_loss = evaluate_turn_gate(base_model, tokenizer, gate, val_loader, device)
        print(
            f"Epoch {epoch + 1}: train_loss={running / max(1, len(train_loader)):.4f} "
            f"val_loss={val_loss:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(gate.state_dict(), output_path)
            print(f"Saved best turn gate to {output_path}")

    gate.load_state_dict(torch.load(output_path, map_location=device))
    return gate


def evaluate_turn_gate(base_model, tokenizer, gate, loader, device) -> float:
    """Validation CE for the turn-level gate."""
    del tokenizer
    gate.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = base_model(**batch, output_hidden_states=True)
            logits = gate.logits(outputs.hidden_states[-1], batch["attention_mask"])
            total += F.cross_entropy(logits, labels).item()
    return total / max(1, len(loader))
