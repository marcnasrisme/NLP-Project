"""Experiment 2 — is the routing signal even present in the features?

The question
------------
The v1 turn gate collapsed to an input-independent prior. There are two very
different explanations, and they demand different fixes:

  (a) **Optimization failure**: the cluster IS linearly decodable from the
      frozen hidden states, but mean-pooling + 2 epochs at lr 1e-4 failed to
      find it. Fix: better pooling / longer training (the v2 gate).
  (b) **Signal absence**: the pooled feature simply does not carry the
      emotion-quadrant information. Fix: nothing — no gate architecture on
      these features can work, and the project's conclusion changes.

The experiment
--------------
Extract frozen base-model features for balanced conversations (both last-token
and mean pooling, so the two pooling strategies are compared on equal footing),
then fit the simplest possible probes:

  - majority-class baseline (sanity floor, = 0.25 for balanced data)
  - multinomial logistic regression  (the LINEAR ceiling)
  - small MLP probe                  (matches the gate head's capacity)

The logistic-regression accuracy is the number that settles the (a)-vs-(b)
question: any trained gate should land between it and the MLP probe. If the
probes sit near chance, v1's collapse was inevitable.

GPU is only needed for `extract_features`; the probes run anywhere from the
saved .npz (download it from Colab and iterate locally).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from paths import OUTPUTS_DIR

EXPERIMENT_DIR = OUTPUTS_DIR / "experiments" / "probe"


# ---------------------------------------------------------------------------
# GPU part: feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    base_model,
    tokenizer,
    records: list[dict],
    max_length: int = 512,
    batch_size: int = 8,
    out_path: Path | None = None,
) -> dict:
    """Frozen last-hidden-layer features for each conversation context.

    For every record (which must carry `cluster_id`), builds the same
    conversation prompt the gate sees at inference, runs the frozen base model
    once, and stores BOTH pooled views:

      - `last`: hidden state of the final real token (causal models summarize
        the full prefix here — what the v2 gate uses)
      - `mean`: attention-masked mean over all tokens (what the collapsed v1
        gate used)

    Saves arrays to `out_path` (.npz) when given. Returns the dict of arrays.
    """
    from tqdm.auto import tqdm

    from gating import build_context_prompt

    base_model.eval()
    device = next(base_model.parameters()).device

    last_feats, mean_feats, labels = [], [], []
    for start in tqdm(range(0, len(records), batch_size), desc="extract features"):
        batch_records = records[start: start + batch_size]
        prompts = [build_context_prompt(record, tokenizer) for record in batch_records]
        batch = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            hidden = base_model(**batch, output_hidden_states=True).hidden_states[-1]
        mask = batch["attention_mask"]
        # mean pool (masked) — correct for either padding side
        m = mask.unsqueeze(-1).to(hidden.dtype)
        mean_pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        # last real token, robust to LEFT or RIGHT padding. `cumsum(mask).argmax`
        # returns the first position holding the running max (= the final 1),
        # which is the last real token regardless of where the pads sit. The
        # naive `mask.sum()-1` is only correct for right padding; the eval
        # tokenizer pads LEFT, so that formula reads the wrong (often pad) token.
        last_idx = mask.long().cumsum(dim=1).argmax(dim=1)
        idx = torch.arange(hidden.size(0), device=device)
        last_pooled = hidden[idx, last_idx]

        last_feats.append(last_pooled.float().cpu().numpy())
        mean_feats.append(mean_pooled.float().cpu().numpy())
        labels.extend(int(record["cluster_id"]) for record in batch_records)

    arrays = {
        "last": np.concatenate(last_feats, axis=0),
        "mean": np.concatenate(mean_feats, axis=0),
        "labels": np.asarray(labels, dtype=np.int64),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **arrays)
        print(f"Wrote {out_path} ({arrays['last'].shape[0]} examples, dim {arrays['last'].shape[1]})")
    return arrays


# ---------------------------------------------------------------------------
# CPU part: the probes
# ---------------------------------------------------------------------------


def run_probes(
    train_arrays: dict,
    test_arrays: dict,
    seed: int = 42,
    out_path: Path | None = None,
) -> dict:
    """Fit majority / logistic / MLP probes on each pooling view.

    `train_arrays` / `test_arrays` are the dicts produced by
    `extract_features` (or `dict(np.load(path))`). Returns a nested result
    dict: results[pooling][probe] -> {"accuracy", "macro_f1", "confusion"}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    results: dict = {"n_train": int(train_arrays["labels"].shape[0]), "n_test": int(test_arrays["labels"].shape[0])}
    y_train, y_test = train_arrays["labels"], test_arrays["labels"]

    majority = int(np.bincount(y_train).argmax())
    majority_acc = float((y_test == majority).mean())

    for pooling in ("last", "mean"):
        scaler = StandardScaler().fit(train_arrays[pooling])
        x_train = scaler.transform(train_arrays[pooling])
        x_test = scaler.transform(test_arrays[pooling])

        probes = {
            "logistic": LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
            # 1024 hidden units mirrors the gate head's bottleneck (4096 -> 1024 -> 4)
            "mlp": MLPClassifier(hidden_layer_sizes=(1024,), max_iter=300, random_state=seed),
        }
        results[pooling] = {"majority": {"accuracy": majority_acc}}
        for name, probe in probes.items():
            probe.fit(x_train, y_train)
            preds = probe.predict(x_test)
            results[pooling][name] = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "macro_f1": float(f1_score(y_test, preds, average="macro")),
                "confusion": confusion_matrix(y_test, preds).tolist(),
            }
            print(
                f"[{pooling:4s}] {name:8s}: acc={results[pooling][name]['accuracy']:.3f} "
                f"macro_f1={results[pooling][name]['macro_f1']:.3f}"
            )
    print(f"majority baseline: acc={majority_acc:.3f} (chance for balanced 4-way = 0.25)")

    if out_path is not None:
        from eval_core import save_json

        save_json(results, out_path)
    return results


def interpret_probe_results(results: dict, strong_signal: float = 0.55, weak_signal: float = 0.35) -> str:
    """Plain-language verdict implementing the experiment's decision rule."""
    best_linear = max(results[p]["logistic"]["accuracy"] for p in ("last", "mean"))
    best_pooling = max(("last", "mean"), key=lambda p: results[p]["logistic"]["accuracy"])
    lines = [f"Best linear probe: {best_linear:.3f} accuracy ({best_pooling} pooling); chance = 0.25."]
    if best_linear >= strong_signal:
        lines.append(
            "VERDICT (a): the cluster is decodable from frozen features. The v1 gate "
            "collapse was an optimization/pooling failure — a properly trained gate "
            "(last-token pooling, longer schedule) should recover this accuracy, and "
            "routing experiments are worth running."
        )
    elif best_linear <= weak_signal:
        lines.append(
            "VERDICT (b): the routing signal is essentially absent from these features. "
            "No gate architecture on frozen pooled hidden states can route reliably; "
            "the project's story becomes WHY the signal is absent (and the "
            "specialization matrix likely shows there was nothing to route anyway)."
        )
    else:
        lines.append(
            "VERDICT: partial signal. The gate can do better than chance but will be "
            "noisy; expect modest routing gains at best. Compare per-class F1 to see "
            "which quadrants are confusable (positive-low vs positive-high arousal is "
            "the usual suspect)."
        )
    delta = results["last"]["logistic"]["accuracy"] - results["mean"]["logistic"]["accuracy"]
    lines.append(f"Last-token minus mean-pool linear accuracy: {delta:+.3f} "
                 "(positive = the v2 pooling fix is justified by the features themselves).")
    return "\n".join(lines)
