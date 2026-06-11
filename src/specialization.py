"""Experiment 1 — the specialization matrix.

The question
------------
Did the four emotion-quadrant adapters actually *specialize*, or did they all
learn the same generic "EmpatheticDialogues response style"? This is the
premise of the whole routing project, and v1 never tested it. Worse, v1
contained evidence against it: a collapsed (input-independent) blend of all
four experts achieved BETTER gold-response perplexity (30.7) than picking the
correct expert with the gold label (39.1) — impossible if the experts were
meaningfully specialized.

The experiment
--------------
Compute gold-response NLL on each cluster's held-out test examples under every
"routing condition":

    condition \\ test cluster   |  0    1    2    3
    ---------------------------+------------------
    base (no adapter)          |  .    .    .    .
    expert_0 .. expert_3       |  .    .    .    .   <- the 4x4 core
    uniform_blend (1/4 each)   |  .    .    .    .
    pooled (one adapter,       |  .    .    .    .
      all clusters' data)      |

How to read it
--------------
- **Diagonal advantage** = for test cluster j, how much better expert_j is than
  the average off-diagonal expert on that column. If ~0, the experts are
  interchangeable -> there is nothing to route -> gate collapse in v1 was the
  *rational* outcome, not a training bug.
- **pooled vs diagonal**: if one generalist adapter (same training budget as a
  single expert) matches the oracle-routed experts, specialization buys nothing
  even when routing is free and perfect.
- **uniform_blend vs diagonal**: if blending everything matches/beats the
  oracle expert, blending acts as a weight-space ensemble and routing precision
  is irrelevant.

Everything is computed per example and persisted as JSONL so CIs and paired
tests can be run offline (see `analysis.py`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval_core import corpus_ppl, per_example_nll, sample_test_examples, save_json, save_jsonl
from paths import OUTPUTS_DIR
from prompts import build_vanilla_prompt

EXPERIMENT_DIR = OUTPUTS_DIR / "experiments" / "specialization"
POOLED_ADAPTER_NAME = "pooled"


def cluster_test_examples(
    cluster_assignments: dict[str, int],
    n_per_cluster: int = 50,
    seed: int = 42,
) -> dict[int, list[dict]]:
    """Stratified test examples, grouped by gold cluster."""
    flat = sample_test_examples(cluster_assignments, n_per_cluster=n_per_cluster, seed=seed)
    grouped: dict[int, list[dict]] = {cid: [] for cid in range(4)}
    for example in flat:
        grouped[example["cluster_id"]].append(example)
    return grouped


def _conditions(model, include_pooled: bool):
    """Yield (condition_name, setup_fn, context_manager) triples.

    Each setup configures the multi-adapter PeftModel's routing state once per
    condition; conditions are input-independent here by design (the whole point
    is to measure each fixed routing choice against every test cluster).
    """
    from inference import adapters_disabled, apply_weighted_adapters

    conditions: list[tuple] = [("base", None, adapters_disabled)]
    for k in range(4):
        conditions.append(
            (f"expert_{k}", (lambda k=k: model.set_adapter(f"cluster_{k}")), None)
        )
    conditions.append(
        ("uniform_blend", (lambda: apply_weighted_adapters(model, np.full(4, 0.25))), None)
    )
    if include_pooled:
        conditions.append(
            ("pooled", (lambda: model.set_adapter(POOLED_ADAPTER_NAME)), None)
        )
    return conditions


def run_specialization_matrix(
    model,
    tokenizer,
    examples_by_cluster: dict[int, list[dict]],
    include_pooled: bool = False,
    out_dir: Path = EXPERIMENT_DIR,
    max_length: int = 512,
) -> dict:
    """Score every condition on every cluster's test examples.

    `model` is the multi-adapter PeftModel from `inference.load_multi_adapter_model`.
    If `include_pooled`, the pooled adapter must already be registered on it:

        model.load_adapter(str(OUTPUTS_DIR / "adapter_pooled" / "final"), adapter_name="pooled")

    Returns {"matrix": {condition: {cluster: corpus_ppl}}, ...} and writes
    per-example records + the matrix summary under `out_dir`.
    """
    out_dir = Path(out_dir)
    all_records: list[dict] = []
    matrix: dict[str, dict[int, float]] = {}

    for name, setup, ctx in _conditions(model, include_pooled):
        if setup is not None:
            setup()
        matrix[name] = {}
        for cid, examples in examples_by_cluster.items():
            records = per_example_nll(
                model,
                tokenizer,
                examples,
                prompt_builder=build_vanilla_prompt,
                context_manager=ctx,
                max_length=max_length,
                desc=f"{name} on cluster {cid}",
            )
            for record in records:
                record["condition"] = name
            all_records.extend(records)
            matrix[name][cid] = corpus_ppl(records)
            print(f"  {name:14s} | test cluster {cid}: PPL = {matrix[name][cid]:.2f}")

    summary = summarize_matrix(matrix)
    save_jsonl(all_records, out_dir / "per_example_nll.jsonl")
    save_json({"matrix": matrix, "summary": summary}, out_dir / "matrix.json")
    return {"matrix": matrix, "summary": summary, "records": all_records}


def summarize_matrix(matrix: dict[str, dict[int, float]]) -> dict:
    """Derive the decision-relevant statistics from the raw PPL matrix."""
    experts = [f"expert_{k}" for k in range(4)]
    diag = {j: matrix[f"expert_{j}"][j] for j in range(4)}
    off_diag_mean = {
        j: float(np.mean([matrix[e][j] for e in experts if e != f"expert_{j}"])) for j in range(4)
    }
    summary = {
        # < 1.0 means the matched expert is better than mismatched experts on
        # that cluster. ~1.0 across the board = no routable specialization.
        "diagonal_advantage_ratio": {
            j: diag[j] / off_diag_mean[j] for j in range(4)
        },
        "diagonal_ppl": diag,
        "off_diagonal_mean_ppl": off_diag_mean,
        "uniform_vs_diagonal_ratio": {
            j: matrix["uniform_blend"][j] / diag[j] for j in range(4)
        },
        "base_ppl": matrix.get("base"),
    }
    if "pooled" in matrix:
        summary["pooled_vs_diagonal_ratio"] = {j: matrix["pooled"][j] / diag[j] for j in range(4)}
    return summary


def interpret_matrix(summary: dict) -> str:
    """Plain-language verdict implementing the experiment's decision rule."""
    ratios = list(summary["diagonal_advantage_ratio"].values())
    mean_ratio = float(np.mean(ratios))
    lines = [
        f"Diagonal advantage ratio (matched-expert PPL / mismatched-expert mean PPL) "
        f"per cluster: {[round(r, 3) for r in ratios]} (mean {mean_ratio:.3f}; 1.0 = no specialization).",
    ]
    if mean_ratio >= 0.95:
        lines.append(
            "VERDICT: the experts are essentially interchangeable — there is no routable "
            "specialization. A perfect router could not beat a fixed blend, so the v1 gate "
            "collapse was the rational outcome, not a training bug. The project's headline "
            "becomes WHY quadrant fine-tuning fails to specialize on EmpatheticDialogues "
            "(shared response register, overlapping vocabularies, LoRA capacity)."
        )
    elif mean_ratio <= 0.80:
        lines.append(
            "VERDICT: strong specialization — matched experts clearly beat mismatched ones. "
            "Routing has real headroom; proceed to the routing-objective experiments and the "
            "final evaluation expecting the learned gate to matter."
        )
    else:
        lines.append(
            "VERDICT: modest specialization. Routing can help, but expect small downstream "
            "gains; report effect sizes with CIs rather than point estimates."
        )
    uniform = summary.get("uniform_vs_diagonal_ratio", {})
    if uniform:
        u = float(np.mean(list(uniform.values())))
        lines.append(
            f"Uniform blend / oracle-expert PPL ratio: {u:.3f} "
            f"({'the blend matches or beats the oracle — blending is an ensemble, routing precision is irrelevant' if u <= 1.02 else 'the oracle expert beats the blend — picking the right expert matters'})."
        )
    pooled = summary.get("pooled_vs_diagonal_ratio")
    if pooled:
        p = float(np.mean(list(pooled.values())))
        lines.append(
            f"Pooled adapter / oracle-expert PPL ratio: {p:.3f} "
            f"({'one generalist adapter matches the routed experts at equal budget — specialization buys nothing' if p <= 1.02 else 'experts beat the generalist — specialization carries real signal'})."
        )
    return "\n".join(lines)


def matrix_dataframe(matrix: dict[str, dict[int, float]]):
    """Rows = conditions, columns = test clusters. For display and the heatmap."""
    import pandas as pd

    df = pd.DataFrame(matrix).T
    df.columns = [f"test_cluster_{c}" for c in df.columns]
    return df


def plot_matrix(matrix: dict[str, dict[int, float]], path: Path | None = None):
    """Heatmap of log-PPL: the diagonal should 'light up' if experts specialized."""
    import matplotlib.pyplot as plt

    df = matrix_dataframe(matrix)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(df) + 2))
    values = np.log(df.values)
    im = ax.imshow(values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=30, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i, j]:.1f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, label="log PPL (lower = better fit)")
    ax.set_title("Specialization matrix: gold-response PPL by routing condition × test cluster")
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"Wrote {path}")
    return fig
