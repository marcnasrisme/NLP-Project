"""Offline aggregation and statistics for DESA experiment results.

Runs anywhere — only numpy / pandas / matplotlib, no torch or GPU. All heavy
notebooks persist per-example JSONL; this module turns those files into the
tables and figures that go in the report, with uncertainty attached:

- `summary_table`     per-system metrics with bootstrap 95% CIs
- `paired_comparisons` paired-bootstrap ΔNLL tests between systems
- `per_cluster_ppl`   where each system wins/loses across the four quadrants
- `routing_table`     entropy / gold-mass / accuracy for routed systems

Statistical conventions:
- PPL CIs: nonparametric bootstrap over examples of corpus PPL
  (exp(Σnll/Σtokens) recomputed per resample).
- Emotion accuracy CIs: bootstrap over per-example 0/1 matches.
- Paired tests: bootstrap of the mean per-example NLL/token difference;
  pairing on example_id removes between-example variance, which is large for
  dialogue data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(path: str | Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8") as f:
        frame = pd.DataFrame(json.loads(line) for line in f if line.strip())
    if "sum_nll" in frame and "n_tokens" in frame:
        frame["nll_per_token"] = frame["sum_nll"] / frame["n_tokens"]
    return frame


# ---------------------------------------------------------------------------
# Bootstrap machinery
# ---------------------------------------------------------------------------


def _bootstrap_ci(values_fn, n_items: int, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Generic bootstrap CI: `values_fn(idx)` maps a resample index array to a statistic."""
    rng = np.random.default_rng(seed)
    stats = np.array([values_fn(rng.integers(0, n_items, size=n_items)) for _ in range(n_boot)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def corpus_ppl_with_ci(sum_nll: np.ndarray, n_tokens: np.ndarray, n_boot: int = 5000, seed: int = 0):
    point = math.exp(sum_nll.sum() / n_tokens.sum())
    low, high = _bootstrap_ci(
        lambda idx: math.exp(sum_nll[idx].sum() / n_tokens[idx].sum()), len(sum_nll), n_boot, seed
    )
    return point, low, high


def accuracy_with_ci(matches: np.ndarray, n_boot: int = 5000, seed: int = 0):
    matches = matches[~np.isnan(matches)]
    if len(matches) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(matches.mean())
    low, high = _bootstrap_ci(lambda idx: matches[idx].mean(), len(matches), n_boot, seed)
    return point, low, high


def _distinct_n(texts, n: int) -> float:
    ngrams = []
    for text in texts:
        tokens = str(text).split()
        ngrams.extend(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def summary_table(frame: pd.DataFrame, n_boot: int = 5000) -> pd.DataFrame:
    """One row per system: PPL [CI], emotion accuracy [CI], distinct-1/2.

    Systems are grouped by information access — comparing across the
    informed/uninformed boundary is flagged in the report as unfair.
    """
    rows = []
    for (informed, system), group in frame.groupby(["informed", "system"]):
        scored = group.dropna(subset=["sum_nll"])
        ppl, ppl_lo, ppl_hi = corpus_ppl_with_ci(
            scored["sum_nll"].to_numpy(), scored["n_tokens"].to_numpy(), n_boot
        )
        match_vals = group["emotion_match"].astype(float).to_numpy()
        acc, acc_lo, acc_hi = accuracy_with_ci(match_vals, n_boot)
        rows.append(
            {
                "system": system,
                "informed": informed,
                "ppl": round(ppl, 2),
                "ppl_ci95": f"[{ppl_lo:.1f}, {ppl_hi:.1f}]",
                "emotion_acc": round(acc, 3),
                "emotion_acc_ci95": f"[{acc_lo:.3f}, {acc_hi:.3f}]",
                "distinct_1": round(_distinct_n(group["generation"], 1), 3),
                "distinct_2": round(_distinct_n(group["generation"], 2), 3),
                "n": len(group),
            }
        )
    out = pd.DataFrame(rows).sort_values(["informed", "ppl"]).set_index(["informed", "system"])
    return out


def paired_comparisons(
    frame: pd.DataFrame,
    pairs: list[tuple[str, str]],
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired bootstrap on per-example NLL/token for chosen (A, B) system pairs.

    Negative mean diff -> A fits the gold responses better than B. A pair is
    'significant' when the 95% CI excludes zero.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sys_a, sys_b in pairs:
        a = frame[frame.system == sys_a].dropna(subset=["nll_per_token"]).set_index("example_id")
        b = frame[frame.system == sys_b].dropna(subset=["nll_per_token"]).set_index("example_id")
        common = a.index.intersection(b.index)
        diffs = (a.loc[common, "nll_per_token"] - b.loc[common, "nll_per_token"]).to_numpy()
        idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
        boot = diffs[idx].mean(axis=1)
        p = 2.0 * min(float((boot >= 0).mean()), float((boot <= 0).mean()))
        rows.append(
            {
                "A": sys_a,
                "B": sys_b,
                "mean_dnll": round(float(diffs.mean()), 4),
                "ci95_low": round(float(np.percentile(boot, 2.5)), 4),
                "ci95_high": round(float(np.percentile(boot, 97.5)), 4),
                "p_value": round(min(1.0, p), 4),
                "winner": sys_a if diffs.mean() < 0 else sys_b,
                "significant": bool(np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0),
                "n_pairs": len(diffs),
            }
        )
    return pd.DataFrame(rows)


DEFAULT_PAIRS = [
    # Does learned routing beat the trivial alternatives? (the project's core claims)
    ("turn_gate", "uniform_blend"),
    ("turn_gate", "pooled_adapter"),
    ("turn_gate", "random_expert"),
    # Is there any routing headroom at all, even with a perfect router?
    ("oracle_expert", "uniform_blend"),
    ("oracle_expert", "pooled_adapter"),
    # Do adapters beat prompting at equal information?
    ("uniform_blend", "generic_empathy"),
    ("pooled_adapter", "base_chat"),
]


def per_cluster_ppl(frame: pd.DataFrame) -> pd.DataFrame:
    """Corpus PPL per (system, gold cluster) — shows where routing could matter."""
    rows = []
    for (system, cid), group in frame.dropna(subset=["sum_nll"]).groupby(["system", "cluster_id"]):
        rows.append(
            {
                "system": system,
                "cluster_id": int(cid),
                "ppl": round(math.exp(group["sum_nll"].sum() / group["n_tokens"].sum()), 2),
            }
        )
    return pd.DataFrame(rows).pivot(index="system", columns="cluster_id", values="ppl")


def routing_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Routing diagnostics for systems that produced alphas.

    std_entropy ~ 0 across examples = the gate is input-independent (collapsed),
    regardless of how good its average looks.
    """
    rows = []
    routed = frame[frame["alpha"].notna()]
    for system, group in routed.groupby("system"):
        alphas = np.array([np.asarray(a, dtype=float) for a in group["alpha"]])
        alphas = alphas / alphas.sum(axis=1, keepdims=True)
        clipped = np.clip(alphas, 1e-9, 1.0)
        entropies = -(clipped * np.log(clipped)).sum(axis=1)
        gold = group["cluster_id"].to_numpy().astype(int)
        rows.append(
            {
                "system": system,
                "mean_entropy": round(float(entropies.mean()), 4),
                "std_entropy": round(float(entropies.std()), 6),
                "max_entropy": round(math.log(alphas.shape[1]), 4),
                "gold_cluster_mass": round(float(alphas[np.arange(len(gold)), gold].mean()), 4),
                "routing_accuracy": round(float((alphas.argmax(axis=1) == gold).mean()), 4),
            }
        )
    return pd.DataFrame(rows).set_index("system")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_ppl_with_ci(summary: pd.DataFrame, path: Path | None = None):
    import matplotlib.pyplot as plt

    frame = summary.reset_index()
    frame[["lo", "hi"]] = frame["ppl_ci95"].str.strip("[]").str.split(",", expand=True).astype(float)
    frame = frame.sort_values("ppl")
    colors = ["tab:orange" if informed else "tab:blue" for informed in frame["informed"]]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(frame) + 2))
    ax.barh(frame["system"], frame["ppl"], color=colors,
            xerr=[frame["ppl"] - frame["lo"], frame["hi"] - frame["ppl"]], capsize=4)
    ax.set_xlabel("Gold-response corpus PPL (lower = better; bars: 95% bootstrap CI)")
    ax.set_title("Final evaluation — blue: uninformed, orange: informed (gold label leaked)")
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"Wrote {path}")
    return fig
