"""Research-grade evaluation primitives for DESA.

Design principles (each one fixes a specific v1 problem):

1. **Per-example records, not corpus scalars.** v1 reported a single corpus
   perplexity per system, so nothing could be said about uncertainty or
   significance. Every function here returns one record per test example;
   aggregation (means, CIs, paired tests) happens afterwards in
   `analysis.py` from persisted JSONL.

2. **Stratified, seeded test sampling.** v1 evaluated on the first N examples
   of the ED test split, whose emotion composition is arbitrary.
   `sample_test_examples` draws a seeded, cluster-balanced sample and filters
   out conversations that do not end on an assistant turn.

3. **Seeded generation.** v1 sampled at temperature 0.7 with no seed, so no
   number was reproducible. `generate` seeds per call.

4. **Prompt/likelihood consistency.** Likelihood scoring takes the exact
   prompt string produced by the system's own prompt builder (see
   `prompts.py`), so PPL always measures the system as it actually runs.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from prompts import gold_response, is_valid_eval_example

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Test-set construction
# ---------------------------------------------------------------------------


def _utterances_from_example(example: dict) -> list[str]:
    """Normalize an ED example to a flat list of utterance strings.

    Handles both the raw HF schema (`conversations`/`messages` as lists of
    {role, content} dicts) and the already-normalized training-split schema
    (`utterances` as a list of strings). Mirrors clustering._normalise_messages
    so eval and training see identical text.
    """
    convo = example.get("conversations") or example.get("messages") or []
    if convo:
        return [str(t.get("content", "")).strip() for t in convo if str(t.get("content", "")).strip()]
    return [str(u).strip() for u in example.get("utterances", []) if str(u).strip()]


def sample_test_examples(
    cluster_assignments: dict[str, int],
    n_per_cluster: int = 50,
    seed: int = 42,
    split: str = "test",
) -> list[dict]:
    """Seeded, cluster-stratified sample of valid evaluation conversations.

    Returns a flat shuffled list. Each example dict gains a `cluster_id` field
    and a stable `example_id` (its index in the raw HF split) so per-example
    results from different systems and sessions can be joined later.
    """
    from datasets import load_dataset

    dataset = load_dataset("Estwld/empathetic_dialogues_llm", split=split)
    by_cluster: dict[int, list[dict]] = {cid: [] for cid in range(4)}
    for raw_idx, example in enumerate(dataset):
        emotion = str(example.get("emotion", "")).lower().strip()
        cid = cluster_assignments.get(emotion)
        if cid is None:
            continue
        record = dict(example)
        # The Estwld/empathetic_dialogues_llm split stores turns under
        # `conversations` ([{role, content}]); the rest of the stack (prompts.py,
        # is_valid_eval_example, gold_response) expects `utterances` ([str]), the
        # same normalized form clustering.py wrote into the training splits.
        # Normalize here so eval examples match the training distribution.
        record["utterances"] = _utterances_from_example(example)
        if not is_valid_eval_example(record):
            continue
        record["example_id"] = raw_idx
        record["cluster_id"] = cid
        by_cluster[cid].append(record)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for cid, pool in by_cluster.items():
        rng.shuffle(pool)
        if len(pool) < n_per_cluster:
            print(f"WARN: cluster {cid} has only {len(pool)} valid test examples (< {n_per_cluster}).")
        sampled.extend(pool[:n_per_cluster])
    rng.shuffle(sampled)
    print(
        f"Sampled {len(sampled)} test examples "
        f"({ {cid: sum(1 for ex in sampled if ex['cluster_id'] == cid) for cid in range(4)} } per cluster, seed={seed})"
    )
    return sampled


# ---------------------------------------------------------------------------
# Likelihood scoring (per example)
# ---------------------------------------------------------------------------


def encode_prompt_response(tokenizer, prompt: str, response: str, device, max_length: int = 512):
    """Encode [prompt | response | eos] with prompt tokens masked out of the loss."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if not response_ids:
        return None
    if len(response_ids) >= max_length:
        input_ids = response_ids[:max_length]
        labels = list(input_ids)
    else:
        prompt_budget = max_length - len(response_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
    return {
        "input_ids": torch.tensor([input_ids], device=device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
        "labels": torch.tensor([labels], device=device),
    }


def example_nll(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device=None,
    max_length: int = 512,
) -> dict | None:
    """Summed negative log-likelihood of `response` given `prompt`, for ONE example.

    Returns {"sum_nll": float, "n_tokens": int} or None when the response is empty.
    Per-example sums let the caller compute either corpus PPL
    (exp(total_nll / total_tokens)) or per-example mean PPL, plus bootstrap CIs.
    """
    device = device or str(next(model.parameters()).device)
    encoded = encode_prompt_response(tokenizer, prompt, response, device, max_length)
    if encoded is None:
        return None
    labels = encoded.pop("labels")
    n_tokens = int((labels != -100).sum().item())
    if n_tokens == 0:
        return None
    with torch.no_grad():
        outputs = model(**encoded, labels=labels)
    return {"sum_nll": float(outputs.loss.item()) * n_tokens, "n_tokens": n_tokens}


def per_example_nll(
    model,
    tokenizer,
    examples: list[dict],
    prompt_builder: Callable[[dict, object], str],
    setup_fn: Callable[[dict], None] | None = None,
    context_manager=None,
    max_length: int = 512,
    desc: str | None = None,
) -> list[dict]:
    """Score every example, returning one NLL record per example.

    - `prompt_builder(example, tokenizer)` must be the SAME builder the system
      uses for generation (prompt/likelihood consistency).
    - `setup_fn(example)` configures per-example model state (e.g. set the
      gold-cluster adapter, or apply gate-predicted blend weights) BEFORE the
      forward pass.
    - `context_manager(model)`, if given, wraps the forward (e.g.
      `adapters_disabled` for no-adapter baselines).
    """
    from tqdm.auto import tqdm

    model.eval()
    device = str(next(model.parameters()).device)
    records: list[dict] = []
    for example in tqdm(examples, desc=desc or "nll"):
        if setup_fn is not None:
            setup_fn(example)
        prompt = prompt_builder(example, tokenizer)
        response = gold_response(example)
        if context_manager is not None:
            with context_manager(model):
                rec = example_nll(model, tokenizer, prompt, response, device, max_length)
        else:
            rec = example_nll(model, tokenizer, prompt, response, device, max_length)
        if rec is None:
            rec = {"sum_nll": float("nan"), "n_tokens": 0}
        rec["example_id"] = example.get("example_id")
        rec["cluster_id"] = example.get("cluster_id")
        rec["emotion"] = example.get("emotion")
        records.append(rec)
    return records


def corpus_ppl(records: Iterable[dict]) -> float:
    """exp(total NLL / total tokens) — comparable to v1's number, now derivable from records."""
    total_nll = sum(r["sum_nll"] for r in records if r["n_tokens"] > 0)
    total_tokens = sum(r["n_tokens"] for r in records if r["n_tokens"] > 0)
    return math.exp(total_nll / total_tokens) if total_tokens else float("nan")


def mean_example_nll(records: Iterable[dict]) -> float:
    """Mean per-example NLL/token — the statistic used for paired tests."""
    vals = [r["sum_nll"] / r["n_tokens"] for r in records if r["n_tokens"] > 0]
    return float(np.mean(vals)) if vals else float("nan")


def median_ppl(records: Iterable[dict]) -> float:
    """Median of per-example PPL — robust to short-response/typo outliers.

    Corpus PPL (exp of token-weighted mean NLL) is fragile: a 3-token gold reply
    where the model assigns ~1e-12 to one token contributes a per-token NLL near
    25, and with no long tail to dilute it the example's PPL hits ~1e11. A few
    such examples distort the corpus number badly (this is exactly what made the
    base row of the specialization matrix read ~1200 instead of a sane floor).
    The median ignores those tails and reports the typical example, so it is the
    primary statistic for the matrix; corpus PPL is kept alongside for continuity.
    """
    vals = [math.exp(r["sum_nll"] / r["n_tokens"]) for r in records if r["n_tokens"] > 0]
    return float(np.median(vals)) if vals else float("nan")


def paired_bootstrap_nll(
    records_a: list[dict],
    records_b: list[dict],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap on per-example NLL/token: is system A better than B?

    Records must be aligned on the same examples (same order; checked via
    example_id when available). Returns the mean difference (A - B; negative
    means A assigns higher likelihood), a 95% CI, and a two-sided p-value.
    """
    ids_a = [r.get("example_id") for r in records_a]
    ids_b = [r.get("example_id") for r in records_b]
    if any(i is not None for i in ids_a) and ids_a != ids_b:
        raise ValueError("paired_bootstrap_nll requires records aligned on the same examples.")

    diffs = np.array(
        [
            a["sum_nll"] / a["n_tokens"] - b["sum_nll"] / b["n_tokens"]
            for a, b in zip(records_a, records_b)
            if a["n_tokens"] > 0 and b["n_tokens"] > 0
        ]
    )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    observed = float(diffs.mean())
    # two-sided p: fraction of bootstrap means on the far side of zero
    p = 2.0 * min(float((boot_means >= 0).mean()), float((boot_means <= 0).mean()))
    return {
        "mean_diff_nll_per_token": observed,
        "ci95_low": float(np.percentile(boot_means, 2.5)),
        "ci95_high": float(np.percentile(boot_means, 97.5)),
        "p_value": min(1.0, p),
        "n_examples": int(len(diffs)),
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    model,
    tokenizer,
    prompt: str,
    seed: int,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    greedy: bool = False,
) -> str:
    """Seeded generation so every reported output is reproducible."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    torch.manual_seed(seed)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------


def distinct_n(texts: Iterable[str], n: int) -> float:
    """Corpus-level lexical diversity: unique n-grams / total n-grams."""
    ngrams = []
    for text in texts:
        tokens = str(text).split()
        ngrams.extend(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


_EMOTION_CLASSIFIER = None


def broad_emotion_predictions(texts: list[str]) -> list[str]:
    """7-way broad emotion label for each generation (anger/disgust/fear/joy/sadness/surprise/neutral).

    NOTE on the known confound: a system whose prompt names the gold emotion can
    satisfy this classifier by parroting the emotion word. Compare only systems
    with the same information access (the `informed` flag in final_eval).
    """
    global _EMOTION_CLASSIFIER
    if _EMOTION_CLASSIFIER is None:
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        _EMOTION_CLASSIFIER = pipeline(
            "text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=device
        )
    return [_EMOTION_CLASSIFIER(str(t)[:512])[0]["label"].lower() for t in texts]


def emotion_match(predictions: list[str], gold_emotions: list[str]) -> list[int | None]:
    """Per-example 0/1 emotion match (None when the gold label has no broad mapping)."""
    from evaluate import EMOTION_TO_BROAD  # the ED-32 -> 7-way mapping lives in legacy evaluate.py

    out: list[int | None] = []
    for pred, gold in zip(predictions, gold_emotions):
        target = EMOTION_TO_BROAD.get(str(gold).lower().strip())
        out.append(None if target is None else int(pred == target))
    return out


def bertscore_f1(texts: list[str], references: list[str]) -> list[float] | None:
    """Reference-based semantic similarity to the gold response (optional dependency).

    Unlike the emotion classifier, this cannot be gamed by naming the emotion:
    it rewards saying something close to what an actual empathetic listener said.
    Returns None when `bert-score` is not installed.
    """
    try:
        from bert_score import score as _bert_score
    except ImportError:
        print("bert-score not installed; skipping BERTScore (pip install bert-score).")
        return None
    _, _, f1 = _bert_score(texts, references, lang="en", rescale_with_baseline=True)
    return [float(v) for v in f1]


# ---------------------------------------------------------------------------
# Routing statistics
# ---------------------------------------------------------------------------


def alpha_entropy(alpha) -> float:
    vec = np.clip(np.asarray(alpha, dtype=float).reshape(-1, np.asarray(alpha).shape[-1]).mean(axis=0), 1e-9, 1.0)
    vec = vec / vec.sum()
    return float(-np.sum(vec * np.log(vec)))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    print(f"Wrote {path}")
    return path


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(payload, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"Wrote {path}")
    return path


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
