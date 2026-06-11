"""Corrected evaluation orchestration for DESA.

LEGACY (v2, never run): superseded by `final_eval.py` + `analysis.py`
(notebook 10), which additionally add information-parity prompts, per-example
records, seeded generation, and bootstrap statistics.

Additive companion to `evaluate.py`. The original `04_evaluation.ipynb` measures
`static_prompt`'s perplexity against an instruction-wrapped prompt (which crams
the whole conversation into one user turn). Mistral-Instruct does not predict
well from that format, so PPL inflates to ~22 866 — not comparable to the other
systems' PPL, which use vanilla chat templates.

This module fixes that by computing PPL under a uniform prompt format for every
system, and adds two reference systems:

- `uniform_blend`: turn-level routing with a hard-coded alpha = (1/4, 1/4, 1/4, 1/4).
  Any trained gate that doesn't beat this baseline learned nothing useful.
- `oracle_blend`: turn-level routing with alpha = e_y (one-hot on the gold cluster).
  This is the achievable ceiling of single-cluster routing under blending.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evaluate import (
    distinct_n,
    emotion_accuracy,
    compute_gold_perplexity,
    gating_alignment,
    gating_entropy,
    unpack_generated,
)
from inference import (
    ADAPTER_NAMES,
    adapters_disabled,
    apply_weighted_adapters,
    build_messages_from_utterances,
    build_prompt,
    build_static_emotion_prompt,
    generate_argmax_adapter,
    generate_static_prompt,
    generate_token_level,
    generate_turn_level,
    _generate,
)
from inference_v2 import compute_turn_alpha_v2
from paths import OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Generation for new baselines
# ---------------------------------------------------------------------------


def generate_uniform_blend(
    model,
    tokenizer,
    conversation_history: list[dict],
    max_new_tokens: int = 100,
) -> tuple[str, np.ndarray]:
    """Apply alpha = (1/4, 1/4, 1/4, 1/4) and generate."""
    alpha = np.full(4, 0.25, dtype=float)
    apply_weighted_adapters(model, alpha)
    prompt = build_prompt(conversation_history, tokenizer=tokenizer, add_generation_prompt=True)
    text = _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    return text, alpha


def generate_oracle_blend(
    model,
    tokenizer,
    conversation_history: list[dict],
    emotion: str,
    cluster_assignments: dict[str, int],
    max_new_tokens: int = 100,
) -> tuple[str, np.ndarray]:
    """Apply alpha = one-hot(gold_cluster) and generate."""
    cid = cluster_assignments.get(emotion.lower().strip(), 0)
    alpha = np.zeros(4, dtype=float)
    alpha[cid] = 1.0
    apply_weighted_adapters(model, alpha)
    prompt = build_prompt(conversation_history, tokenizer=tokenizer, add_generation_prompt=True)
    text = _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    return text, alpha


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------


def run_all_systems_v2(
    examples: list[dict],
    multi_adapter_model,
    xlora_model,
    tokenizer,
    turn_gate_v2,
    cluster_assignments: dict[str, int],
    max_new_tokens: int = 100,
) -> dict:
    """Generate for the four original systems plus uniform_blend and oracle_blend.

    The original four (`static_prompt`, `argmax_adapter`, `turn_level`,
    `token_level`) are produced with the same helpers as `inference.run_all_systems`,
    but the turn-level system uses the v2 last-token alpha and the v2 X-LoRA model.
    """
    from inference import compute_turn_alpha  # noqa: F401  (kept available for callers)

    results = {
        "static_prompt": [],
        "argmax_adapter": [],
        "turn_level": [],
        "token_level": [],
        "uniform_blend": [],
        "oracle_blend": [],
    }
    for idx, example in enumerate(examples):
        history = build_messages_from_utterances(example.get("utterances", []), include_last=False)
        emotion = example.get("emotion", "")
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(examples)}")

        results["static_prompt"].append(
            generate_static_prompt(multi_adapter_model, tokenizer, history, emotion, max_new_tokens)
        )
        results["argmax_adapter"].append(
            generate_argmax_adapter(
                multi_adapter_model, tokenizer, history, emotion, cluster_assignments, max_new_tokens
            )
        )

        # turn_level uses the v2 alpha computation (last-token pooling)
        alpha_v2 = compute_turn_alpha_v2(multi_adapter_model, tokenizer, turn_gate_v2, history)
        apply_weighted_adapters(multi_adapter_model, alpha_v2)
        prompt = build_prompt(history, tokenizer=tokenizer, add_generation_prompt=True)
        turn_text = _generate(multi_adapter_model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        results["turn_level"].append((turn_text, alpha_v2))

        # uniform and oracle baselines reuse the same multi_adapter_model
        results["uniform_blend"].append(
            generate_uniform_blend(multi_adapter_model, tokenizer, history, max_new_tokens)
        )
        results["oracle_blend"].append(
            generate_oracle_blend(
                multi_adapter_model, tokenizer, history, emotion, cluster_assignments, max_new_tokens
            )
        )

        # X-LoRA token-level (uses its own model)
        token_text, token_alpha = generate_token_level(xlora_model, tokenizer, history, max_new_tokens)
        results["token_level"].append((token_text, token_alpha))

    return results


# ---------------------------------------------------------------------------
# Evaluation with uniform PPL prompt format
# ---------------------------------------------------------------------------


def _setup_argmax_factory(model, cluster_assignments):
    def setup(example):
        cid = cluster_assignments.get(example["emotion"].lower().strip(), 0)
        model.set_adapter(f"cluster_{cid}")
    return setup


def _setup_turn_factory(model, tokenizer, gate):
    def setup(example):
        history = build_messages_from_utterances(example.get("utterances", []), include_last=False)
        alpha = compute_turn_alpha_v2(model, tokenizer, gate, history)
        apply_weighted_adapters(model, alpha)
    return setup


def _setup_uniform_factory(model):
    def setup(example):
        del example
        apply_weighted_adapters(model, np.full(4, 0.25, dtype=float))
    return setup


def _setup_oracle_factory(model, cluster_assignments):
    def setup(example):
        cid = cluster_assignments.get(example["emotion"].lower().strip(), 0)
        alpha = np.zeros(4, dtype=float)
        alpha[cid] = 1.0
        apply_weighted_adapters(model, alpha)
    return setup


def _evaluate_one(
    name: str,
    generated_outputs,
    gold_emotions: list[str],
    examples: list[dict],
    *,
    model=None,
    tokenizer=None,
    cluster_assignments=None,
    pre_forward=None,
    context_manager=None,
) -> dict:
    """Run all metrics for one system. PPL always uses the vanilla prompt format."""
    print(f"\n[v2] Evaluating: {name}")
    generated_texts, alpha_list = unpack_generated(generated_outputs)
    out = {
        "system": name,
        "distinct_1": distinct_n(generated_texts, 1),
        "distinct_2": distinct_n(generated_texts, 2),
        "emotion_accuracy": emotion_accuracy(generated_texts, gold_emotions),
    }
    if model is not None and tokenizer is not None:
        out["perplexity"] = compute_gold_perplexity(
            model,
            tokenizer,
            examples,
            pre_forward=pre_forward,
            context_manager=context_manager,
            prompt_builder=None,  # vanilla chat template across all systems
        )
    if alpha_list is not None:
        out.update(gating_entropy(alpha_list))
        if cluster_assignments:
            out.update(gating_alignment(alpha_list, gold_emotions, cluster_assignments))
    print(json.dumps(out, indent=2))
    return out


def evaluate_all_v2(
    examples: list[dict],
    all_outputs: dict,
    multi_adapter_model,
    xlora_model,
    tokenizer,
    turn_gate_v2,
    cluster_assignments: dict[str, int],
) -> list[dict]:
    """Evaluate all six systems. Returns the list of result dicts."""
    gold_emotions = [ex["emotion"] for ex in examples]
    results: list[dict] = []

    # static_prompt: adapters disabled, vanilla PPL prompt (the fix).
    results.append(
        _evaluate_one(
            "static_prompt",
            all_outputs["static_prompt"],
            gold_emotions,
            examples,
            model=multi_adapter_model,
            tokenizer=tokenizer,
            context_manager=adapters_disabled,
        )
    )

    # argmax_adapter: per-example sets the gold-cluster adapter, vanilla PPL prompt.
    results.append(
        _evaluate_one(
            "argmax_adapter",
            all_outputs["argmax_adapter"],
            gold_emotions,
            examples,
            model=multi_adapter_model,
            tokenizer=tokenizer,
            pre_forward=_setup_argmax_factory(multi_adapter_model, cluster_assignments),
        )
    )

    # turn_level (v2): per-example computes alpha and applies blended adapter.
    results.append(
        _evaluate_one(
            "turn_level_v2",
            all_outputs["turn_level"],
            gold_emotions,
            examples,
            model=multi_adapter_model,
            tokenizer=tokenizer,
            cluster_assignments=cluster_assignments,
            pre_forward=_setup_turn_factory(multi_adapter_model, tokenizer, turn_gate_v2),
        )
    )

    # uniform_blend baseline.
    results.append(
        _evaluate_one(
            "uniform_blend",
            all_outputs["uniform_blend"],
            gold_emotions,
            examples,
            model=multi_adapter_model,
            tokenizer=tokenizer,
            cluster_assignments=cluster_assignments,
            pre_forward=_setup_uniform_factory(multi_adapter_model),
        )
    )

    # oracle_blend ceiling.
    results.append(
        _evaluate_one(
            "oracle_blend",
            all_outputs["oracle_blend"],
            gold_emotions,
            examples,
            model=multi_adapter_model,
            tokenizer=tokenizer,
            cluster_assignments=cluster_assignments,
            pre_forward=_setup_oracle_factory(multi_adapter_model, cluster_assignments),
        )
    )

    # token_level X-LoRA: PPL is computed on the X-LoRA model itself.
    results.append(
        _evaluate_one(
            "token_level_xlora_v2",
            all_outputs["token_level"],
            gold_emotions,
            examples,
            model=xlora_model,
            tokenizer=tokenizer,
            cluster_assignments=cluster_assignments,
        )
    )

    return results


def save_results_v2(
    all_results: list[dict],
    output_dir: str | Path = OUTPUTS_DIR,
    json_name: str = "eval_results_v2.json",
    csv_name: str = "eval_results_v2.csv",
) -> tuple[Path, Path | None]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / json_name
    csv_path = output_dir / csv_name
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    csv_written: Path | None = None
    try:
        import pandas as pd

        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        csv_written = csv_path
    except Exception as exc:
        print(f"Could not write CSV: {exc}")
    print(f"Saved v2 results to {json_path}")
    return json_path, csv_written


# Re-export for notebook convenience.
__all__ = [
    "generate_uniform_blend",
    "generate_oracle_blend",
    "run_all_systems_v2",
    "evaluate_all_v2",
    "save_results_v2",
]
