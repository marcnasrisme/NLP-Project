"""Final evaluation — every system, fair conditions, per-example records.

What was wrong with v1's comparison, and how each is fixed here
---------------------------------------------------------------
1. `static_prompt` generated with an instruction prompt but had PPL computed
   on a different prompt with an undefined adapter state (PPL 22,866).
   -> Every system declares ONE prompt builder, used for BOTH generation and
   likelihood scoring, and a setup function that fixes the adapter state
   before every forward.
2. The static baseline was told the gold emotion; adapter systems were not, so
   "emotion accuracy" compared systems with different information.
   -> Every system carries an `informed` flag. Uninformed systems see only the
   conversation. Informed systems (gold label leaked) are reported separately;
   the honest prompt baseline is `generic_empathy` (instruction, no label).
3. Single corpus PPL, unseeded sampling, first-N test slice.
   -> Per-example NLL records, seeded generation, stratified test sample; all
   rows persisted to JSONL for bootstrap CIs in `analysis.py`.

The systems
-----------
uninformed (see only the conversation):
  base_chat        frozen Mistral, plain chat prompt           [floor]
  generic_empathy  frozen Mistral + empathy instruction        [prompt baseline]
  pooled_adapter   one LoRA trained on all clusters            [no-routing adapter]
  uniform_blend    fixed 1/4 blend of the four experts         [routing-free blend]
  random_expert    seeded random single expert per example     [routing floor]
  turn_gate        learned gate -> blended experts             [the DESA system]
informed (gold emotion label leaks into the system):
  emotion_prompt   frozen Mistral + instruction naming emotion [prompt ceiling]
  oracle_expert    gold-cluster expert via the label           [routing ceiling]

The decisive comparisons are within the uninformed block:
  turn_gate vs uniform_blend  — does LEARNED routing beat a constant blend?
  turn_gate vs pooled_adapter — does routing beat one generalist adapter?
  oracle_expert vs uniform_blend (cross-block) — is there ANY routing headroom
  even with a perfect router? (Should agree with the specialization matrix.)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from eval_core import (
    broad_emotion_predictions,
    distinct_n,
    emotion_match,
    example_nll,
    generate,
    save_jsonl,
)
from paths import OUTPUTS_DIR
from prompts import (
    build_emotion_informed_prompt,
    build_generic_empathy_prompt,
    build_vanilla_prompt,
    gold_response,
)

EXPERIMENT_DIR = OUTPUTS_DIR / "experiments" / "final_eval"
POOLED_ADAPTER_NAME = "pooled"


# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------


def build_systems(
    model,
    tokenizer,
    cluster_assignments: dict[str, int],
    turn_gate=None,
    include_pooled: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Construct the system specs against a shared multi-adapter PeftModel.

    Each spec: {name, informed, prompt_builder, setup(example) -> alpha|None,
    context_manager|None}. `setup` runs before BOTH generation and NLL scoring
    of each example and returns the routing weights it applied (for routing
    diagnostics), or None for non-routed systems.
    """
    from inference import adapters_disabled, apply_weighted_adapters

    rng = random.Random(seed)

    def setup_noop(example):
        return None

    def setup_uniform(example):
        alpha = np.full(4, 0.25)
        apply_weighted_adapters(model, alpha)
        return alpha

    def setup_random(example):
        cid = rng.randrange(4)
        model.set_adapter(f"cluster_{cid}")
        alpha = np.zeros(4)
        alpha[cid] = 1.0
        return alpha

    def setup_oracle(example):
        cid = cluster_assignments.get(str(example.get("emotion", "")).lower().strip(), 0)
        model.set_adapter(f"cluster_{cid}")
        alpha = np.zeros(4)
        alpha[cid] = 1.0
        return alpha

    def setup_pooled(example):
        model.set_adapter(POOLED_ADAPTER_NAME)
        return None

    systems = [
        dict(name="base_chat", informed=False, prompt_builder=build_vanilla_prompt,
             setup=setup_noop, context_manager=adapters_disabled),
        dict(name="generic_empathy", informed=False, prompt_builder=build_generic_empathy_prompt,
             setup=setup_noop, context_manager=adapters_disabled),
        dict(name="uniform_blend", informed=False, prompt_builder=build_vanilla_prompt,
             setup=setup_uniform, context_manager=None),
        dict(name="random_expert", informed=False, prompt_builder=build_vanilla_prompt,
             setup=setup_random, context_manager=None),
        dict(name="emotion_prompt", informed=True, prompt_builder=build_emotion_informed_prompt,
             setup=setup_noop, context_manager=adapters_disabled),
        dict(name="oracle_expert", informed=True, prompt_builder=build_vanilla_prompt,
             setup=setup_oracle, context_manager=None),
    ]

    if include_pooled:
        systems.insert(2, dict(name="pooled_adapter", informed=False,
                               prompt_builder=build_vanilla_prompt,
                               setup=setup_pooled, context_manager=None))

    if turn_gate is not None:
        from inference_v2 import compute_turn_alpha_v2
        from prompts import history_messages

        def setup_turn_gate(example):
            history = history_messages(example.get("utterances", []))
            alpha = compute_turn_alpha_v2(model, tokenizer, turn_gate, history)
            apply_weighted_adapters(model, alpha)
            return alpha

        systems.append(dict(name="turn_gate", informed=False, prompt_builder=build_vanilla_prompt,
                            setup=setup_turn_gate, context_manager=None))
    return systems


# ---------------------------------------------------------------------------
# The evaluation loop
# ---------------------------------------------------------------------------


def run_final_eval(
    systems: list[dict],
    model,
    tokenizer,
    examples: list[dict],
    out_path: Path = EXPERIMENT_DIR / "per_example.jsonl",
    max_new_tokens: int = 100,
    max_length: int = 512,
    seed: int = 42,
    greedy: bool = False,
) -> list[dict]:
    """For every (system, example): set state -> generate (seeded) -> score NLL.

    One JSONL row per (system, example) with everything needed downstream:
    generation text, routing alpha, NLL sum and token count, gold metadata.
    Emotion-classifier predictions are appended afterwards (batched).
    """
    from tqdm.auto import tqdm

    model.eval()
    device = str(next(model.parameters()).device)
    rows: list[dict] = []

    for spec in systems:
        name = spec["name"]
        print(f"\n=== system: {name} ({'informed' if spec['informed'] else 'uninformed'}) ===")
        for idx, example in enumerate(tqdm(examples, desc=name)):
            alpha = spec["setup"](example)
            prompt = spec["prompt_builder"](example, tokenizer)
            gen_seed = seed * 100_000 + idx  # same per example across systems

            ctx = spec["context_manager"]
            if ctx is not None:
                with ctx(model):
                    text = generate(model, tokenizer, prompt, seed=gen_seed,
                                    max_new_tokens=max_new_tokens, greedy=greedy)
                    nll = example_nll(model, tokenizer, prompt, gold_response(example),
                                      device, max_length)
            else:
                text = generate(model, tokenizer, prompt, seed=gen_seed,
                                max_new_tokens=max_new_tokens, greedy=greedy)
                nll = example_nll(model, tokenizer, prompt, gold_response(example),
                                  device, max_length)

            rows.append(
                {
                    "system": name,
                    "informed": spec["informed"],
                    "example_id": example.get("example_id"),
                    "cluster_id": example.get("cluster_id"),
                    "emotion": example.get("emotion"),
                    "generation": text,
                    "alpha": None if alpha is None else [float(a) for a in alpha],
                    "sum_nll": None if nll is None else nll["sum_nll"],
                    "n_tokens": None if nll is None else nll["n_tokens"],
                    "gold_response": gold_response(example),
                }
            )

    rows = append_emotion_predictions(rows)
    save_jsonl(rows, out_path)
    return rows


def append_emotion_predictions(rows: list[dict]) -> list[dict]:
    """Batch the 7-way emotion classifier over all generations and store
    per-row predictions + 0/1 match against the broad gold label."""
    texts = [row["generation"] for row in rows]
    preds = broad_emotion_predictions(texts)
    matches = emotion_match(preds, [row["emotion"] for row in rows])
    for row, pred, match in zip(rows, preds, matches):
        row["emotion_pred"] = pred
        row["emotion_match"] = match
    return rows


def quick_table(rows: list[dict]):
    """Small in-notebook sanity table; the real aggregation is analysis.py."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame["nll_per_token"] = frame["sum_nll"] / frame["n_tokens"]
    out = frame.groupby(["informed", "system"]).agg(
        corpus_ppl=("nll_per_token", lambda s: float(np.exp(
            frame.loc[s.index, "sum_nll"].sum() / frame.loc[s.index, "n_tokens"].sum()))),
        emotion_acc=("emotion_match", "mean"),
        n=("system", "size"),
    )
    for (informed, system), group in frame.groupby(["informed", "system"]):
        out.loc[(informed, system), "distinct_1"] = distinct_n(group["generation"], 1)
        out.loc[(informed, system), "distinct_2"] = distinct_n(group["generation"], 2)
    return out.sort_values("corpus_ppl")
