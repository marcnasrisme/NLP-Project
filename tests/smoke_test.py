"""Local smoke test for the GPU-free parts of the DESA experiment stack.

Run from anywhere:  python3 tests/smoke_test.py
Covers prompts.py, eval_core.py statistics, specialization.py summaries, and
analysis.py tables — everything that does not need a GPU or model downloads.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

# --- prompts.py ---------------------------------------------------------
from prompts import (
    build_emotion_informed_prompt, build_generic_empathy_prompt, build_vanilla_prompt,
    gold_response, history_messages, is_valid_eval_example,
)

class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = " ".join(f"<{m['role']}>{m['content']}" for m in messages)
        return f"<s>{body}" + ("<gen>" if add_generation_prompt else "")

ex = {"utterances": ["I lost my job", "I'm so sorry to hear that", "yeah it's rough", "That sounds really hard"],
      "emotion": "devastated"}
assert is_valid_eval_example(ex)
assert not is_valid_eval_example({"utterances": ["a", "b", "c"]})  # ends on user turn
assert gold_response(ex) == "That sounds really hard"
msgs = history_messages(ex["utterances"])
assert len(msgs) == 3 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
tok = FakeTok()
v = build_vanilla_prompt(ex, tok)
g = build_generic_empathy_prompt(ex, tok)
i = build_emotion_informed_prompt(ex, tok)
assert "I lost my job" in v and "<gen>" in v
assert "empathetic" in g and "devastated" not in g, "generic prompt must NOT leak the label"
assert "devastated" in i, "informed prompt must name the label"
# chat structure preserved: instruction merged into first user turn, 3 turns total
assert g.count("<user>") == 2 and g.count("<assistant>") == 1
print("prompts.py OK")

# --- eval_core bootstrap -------------------------------------------------
from eval_core import corpus_ppl, mean_example_nll, paired_bootstrap_nll, distinct_n
from eval_core import _utterances_from_example

# Regression: the ED-llm test split stores turns under `conversations`
# ([{role, content}]), not `utterances`. sample_test_examples must normalize
# them or it silently samples 0 examples (the bug seen in notebook 07).
convo_ex = {"emotion": "guilty", "conversations": [
    {"role": "user", "content": "I felt bad about it."},
    {"role": "assistant", "content": "That sounds tough."},
]}
norm = _utterances_from_example(convo_ex)
assert norm == ["I felt bad about it.", "That sounds tough."], norm
assert is_valid_eval_example({**convo_ex, "utterances": norm}), "even-length convo must be valid"
# already-normalized utterances must pass through unchanged
assert _utterances_from_example({"utterances": ["a", "b"]}) == ["a", "b"]

rng = np.random.default_rng(0)
recs_a = [{"sum_nll": float(rng.normal(20, 2)), "n_tokens": 10, "example_id": k} for k in range(200)]
recs_b = [{"sum_nll": recs_a[k]["sum_nll"] + 3.0, "n_tokens": 10, "example_id": k} for k in range(200)]
res = paired_bootstrap_nll(recs_a, recs_b, n_boot=2000, seed=1)
assert res["mean_diff_nll_per_token"] < 0 and res["ci95_high"] < 0 and res["p_value"] < 0.01
assert corpus_ppl(recs_a) > 1.0
assert abs(mean_example_nll(recs_a) - np.mean([r["sum_nll"]/10 for r in recs_a])) < 1e-9
assert distinct_n(["a b c", "a b d"], 2) == 3/4
# misaligned pairing must raise
try:
    paired_bootstrap_nll(recs_a, list(reversed(recs_b)), n_boot=10)
    raise AssertionError("should have raised on misaligned ids")
except ValueError:
    pass
print("eval_core.py OK")

# --- specialization summary ----------------------------------------------
from specialization import matrix_dataframe, summarize_matrix

matrix = {
    "base": {0: 60.0, 1: 60.0, 2: 60.0, 3: 60.0},
    "expert_0": {0: 30.0, 1: 42.0, 2: 44.0, 3: 43.0},
    "expert_1": {0: 41.0, 1: 31.0, 2: 45.0, 3: 42.0},
    "expert_2": {0: 42.0, 1: 43.0, 2: 29.0, 3: 41.0},
    "expert_3": {0: 43.0, 1: 41.0, 2: 42.0, 3: 30.0},
    "uniform_blend": {0: 36.0, 1: 36.0, 2: 36.0, 3: 36.0},
    "pooled": {0: 33.0, 1: 33.0, 2: 33.0, 3: 33.0},
}
s = summarize_matrix(matrix)
assert all(v < 0.8 for v in s["diagonal_advantage_ratio"].values()), s
assert "pooled_vs_diagonal_ratio" in s
df = matrix_dataframe(matrix)
assert df.shape == (7, 4)
print("specialization.py OK")

# --- analysis.py ----------------------------------------------------------
from analysis import (
    DEFAULT_PAIRS, load_results, paired_comparisons, per_cluster_ppl, routing_table, summary_table,
)

rows = []
for system, informed, shift in [("base_chat", False, 1.0), ("uniform_blend", False, 0.6),
                                ("turn_gate", False, 0.55), ("oracle_expert", True, 0.5)]:
    for k in range(120):
        cid = k % 4
        alpha = None
        if system in ("turn_gate", "oracle_expert"):
            a = np.full(4, 0.1); a[cid] = 0.7
            alpha = a.tolist()
        rows.append({
            "system": system, "informed": informed, "example_id": k, "cluster_id": cid,
            "emotion": "sad", "generation": f"gen {system} {k} words vary {k*7%13}",
            "alpha": alpha, "sum_nll": float(rng.normal(30*shift, 2)), "n_tokens": 12,
            "gold_response": "g", "emotion_pred": "sadness", "emotion_match": int(k % 3 == 0),
        })
p = str(Path(tempfile.mkdtemp()) / "smoke_rows.jsonl")
with open(p, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
frame = load_results(p)
summ = summary_table(frame, n_boot=300)
assert {"ppl", "emotion_acc", "distinct_2"}.issubset(summ.columns)
pairs = paired_comparisons(frame, [("turn_gate", "uniform_blend"), ("turn_gate", "base_chat")], n_boot=500)
assert pairs.loc[1, "winner"] == "turn_gate" and pairs.loc[1, "significant"]
pc = per_cluster_ppl(frame)
assert pc.shape == (4, 4)
rt = routing_table(frame)
assert "turn_gate" in rt.index and rt.loc["turn_gate", "routing_accuracy"] == 1.0
print("analysis.py OK")

print("\nALL SMOKE TESTS PASSED")
