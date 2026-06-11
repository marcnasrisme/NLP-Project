# DESA Research Plan (v3) — When Does Learned Routing Over LoRA Experts Actually Help?

**Quantitative Methods for NLP, MIT Spring 2026**
Status: redesign after the v1/v2 evaluation. This document is the source of truth for what
we are claiming, how each claim is tested, and what we will report.

---

## 1. Why the redesign

The v1 pipeline (notebooks 01–04) trained four emotion-quadrant LoRA experts on
EmpatheticDialogues and two learned routers, then compared four systems. The results
(`outputs/eval_results.json`, dissected in `PROJECT_EXPLAINER.md`) contained three fatal
problems:

1. **Both routers collapsed.** The turn gate emitted one constant vector for every input
   (`std_entropy ≈ 4e-16`); the X-LoRA classifier emitted exactly uniform weights
   (entropy = log 4 ± 5e-6).
2. **The evaluation was unfair and partly broken.** The static baseline's PPL (22,866) was
   computed on a different prompt than it generated with, under an undefined adapter state;
   its emotion-accuracy "win" came from being told the gold label; no metric had any
   uncertainty attached.
3. **A buried contradiction.** The *collapsed* blend (PPL 30.7) beat the *oracle-routed*
   expert (PPL 39.1) — impossible if the experts were meaningfully specialized — and X-LoRA
   at uniform weights (PPL 7,272) differed by 200× from the mathematically equivalent
   uniform blend, proving an integration bug.

The redesign stops treating "routing will help" as a premise and makes it the research
question. The negative results become the object of study.

## 2. The research question and hypotheses

> **RQ: Under what conditions does learned routing over emotion-specialized LoRA experts
> improve empathetic dialogue generation — and why did it fail here?**

- **H1 (specialization).** The four quadrant experts are largely interchangeable on
  EmpatheticDialogues: the diagonal of the expert×cluster PPL matrix is not meaningfully
  better than the off-diagonal, and one pooled-data adapter at equal budget matches them.
- **H2 (signal).** The emotion quadrant is decodable from frozen base-model hidden states
  (linear probe ≫ chance), i.e., the v1 gate collapse was an optimization/pooling failure,
  not signal absence.
- **H3 (granularity).** Routing supervision must match routing granularity: pooled CE over
  ~2,560 per-(token,layer) decisions provably dilutes per-position gradients by 1/(T·L) and
  empirically collapses routing to uniform, while per-token CE and NTP-through-the-mixture
  do not.
- **H4 (system).** Given H1, the learned gate will not significantly beat a uniform blend
  or a pooled adapter on gold-response NLL — and with the fair evaluation we can state this
  with confidence intervals instead of anecdotes.

H1+H3 together explain every v1 pathology mechanistically. If H1 is *rejected* (experts do
specialize), the same experiments instead quantify how much routing recovers of the oracle
gap — either outcome is a complete, defensible story.

## 3. Experiments → notebooks → modules

| # | Experiment | Notebook | Module | Decides |
|---|---|---|---|---|
| 0 | Pooled control adapter (all data, one expert's budget) | 06 | `train_adapter.train_pooled` | enables E1/E4 |
| 1 | **Specialization matrix**: {base, expert_0..3, uniform, pooled} × 4 test clusters, gold-response PPL | 07 | `specialization.py` | H1 |
| 2 | **Routing-signal probe**: logistic/MLP probes on frozen features, last-token vs mean pooling | 08 | `probe.py` | H2 (+ validates the v2 pooling fix) |
| 3 | **Supervision granularity**: same X-LoRA classifier trained with pooled-CE vs per-token-CE vs NTP; identical measurement. Preceded by the **uniform-consistency check** (forced-uniform X-LoRA must match `set_adapters` uniform blend) | 09 | `routing_objectives.py` | H3 (+ localizes the v1 X-LoRA bug) |
| 4 | **Final evaluation**: 8 systems, info-parity prompts, per-example records, bootstrap CIs, paired tests | 10 | `final_eval.py`, `analysis.py` | H4 |

Execution order matters: E1 is the fork in the road. If the matrix is flat, E3/E4 are run
to *confirm* the mechanism and bound the null effect; if it shows a diagonal advantage, they
are run to *exploit* it.

## 4. Evaluation standards (apply to every experiment)

- **One prompt builder per system**, used for both generation and likelihood scoring
  (`prompts.py` is the single source of truth).
- **Information parity**: systems are labeled `informed` (gold emotion leaks in) or
  `uninformed`; cross-class comparisons are never reported as wins.
- **Stratified, seeded test sets** (`eval_core.sample_test_examples`): equal examples per
  cluster, filtered to conversations ending on an assistant turn.
- **Per-example persistence**: every (system, example) writes one JSONL row; all tables are
  derived offline by `analysis.py` with bootstrap 95% CIs and paired significance tests.
- **Seeded generation**, with seeds shared across systems per example so sampling noise is
  paired out.

## 5. What the report will contain

1. The v1 autopsy (one page): collapse signatures, the blend>oracle contradiction, the
   broken static PPL — as motivation for the redesign.
2. The specialization matrix heatmap + diagonal-advantage table (E1).
3. Probe accuracies with confusion matrices, last-token vs mean (E2).
4. The granularity result: routing entropy/accuracy/PPL per objective, with the 1/(T·L)
   gradient-dilution derivation (E3), plus the consistency-check outcome.
5. The final system table with CIs and the paired-test matrix (E4).
6. Discussion: when is mixture-of-LoRA routing worth it? What property of
   EmpatheticDialogues (shared empathetic register across emotions) predicts the answer?

## 6. Known limitations / honest caveats

- The ED "emotion" label describes the *speaker's situation*; listeners respond in a shared
  empathetic register, which is itself a candidate explanation for weak specialization.
- The 7-way emotion classifier metric remains a weak proxy even with information parity;
  BERTScore against the gold response is available in `eval_core.bertscore_f1` as a
  complementary reference-based metric.
- X-LoRA runs on an FP16 base while other systems run 4-bit (PEFT incompatibility), so
  X-LoRA is only ever compared against baselines on its own precision (notebook 09).
- PEFT internals were verified against `peft==0.18.1`; the `routing_objectives.py` docstring
  records exactly which behaviors were checked and why.
