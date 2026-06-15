# DESA — Dynamic Emotional-State Architecture

**Quantitative Methods for NLP, MIT Spring 2026**
Matthieu Hakim · Marc Nasr · Elie Juvenspan · Youngjae Shim

DESA implements dynamic emotion-gated LoRA blending for empathetic dialogue generation on Mistral-7B-Instruct. Four QLoRA adapters are fine-tuned on VAD-quadrant emotion clusters of EmpatheticDialogues, and a lightweight gating head learns to blend them dynamically per turn or per token (X-LoRA style).

> **Team project.** This is a four-person collaboration. Canonical team repo at [Matthieuhakim/DESA](https://github.com/Matthieuhakim/DESA). I owned cluster 1 (positive-low-arousal) adapter training, contributed the v2 corrected gating + evaluation, and designed and built the **v3 experiment redesign** hosted here: after both v1 routers collapsed, I reframed the project around a testable question and rebuilt the evaluation to research grade (per-example records, bootstrap CIs, paired tests).

## The v3 redesign — from "fix the router" to a research question

The v1 results contained a contradiction: a **collapsed, input-independent blend** of all four experts scored *better* gold-response perplexity (30.7) than an **oracle** that always picks the correct expert via the gold label (39.1). That's impossible if the experts are truly specialized — which means the premise behind routing was never tested. The project now asks:

> **When does learned routing over emotion-specialized LoRA experts actually help — and why did it fail here?**

Four hypotheses, four experiments (full design in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)):

| H | Question | Experiment |
| --- | --- | --- |
| H1 | Did the experts specialize at all? | **Specialization matrix**: expert × cluster PPL grid + a pooled-data control adapter at equal budget |
| H2 | Is the emotion quadrant even decodable from the frozen features the gate reads? | **Linear/MLP probes**, last-token vs mean pooling |
| H3 | Does supervision granularity cause router collapse? (pooled CE dilutes per-position gradients by 1/(T·L) ≈ 1/2560) | Train the same X-LoRA classifier under **pooled-CE vs per-token-CE vs NTP**, plus a uniform-consistency check that localizes a v1 integration bug |
| H4 | Does the learned gate beat trivial baselines under a fair evaluation? | **8-system comparison** with information-parity prompts, seeded generation, bootstrap CIs, paired tests |

## The architecture

1. **Cluster EmpatheticDialogues** into 4 VAD quadrants (NRC VAD lexicon → valence × arousal)
2. **Train 4 QLoRA adapters** on Mistral-7B-Instruct, one per cluster (rank 16, 4-bit), plus a **pooled control adapter** (all clusters' data, single-expert budget)
3. **Train gating** — two variants:
   - **Turn-level**: last-token hidden state → one alpha per turn (v2 head; v1's mean-pooled head collapsed)
   - **Token-level**: PEFT X-LoRA per-token, per-layer scaling, trained under three objectives
4. **Evaluate** with one prompt builder per system (generation *and* scoring), stratified seeded test sets, and per-example JSONL records

## Cluster assignments

The proposal defines clusters as VAD quadrants, not learned K-means groups:

| ID | Name | Rule | Example emotions | Owner |
| --- | --- | --- | --- | --- |
| 0 | positive_high_arousal | `valence >= 0.5`, `arousal >= 0.5` | excited, joyful, surprised | Matthieu |
| 1 | positive_low_arousal | `valence >= 0.5`, `arousal < 0.5` | content, grateful, trusting | **Marc** |
| 2 | negative_high_arousal | `valence < 0.5`, `arousal >= 0.5` | angry, terrified, anxious | Elie |
| 3 | negative_low_arousal | `valence < 0.5`, `arousal < 0.5` | sad, lonely | Youngjae |

## Run order

| Step | Notebook | Runtime | Answers | Output |
| --- | --- | --- | --- | --- |
| 1 | `01_clustering.ipynb` | CPU | data prep | cluster assignments, train/val splits |
| 2 | `02_train_adapters.ipynb` | GPU ×4 | the four experts | `outputs/adapter_cluster_<id>/final/` |
| 3 | `06_train_pooled_adapter.ipynb` | GPU | the generalist control | `outputs/adapter_pooled/final/` |
| 4 | `07_specialization_matrix.ipynb` | GPU ~20 min | **H1 — the fork in the road** | `outputs/experiments/specialization/` |
| 5 | `08_routing_probe.ipynb` | GPU ~30 min | **H2** | `outputs/experiments/probe/` |
| 6 | `09_routing_objectives.ipynb` | GPU FP16, heavy | **H3** + consistency check | `outputs/experiments/routing/` |
| 7 | `10_final_evaluation.ipynb` | GPU 2–4 h | **H4** | `outputs/experiments/final_eval/` |

Run 07 before committing GPU hours to 09/10 — its verdict decides which story the remaining experiments tell. Every experiment notebook persists per-example JSONL; report tables (CIs, paired tests) are derived offline with `src/analysis.py`, no GPU needed.

Legacy (v1/v2, superseded, kept for provenance): `03_train_gating.ipynb`, `04_evaluation.ipynb`, `05_corrected_gating_and_eval.ipynb`.

## Project structure

```
DESA/
├── RESEARCH_PLAN.md          v3 research question, hypotheses, experiment design
├── configs/qlora_config.yaml
├── src/
│   ├── clustering.py         NRC VAD -> deterministic K=4 quadrants
│   ├── train_adapter.py      QLoRA fine-tuning per cluster + pooled control
│   │
│   │   # v3 experiment stack (notebooks 06-10)
│   ├── prompts.py            canonical prompt builders — one format per system
│   ├── eval_core.py          per-example NLL, stratified sampling, bootstrap
│   ├── specialization.py     H1: expert x cluster PPL matrix
│   ├── probe.py              H2: routing-signal probes
│   ├── routing_objectives.py H3: X-LoRA objectives + consistency check
│   ├── final_eval.py         H4: fair 8-system comparison
│   ├── analysis.py           offline tables/CIs/plots from JSONL (no GPU)
│   │
│   │   # legacy v1/v2 (notebooks 03-05)
│   ├── gating.py             v1 turn gate (mean pooling; collapsed)
│   ├── gating_v2.py          v2 last-token gate (still used by notebook 10)
│   ├── inference.py          model loading + v1 systems (loaders reused by v3)
│   ├── inference_v2.py       v2 inference path
│   ├── evaluate.py           v1 metrics
│   ├── evaluate_v2.py        v2 evaluation
│   ├── colab_io.py           upload/download helpers
│   └── paths.py              local/Colab repo-root resolution
├── notebooks/                01-02 (data + experts), 03-05 (legacy), 06-10 (v3 experiments)
├── tests/smoke_test.py       local CPU tests for the GPU-free logic
├── data/                     cluster assignments and train/val splits
└── outputs/                  adapters, gating heads, experiments/ JSONL results
```

## Systems evaluated (notebook 10)

**Uninformed** — see only the conversation (the fair fight):
`base_chat` · `generic_empathy` (instruction, no label) · `pooled_adapter` · `uniform_blend` · `random_expert` · `turn_gate` (the DESA system)

**Informed** — gold emotion label leaks in (ceilings, reported separately):
`emotion_prompt` · `oracle_expert`

The decisive comparisons: `turn_gate` vs `uniform_blend` (does *learned* routing beat a constant blend?) and `turn_gate` vs `pooled_adapter` (does routing beat one generalist adapter?). v1 had neither baseline — and its static baseline was told the gold emotion, so its emotion-accuracy "win" measured prompt-following, not empathy.

Metrics: gold-response perplexity with bootstrap 95% CIs, paired-bootstrap NLL tests, Distinct-1/2, information-parity emotion accuracy, routing entropy/alignment/accuracy.

## Pre-trained adapters (run without retraining)

The four trained QLoRA adapters are published as a GitHub Release so you can run the
experiment notebooks (07–10) without redoing notebook 02:
[**releases/tag/adapters-v1**](https://github.com/marcnasrisme/DESA/releases/tag/adapters-v1).

You don't need to fetch them by hand — the notebooks call `ensure_adapters_present([0,1,2,3])`,
which now resolves adapters in this order: **already on disk → download from the Release →
prompt for manual upload**. So in a fresh Colab runtime the adapters download automatically.

To grab them manually:

```bash
for c in 0 1 2 3; do
  curl -L -o outputs/adapter_cluster_$c.zip \
    https://github.com/marcnasrisme/DESA/releases/download/adapters-v1/adapter_cluster_$c.zip
  unzip -o outputs/adapter_cluster_$c.zip -d .
done
```

(The pooled control adapter from notebook 06 will be added to the Release once trained. Model
weights stay out of git history — only small results live in the repo.)

## Colab quickstart

The laptop repository is the source of truth. Colab is only a temporary GPU runner.

1. In Colab: `File → Open notebook → GitHub → marcnasrisme/DESA`
2. For notebooks 02–10, switch runtime to GPU (A100 recommended; notebook 09 needs it)
3. Run the boot cell — it clones the repo, installs requirements, and imports `src/` modules
4. The adapter-check cell auto-downloads the pre-trained adapters from the Release (notebooks 07–10)
5. Run all cells; the final cell downloads new artifacts as a zip

## Built with

Python · PyTorch · HuggingFace (`transformers`, `peft`, `datasets`, `trl`, `accelerate`) · `bitsandbytes` (4-bit) · scikit-learn (probes) · pandas/matplotlib (analysis)

## References

- X-LoRA: Buehler & Buehler, APL Machine Learning, 2024
- LoRAMoE: Dou et al., 2024 · MoLE: Wu et al., ICLR 2024
- MoEL: Lin et al., EMNLP 2019
- MIME: Majumder et al., EMNLP 2020
- EmpatheticDialogues: Rashkin et al., ACL 2019
- NRC VAD Lexicon: Mohammad, ACL 2018
