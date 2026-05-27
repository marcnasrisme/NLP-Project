# DESA — Dynamic Emotional-State Architecture

**Quantitative Methods for NLP, MIT Spring 2026**
Matthieu Hakim · Marc Nasr · Elie Juvenspan · Youngjae Shim

DESA implements dynamic emotion-gated LoRA blending for empathetic dialogue generation on Mistral-7B-Instruct. Four QLoRA adapters are fine-tuned on VAD-quadrant emotion clusters of EmpatheticDialogues, and a lightweight gating head learns to blend them dynamically per turn or per token (X-LoRA style).

> **Team project.** This is a four-person collaboration. Canonical team repo at [Matthieuhakim/DESA](https://github.com/Matthieuhakim/DESA). I owned cluster 1 (positive-low-arousal) adapter training, contributed the v2 corrected gating + evaluation, and fixed the `LastTokenGatingHead` BF16 dtype mismatch. Most code was pair-programmed and committed under Matthieu's account; this mirror exists so my profile reflects the work.

## The architecture

1. **Cluster EmpatheticDialogues** into 4 VAD quadrants (NRC VAD lexicon → valence × arousal)
2. **Train 4 QLoRA adapters** on Mistral-7B-Instruct, one per cluster (rank 16, 4-bit)
3. **Train a gating head** — two variants:
   - **Turn-level**: mean-pool hidden states over the last utterance, output one alpha per turn
   - **Token-level**: PEFT X-LoRA per-token, per-layer scaling
4. **Evaluate** against (a) static prompt baseline, (b) argmax single-adapter selection, (c) turn-level gating, (d) X-LoRA token-level

Metrics: gold-response perplexity, Distinct-1/2, emotion classification accuracy of generated responses, gating entropy, gating alignment with gold cluster.

## Cluster assignments

The proposal defines clusters as VAD quadrants, not learned K-means groups:

| ID | Name | Rule | Example emotions | Owner |
| --- | --- | --- | --- | --- |
| 0 | positive_high_arousal | `valence >= 0.5`, `arousal >= 0.5` | excited, joyful, surprised | Matthieu |
| 1 | positive_low_arousal | `valence >= 0.5`, `arousal < 0.5` | content, grateful, trusting | **Marc** |
| 2 | negative_high_arousal | `valence < 0.5`, `arousal >= 0.5` | angry, terrified, anxious | Elie |
| 3 | negative_low_arousal | `valence < 0.5`, `arousal < 0.5` | sad, lonely | Youngjae |

## Run order

| Step | Notebook | Runtime | Output |
| --- | --- | --- | --- |
| 1 | `notebooks/01_clustering.ipynb` | CPU | cluster assignments, train/val splits, visualization |
| 2 | `notebooks/02_train_adapters.ipynb` | GPU | `outputs/adapter_cluster_<id>/final/` + zip |
| 3 | `notebooks/03_train_gating.ipynb` | GPU | `turn_gate.pt`, `xlora_classifier.pt`, gating zip |
| 4 | `notebooks/04_evaluation.ipynb` | GPU | `eval_results.json`, `eval_results.csv`, comparison plot |
| 5 | `notebooks/05_corrected_gating_and_eval.ipynb` | GPU | v2 corrected gating + entropy-regularized re-evaluation |

## Project structure

```
DESA/
├── configs/qlora_config.yaml
├── src/
│   ├── clustering.py         NRC VAD -> deterministic K=4 quadrants
│   ├── train_adapter.py      QLoRA fine-tuning per cluster
│   ├── gating.py             turn-level gate
│   ├── gating_v2.py          corrected gating with last-token pooling + entropy regularization
│   ├── inference.py          four proposal systems
│   ├── inference_v2.py       v2 inference path
│   ├── evaluate.py           PPL, Distinct-1/2, emotion accuracy, gating analysis
│   ├── evaluate_v2.py        v2 evaluation
│   ├── colab_io.py           upload/download helpers
│   └── paths.py              local/Colab repo-root resolution
├── notebooks/                01–05 (clustering through corrected gating)
├── data/                     cluster assignments and train/val splits
└── outputs/                  trained adapters, gating heads, eval results
```

## Systems evaluated

1. Static emotion prompt on Mistral-7B-Instruct (no adapters)
2. Argmax adapter selection from gold emotion cluster
3. Turn-level learned blending — mean-pooled hidden state → adapter weights
4. PEFT X-LoRA token-level, layerwise blending

## Colab quickstart

The laptop repository is the source of truth. Colab is only a temporary GPU runner.

1. In Colab: `File → Open notebook → GitHub → marcnasrisme/DESA`
2. For notebooks 02–05, switch runtime to GPU (A100 recommended)
3. Run the boot cell — it clones the repo, installs requirements, and imports `src/` modules
4. For notebooks 03–04, upload the adapter/gating zips when prompted
5. Run all cells; the final cell downloads new artifacts as a zip

## Built with

Python · PyTorch · HuggingFace (`transformers`, `peft`, `datasets`, `trl`, `accelerate`) · `bitsandbytes` (4-bit) · `xlora` (token-level gating reference)

## References

- X-LoRA: Buehler & Buehler, APL Machine Learning, 2024
- MoEL: Lin et al., EMNLP 2019
- MIME: Majumder et al., EMNLP 2020
- EmpatheticDialogues: Rashkin et al., ACL 2019
- NRC VAD Lexicon: Mohammad, ACL 2018
