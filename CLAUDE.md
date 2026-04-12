# CLAUDE.md

## Project: Dynamic Emotion-Gated LoRA Blending for Empathetic Dialogue

MIT MBAn, Quantitative Methods for NLP, Spring 2026.
Team of 4, 5-week timeline. Compute: Google Colab Pro (A100/V100).

## What we're building

We train K=4 emotion-specialized QLoRA adapters on Mistral-7B-Instruct, then learn a lightweight gating head that dynamically blends them based on conversational context. The core research question: does turn-level or token-level gating work better for empathetic dialogue?

### Architecture overview

1. **Data**: EmpatheticDialogues (HuggingFace: `facebook/empathetic_dialogues`), 25k conversations, 32 emotion labels. We cluster these into K=4 groups using NRC VAD lexicon mappings (positive-high-arousal, positive-low-arousal, negative-high-arousal, negative-low-arousal).

2. **Adapter training**: 4 separate QLoRA adapters on Mistral-7B-Instruct (`mistralai/Mistral-7B-Instruct-v0.3`), each fine-tuned on its emotion cluster's dialogue subset using `peft` + `bitsandbytes` (4-bit) + `trl.SFTTrainer`.

3. **Gating**: freeze base model + all 4 adapters, train a small gating head that outputs blending weights alpha. Two variants:
   - **Turn-level**: mean-pool hidden states over last utterance, compute one alpha per turn
   - **Token-level**: per-token, per-layer scalings (X-LoRA style, reference: `xlora` library by EricLBuehler)

4. **Baselines**:
   - System 1: static prompt injection (no adapters)
   - System 2: argmax single-adapter selection per turn
   - System 3: turn-level gated blending
   - System 4: token-level gated blending

5. **Evaluation metrics**: perplexity, Distinct-1/2, emotion classification accuracy of generated responses, gating weight pattern analysis.

6. **Stretch goal**: VAD-conditioned turn-level gating where alpha_i = softmax(w_i^T E_t), with E_t updated via momentum from a GoEmotions classifier.

## Tech stack

- Python 3.10+
- PyTorch
- HuggingFace: `transformers`, `peft`, `datasets`, `trl`, `accelerate`
- `bitsandbytes` for 4-bit quantization
- `xlora` (reference implementation for token-level gating)
- `scikit-learn` for emotion clustering
- `wandb` for experiment tracking (optional)

## Repo structure

```
├── CLAUDE.md                  # this file
├── requirements.txt
├── configs/
│   ├── adapter_training.yaml  # QLoRA hyperparameters
│   ├── gating_training.yaml   # gating head hyperparameters
│   └── emotion_clusters.yaml  # cluster definitions and VAD mappings
├── src/
│   ├── data/
│   │   ├── load.py            # load and preprocess EmpatheticDialogues
│   │   ├── cluster.py         # emotion clustering via NRC VAD
│   │   └── format.py          # format dialogues for SFTTrainer
│   ├── models/
│   │   ├── adapter.py         # QLoRA adapter setup and training config
│   │   ├── gating.py          # turn-level and token-level gating heads
│   │   └── blending.py        # adapter blending logic (apply gated weights)
│   ├── eval/
│   │   ├── metrics.py         # perplexity, distinct-n, emotion accuracy
│   │   ├── generate.py        # generate responses from each system
│   │   └── analysis.py        # gating pattern visualization
│   └── baselines/
│       ├── prompt_injection.py # system 1: static prompt baseline
│       └── argmax_select.py   # system 2: single adapter selection
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_adapters.ipynb     # thin notebook: imports src, runs on Colab
│   ├── 03_train_gating.ipynb
│   └── 04_evaluate.ipynb
├── scripts/
│   ├── cluster_emotions.py    # standalone: run clustering, save to configs/
│   └── run_eval.py            # standalone: full evaluation pipeline
└── tests/
    ├── test_data.py
    ├── test_gating.py
    └── test_blending.py
```

## Code style

- minimal comments, lowercase comment starts, no em dashes
- human-sounding variable names, not overly abbreviated
- type hints on function signatures
- no boilerplate docstrings, just a one-liner if the function name isn't obvious
- prefer explicit over clever
- configs loaded from yaml, not hardcoded

## Key constraints

- Colab Pro GPU sessions have time limits. training code must checkpoint frequently and resume cleanly.
- Mistral-7B in 4-bit is ~5GB VRAM. with QLoRA overhead, fits comfortably on A100 40GB.
- notebooks should be thin wrappers. all logic lives in src/ so Claude Code can iterate on it locally.
- local testing uses tiny dummy data and cpu. never assume GPU availability in src/ modules.
- the team uses git with feature branches and PRs.

## Current status

Starting fresh. Need to scaffold the repo and begin with data loading + emotion clustering.

## References

- X-LoRA paper: Buehler & Buehler, APL Machine Learning, 2024
- X-LoRA code: github.com/EricLBuehler/xlora
- MoEL: Lin et al., EMNLP 2019
- MIME: Majumder et al., EMNLP 2020
- EmpatheticDialogues: Rashkin et al., ACL 2019
- NRC VAD Lexicon: Mohammad, ACL 2018
