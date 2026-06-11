# DESA — Dynamic Emotional-State Architecture

**Quantitative Methods for NLP, MIT Spring 2026**  
Matthieu Hakim · Marc Nasr · Elie Juvenspan · Youngjae Shim

DESA implements the proposal: dynamic emotion-gated LoRA blending for empathetic dialogue generation on Mistral-7B-Instruct.

**The project was redesigned after the v1 evaluation** (both routers collapsed and the
comparison itself was unfair — see `PROJECT_EXPLAINER.md` for the autopsy). The current
research question, hypotheses, and experiment design live in **`RESEARCH_PLAN.md`** — read
that first. Notebooks 03–05 are the legacy v1/v2 pipeline kept for provenance; the active
pipeline is notebooks 01, 02, then 06–10.

## Colab Quickstart

The laptop repository is the source of truth. Colab is only a temporary GPU runner. Google Drive is not used.

### One-Time Setup

1. Push this repository to GitHub.
2. If the repo is private, create a GitHub personal access token with `repo` access.
3. In Colab, open the Secrets panel and add `GITHUB_PAT` with notebook access enabled.

### Per Colab Session

1. Open the notebook from GitHub: `File -> Open notebook -> GitHub -> marcnasrisme/DESA`.
2. For notebooks 02-04, switch runtime to GPU. A100 is recommended.
3. Run the boot cell. It clones or pulls the repo into `/content/DESA`, installs the unpinned packages from `requirements.txt`, and imports local `src/` modules. Because the requirements are unpinned, Colab keeps package versions it already has.
4. For notebooks 03-04, upload the adapter/gating zips from your laptop when prompted.
5. Run all cells.
6. The final cell downloads new artifacts as a zip.

Copy-paste boot cell used by the notebooks:

```python
import importlib, os, subprocess, sys
from pathlib import Path


def _sh(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


if "google.colab" in sys.modules:
    GITHUB_USER = "marcnasrisme"  # change if the repo is under an org/team
    REPO = "DESA"                  # must match the GitHub repo name
    BRANCH = os.environ.get("DESA_GITHUB_BRANCH", "main")

    try:
        from google.colab import userdata
        try:
            token = userdata.get("GITHUB_PAT")
        except Exception:
            token = None
    except Exception:
        token = None

    repo_dir = Path("/content") / REPO
    if token in (None, ""):
        clone_url = f"https://github.com/{GITHUB_USER}/{REPO}.git"
    else:
        clone_url = f"https://{GITHUB_USER}:{token}@github.com/{GITHUB_USER}/{REPO}.git"

    if repo_dir.exists():
        _sh(["git", "-C", str(repo_dir), "fetch", "origin", BRANCH])
        _sh(["git", "-C", str(repo_dir), "checkout", "-B", BRANCH, f"origin/{BRANCH}"])
    else:
        _sh(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                BRANCH,
                clone_url,
                str(repo_dir),
            ]
        )

    os.environ["DESA_REPO_ROOT"] = str(repo_dir)
    os.chdir(repo_dir)
else:
    here = Path.cwd()
    if here.name == "notebooks":
        here = here.parent
    os.environ["DESA_REPO_ROOT"] = str(here)
    os.chdir(here)

# Import order matters: set DESA_REPO_ROOT before `paths` is first imported.
sys.path.insert(0, str(Path(os.environ["DESA_REPO_ROOT"]) / "src"))

_sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

# Reload paths after install (safe if first run, harmless if re-run).
if "paths" in sys.modules:
    import paths as _paths
    importlib.reload(_paths)
else:
    import paths as _paths

importlib.reload(_paths)

import torch
from colab_io import download_outputs, ensure_adapters_present, ensure_files_present, in_colab, upload_outputs, zip_outputs
from paths import CLUSTER_DIR, OUTPUTS_DIR, REPO_ROOT, SPLITS_DIR

print("REPO_ROOT:", REPO_ROOT)
print("CWD:", Path.cwd())
print("Colab:", in_colab())
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### After Colab Downloads

Move downloaded zips into this local repo and unzip them in place:

```bash
mv ~/Downloads/adapter_cluster_2.zip outputs/
unzip -o outputs/adapter_cluster_2.zip -d .
```

Small results such as `outputs/eval_results.json`, `outputs/eval_results.csv`, and plots are ignored by default with the rest of `outputs/`; use `git add -f` only if you intentionally want to version them. Adapter weights and checkpoints should stay ignored.

## Run Order

| Step | Notebook | Runtime | Answers | Output |
| --- | --- | --- | --- | --- |
| 1 | `01_clustering.ipynb` | CPU | — (data prep) | `data/cluster/cluster_assignments.json`, `data/splits/cluster_*_{train,val}.jsonl` |
| 2 | `02_train_adapters.ipynb` | GPU ×4 | — (the four experts) | `outputs/adapter_cluster_<id>/final/` |
| 3 | `06_train_pooled_adapter.ipynb` | GPU | — (the generalist control) | `outputs/adapter_pooled/final/` |
| 4 | `07_specialization_matrix.ipynb` | GPU | **Did the experts specialize? (H1)** | `outputs/experiments/specialization/` |
| 5 | `08_routing_probe.ipynb` | GPU | **Is the routing signal in the features? (H2)** | `outputs/experiments/probe/` |
| 6 | `09_routing_objectives.ipynb` | GPU (FP16, heavy) | **Does supervision granularity drive router collapse? (H3)** + the X-LoRA consistency check | `outputs/experiments/routing/` |
| 7 | `10_final_evaluation.ipynb` | GPU | **Does learned routing beat trivial baselines? (H4)** | `outputs/experiments/final_eval/` |

Run notebook 07 before committing GPU hours to 09/10 — it is the fork in the road (see
`RESEARCH_PLAN.md` §3). All experiment notebooks persist per-example JSONL; the report
tables (bootstrap CIs, paired tests) are derived offline with `src/analysis.py`, no GPU
needed.

Legacy (v1/v2, superseded, kept for provenance): `03_train_gating.ipynb`,
`04_evaluation.ipynb`, `05_corrected_gating_and_eval.ipynb`.

Notebook 02 is run once per cluster:

| Cluster | Name | Owner |
| --- | --- | --- |
| 0 | positive_high_arousal | Matthieu |
| 1 | positive_low_arousal | Marc |
| 2 | negative_high_arousal | Elie |
| 3 | negative_low_arousal | Youngjae |

The existing `outputs/adapter_cluster_0/checkpoint-*` was trained before the deterministic quadrant fix and should be retrained. Use `FRESH = True` in notebook 02 only when intentionally deleting and retraining a cluster's output directory.

## Project Structure

```text
DESA/
├── RESEARCH_PLAN.md          # research question, hypotheses, experiment design — read first
├── PROJECT_EXPLAINER.md      # autopsy of the v1 results (why the redesign)
├── requirements.txt
├── configs/
│   └── qlora_config.yaml
├── src/
│   ├── paths.py              # local/Colab repo-root resolution
│   ├── colab_io.py           # upload/download helpers
│   ├── clustering.py         # NRC VAD -> deterministic K=4 quadrants
│   ├── train_adapter.py      # QLoRA fine-tuning per cluster + pooled control adapter
│   │
│   │   # --- active experiment stack (notebooks 06-10) ---
│   ├── prompts.py            # canonical prompt builders (one format per system, gen + PPL)
│   ├── eval_core.py          # per-example NLL, stratified sampling, seeded gen, bootstrap
│   ├── specialization.py     # Experiment 1: expert × cluster PPL matrix (H1)
│   ├── probe.py              # Experiment 2: routing-signal probes (H2)
│   ├── routing_objectives.py # Experiment 3: X-LoRA objectives + consistency check (H3)
│   ├── final_eval.py         # Experiment 4: fair 8-system comparison (H4)
│   ├── analysis.py           # offline tables/CIs/plots from JSONL (no GPU)
│   │
│   │   # --- legacy v1/v2 (notebooks 03-05), kept for provenance ---
│   ├── gating.py             # v1 turn gate (mean pooling; collapsed)
│   ├── gating_v2.py          # v2 fixes (never run; superseded)
│   ├── inference.py          # model loading + v1 systems (loaders still used by v3)
│   ├── inference_v2.py       # v2 gate loading (turn gate still used by notebook 10)
│   ├── evaluate.py           # v1 metrics (EMOTION_TO_BROAD mapping still imported)
│   └── evaluate_v2.py        # v2 eval orchestration (superseded)
├── notebooks/
│   ├── 01_clustering.ipynb
│   ├── 02_train_adapters.ipynb
│   ├── 03..05_*.ipynb        # legacy
│   ├── 06_train_pooled_adapter.ipynb
│   ├── 07_specialization_matrix.ipynb
│   ├── 08_routing_probe.ipynb
│   ├── 09_routing_objectives.ipynb
│   └── 10_final_evaluation.ipynb
├── data/
│   ├── cluster/
│   └── splits/
└── outputs/
    ├── adapter_cluster_{0..3}/
    ├── adapter_pooled/
    └── experiments/
        ├── specialization/   # per_example_nll.jsonl, matrix.json, heatmap
        ├── probe/            # features_{train,val}.npz, probe_results.json
        ├── routing/          # router_{objective}.pt, routing_objectives.json, consistency_check.json
        └── final_eval/       # per_example.jsonl, ppl_comparison.png
```

## Cluster Assignments

The proposal defines the clusters as VAD quadrants, not learned K-means groups:

| ID | Name | Rule | Example emotions |
| --- | --- | --- | --- |
| 0 | positive_high_arousal | `valence >= 0.5`, `arousal >= 0.5` | excited, joyful, surprised |
| 1 | positive_low_arousal | `valence >= 0.5`, `arousal < 0.5` | content, grateful, trusting |
| 2 | negative_high_arousal | `valence < 0.5`, `arousal >= 0.5` | angry, terrified, anxious |
| 3 | negative_low_arousal | `valence < 0.5`, `arousal < 0.5` | sad, lonely |

## Systems Evaluated (notebook 10)

Uninformed (see only the conversation): `base_chat`, `generic_empathy` (instruction, no
label), `pooled_adapter`, `uniform_blend`, `random_expert`, `turn_gate` (the DESA system).
Informed (gold emotion label leaks in, reported separately): `emotion_prompt`,
`oracle_expert`. X-LoRA routers are evaluated in notebook 09 on their own (FP16) precision.

Metrics: gold-response perplexity with bootstrap 95% CIs, paired-bootstrap NLL tests
between systems, Distinct-1/2, information-parity emotion-classification accuracy, routing
entropy/alignment/accuracy. Every metric is derived from per-example JSONL records by
`src/analysis.py`.
