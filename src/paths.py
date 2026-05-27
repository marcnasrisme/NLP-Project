"""Repository-relative paths for local and Colab execution.

Colab VMs are ephemeral. The laptop repository remains canonical; notebooks clone
the repo into /content/DESA and download artifacts back at session end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def in_colab() -> bool:
    """Return True when running inside a Google Colab notebook."""
    return "google.colab" in sys.modules


def get_repo_root() -> Path:
    """Resolve the DESA repository root in local shells and Colab runtimes."""
    if env_root := os.environ.get("DESA_REPO_ROOT"):
        return Path(env_root).expanduser().resolve()
    if in_colab():
        return Path("/content/DESA")
    return Path(__file__).resolve().parents[1]


REPO_ROOT = get_repo_root()
SRC_DIR = REPO_ROOT / "src"
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
CLUSTER_DIR = DATA_DIR / "cluster"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = REPO_ROOT / "outputs"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
CONFIG_PATH = CONFIG_DIR / "qlora_config.yaml"


def ensure_project_dirs() -> None:
    """Create mutable project directories if they do not exist."""
    for path in (DATA_DIR, CLUSTER_DIR, SPLITS_DIR, OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
