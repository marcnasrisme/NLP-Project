"""Colab upload/download helpers for laptop-canonical artifact storage."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from paths import OUTPUTS_DIR, REPO_ROOT


def in_colab() -> bool:
    """Return True when running inside a Google Colab notebook."""
    return "google.colab" in sys.modules


def clone_repo(github_user: str, repo: str, branch: str = "main", token: str | None = None) -> Path:
    """Clone or update the repository in /content for Colab sessions."""
    if not in_colab():
        return REPO_ROOT

    target = Path("/content") / repo
    if token:
        url = f"https://{github_user}:{token}@github.com/{github_user}/{repo}.git"
    else:
        url = f"https://github.com/{github_user}/{repo}.git"

    if target.exists():
        subprocess.run(["git", "-C", str(target), "fetch", "origin", branch], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", branch], check=True)
        subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--branch", branch, url, str(target)], check=True)
    return target


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)


def upload_artifact(target_dir: Path, prompt: str = "Upload artifact zip") -> Path | None:
    """Upload a zip in Colab and extract it into target_dir."""
    if not in_colab():
        print(f"Not in Colab; expecting artifact already under {target_dir}")
        return None

    from google.colab import files

    print(prompt)
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No artifact uploaded.")

    uploaded_name = next(iter(uploaded))
    zip_path = Path(uploaded_name)
    _extract_zip(zip_path, target_dir)
    print(f"Extracted {zip_path} into {target_dir}")
    return zip_path


def upload_outputs(target_dir: Path = REPO_ROOT, prompt: str = "Upload outputs zip") -> Path | None:
    """Plan-compatible alias for uploading and extracting an artifact zip."""
    return upload_artifact(target_dir=target_dir, prompt=prompt)


def zip_outputs(paths: list[Path], zip_name: str) -> Path:
    """Zip selected output paths; safe in local shells and Colab."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUTPUTS_DIR / zip_name
    if zip_path.suffix != ".zip":
        zip_path = zip_path.with_suffix(".zip")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            path = Path(path)
            if not path.exists():
                print(f"Skipping missing path: {path}")
                continue
            if path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, file_path.relative_to(REPO_ROOT))
            else:
                zf.write(path, path.relative_to(REPO_ROOT))

    print(f"Created {zip_path}")
    return zip_path


def download_outputs(paths: list[Path], zip_name: str) -> Path:
    """Zip selected paths and download in Colab; locally just create the zip."""
    zip_path = zip_outputs(paths=paths, zip_name=zip_name)
    if in_colab():
        from google.colab import files

        files.download(str(zip_path))
    return zip_path


# Pre-trained adapters are published as GitHub Release assets so the experiment
# notebooks can run without retraining (notebook 02). See the release:
#   https://github.com/marcnasrisme/DESA/releases/tag/adapters-v1
ADAPTERS_RELEASE_BASE = "https://github.com/marcnasrisme/DESA/releases/download/adapters-v1"


def _adapter_complete(cid: int) -> bool:
    final = OUTPUTS_DIR / f"adapter_cluster_{cid}" / "final"
    has_cfg = (final / "adapter_config.json").exists()
    has_w = (final / "adapter_model.safetensors").exists() or (final / "adapter_model.bin").exists()
    return final.exists() and has_cfg and has_w


def download_adapters_from_release(cluster_ids: list[int], base_url: str = ADAPTERS_RELEASE_BASE) -> bool:
    """Download + extract the pre-trained adapter zips from the GitHub Release.

    Returns True if every requested adapter is present afterwards. Skips any
    adapter already on disk. Returns False (without raising) on any download
    error, so callers can fall back to the manual-upload path.
    """
    import urllib.error
    import urllib.request

    for cid in cluster_ids:
        if _adapter_complete(cid):
            print(f"Adapter {cid}: already present")
            continue
        url = f"{base_url}/adapter_cluster_{cid}.zip"
        zip_path = OUTPUTS_DIR / f"adapter_cluster_{cid}.zip"
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Adapter {cid}: downloading {url}")
            urllib.request.urlretrieve(url, zip_path)
        except (urllib.error.URLError, OSError) as exc:
            print(f"Adapter {cid}: download failed ({exc}).")
            return False
        _extract_zip(zip_path, REPO_ROOT)
        if not _adapter_complete(cid):
            print(f"Adapter {cid}: zip did not produce a complete adapter.")
            return False
        print(f"Adapter {cid}: OK ({OUTPUTS_DIR / f'adapter_cluster_{cid}' / 'final'})")
    return all(_adapter_complete(cid) for cid in cluster_ids)


def ensure_adapters_present(cluster_ids: list[int]) -> None:
    """Make sure each adapter exists: present on disk -> Release download -> manual upload."""
    # Fast path: already on disk (e.g. local run, or a re-run in the same session).
    if all(_adapter_complete(cid) for cid in cluster_ids):
        for cid in cluster_ids:
            print(f"Adapter {cid}: OK ({OUTPUTS_DIR / f'adapter_cluster_{cid}' / 'final'})")
        return

    # Preferred path: pull the pre-trained adapters from the GitHub Release.
    if download_adapters_from_release(cluster_ids):
        return

    # Fallback: prompt for manual upload (Colab) of any still-missing adapter.
    for cid in cluster_ids:
        if _adapter_complete(cid):
            continue
        print(f"Adapter {cid}: not on disk or release; upload it manually.")
        upload_artifact(
            target_dir=REPO_ROOT,
            prompt=f"Upload adapter_cluster_{cid}.zip from your laptop.",
        )
        if not _adapter_complete(cid):
            raise FileNotFoundError(
                f"Adapter {cid} still incomplete after upload. "
                "Zip paths should include outputs/adapter_cluster_<id>/final/..."
            )


def ensure_files_present(paths: list[Path], artifact_name: str) -> None:
    """Ensure arbitrary files exist, prompting for a zip when they are missing."""
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if not missing:
        for path in paths:
            print(f"Artifact OK: {path}")
        return

    print("Missing artifacts:")
    for path in missing:
        print(f"  - {path}")
    upload_artifact(REPO_ROOT, prompt=f"Upload {artifact_name}.zip from your laptop.")

    still_missing = [Path(p) for p in paths if not Path(p).exists()]
    if still_missing:
        raise FileNotFoundError(f"Still missing after upload: {still_missing}")


def clean_dir(path: Path) -> None:
    """Remove a directory if it exists."""
    if path.exists():
        shutil.rmtree(path)
