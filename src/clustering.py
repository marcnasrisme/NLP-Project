"""Emotion clustering for DESA.

The proposal defines four emotion clusters as valence/arousal quadrants:

0. positive_high_arousal
1. positive_low_arousal
2. negative_high_arousal
3. negative_low_arousal

This module uses that deterministic rule directly instead of K-means so the
cluster IDs remain stable, balanced, and semantically named across machines.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset

from paths import CLUSTER_DIR, SPLITS_DIR, ensure_project_dirs


NRC_VAD = {
    "afraid": [0.139, 0.654, 0.208],
    "angry": [0.167, 0.865, 0.657],
    "annoyed": [0.293, 0.608, 0.491],
    "anticipating": [0.620, 0.691, 0.606],
    "anxious": [0.268, 0.739, 0.267],
    "apprehensive": [0.263, 0.580, 0.308],
    "ashamed": [0.146, 0.542, 0.192],
    "caring": [0.787, 0.508, 0.629],
    "confident": [0.827, 0.600, 0.862],
    "content": [0.775, 0.380, 0.645],
    "devastated": [0.043, 0.674, 0.130],
    "disappointed": [0.237, 0.448, 0.322],
    "disgusted": [0.133, 0.673, 0.511],
    "embarrassed": [0.243, 0.596, 0.238],
    "excited": [0.840, 0.857, 0.701],
    "faithful": [0.780, 0.420, 0.650],
    "furious": [0.121, 0.922, 0.706],
    "grateful": [0.862, 0.467, 0.650],
    "guilty": [0.157, 0.548, 0.213],
    "hopeful": [0.782, 0.600, 0.583],
    "impressed": [0.750, 0.650, 0.550],
    "jealous": [0.213, 0.680, 0.420],
    "joyful": [0.922, 0.784, 0.750],
    "lonely": [0.171, 0.404, 0.206],
    "nostalgic": [0.600, 0.370, 0.490],
    "prepared": [0.680, 0.500, 0.720],
    "proud": [0.849, 0.680, 0.826],
    "sad": [0.172, 0.420, 0.224],
    "sentimental": [0.570, 0.380, 0.430],
    "surprised": [0.650, 0.820, 0.530],
    "terrified": [0.051, 0.820, 0.130],
    "trusting": [0.776, 0.430, 0.660],
}

CLUSTER_NAMES = {
    0: "positive_high_arousal",
    1: "positive_low_arousal",
    2: "negative_high_arousal",
    3: "negative_low_arousal",
}


def build_vad_matrix() -> pd.DataFrame:
    """Return a DataFrame indexed by emotion with valence/arousal/dominance."""
    records = [
        {"emotion": emotion, "valence": vad[0], "arousal": vad[1], "dominance": vad[2]}
        for emotion, vad in NRC_VAD.items()
    ]
    return pd.DataFrame(records).set_index("emotion").sort_index()


def emotion_to_quadrant(valence: float, arousal: float) -> int:
    """Map valence/arousal to the proposal's fixed cluster IDs."""
    if valence >= 0.5 and arousal >= 0.5:
        return 0
    if valence >= 0.5 and arousal < 0.5:
        return 1
    if valence < 0.5 and arousal >= 0.5:
        return 2
    return 3


def cluster_emotions(k: int = 4, random_state: int | None = None):
    """Return deterministic emotion -> cluster assignments.

    The `k` and `random_state` arguments are accepted for backward-compatible
    notebook calls; only `k=4` is supported because the proposal fixes K=4.
    """
    if k != 4:
        raise ValueError("DESA uses the proposal's fixed K=4 quadrant clusters.")

    df = build_vad_matrix()
    assignments = {
        emotion: emotion_to_quadrant(row.valence, row.arousal)
        for emotion, row in df.iterrows()
    }
    return assignments, None, df


def label_clusters(assignments: dict[str, int] | None = None, df: pd.DataFrame | None = None):
    """Return the fixed cluster ID -> human-readable name mapping."""
    return CLUSTER_NAMES.copy()


def _normalise_messages(example: dict) -> list[str]:
    conversations = example.get("conversations") or example.get("messages") or []
    if conversations:
        return [turn.get("content", "").strip() for turn in conversations if turn.get("content")]
    utterances = example.get("utterances") or []
    return [str(utt).strip() for utt in utterances if str(utt).strip()]


def _serialise_examples(path: Path, examples: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def split_dataset_by_cluster(
    assignments: dict[str, int],
    output_dir: str | Path = SPLITS_DIR,
    val_fraction: float = 0.1,
    seed: int = 42,
):
    """Split EmpatheticDialogues train data by cluster, with train/val files."""
    ensure_project_dirs()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("Estwld/empathetic_dialogues_llm", split="train")
    cluster_data: dict[int, list[dict]] = {i: [] for i in range(4)}

    for example in dataset:
        emotion = str(example["emotion"]).lower().strip()
        if emotion not in assignments:
            continue
        utterances = _normalise_messages(example)
        if len(utterances) < 2:
            continue
        cluster_data[assignments[emotion]].append(
            {
                "conv_id": example.get("conv_id", ""),
                "emotion": emotion,
                "cluster_id": assignments[emotion],
                "cluster_name": CLUSTER_NAMES[assignments[emotion]],
                "situation": example.get("situation", ""),
                "utterances": utterances,
            }
        )

    rng = random.Random(seed)
    for cid, examples in cluster_data.items():
        rng.shuffle(examples)
        val_size = max(1, int(len(examples) * val_fraction))
        val_examples = examples[:val_size]
        train_examples = examples[val_size:]

        _serialise_examples(output_dir / f"cluster_{cid}.jsonl", examples)
        _serialise_examples(output_dir / f"cluster_{cid}_train.jsonl", train_examples)
        _serialise_examples(output_dir / f"cluster_{cid}_val.jsonl", val_examples)
        print(
            f"Cluster {cid} ({CLUSTER_NAMES[cid]}): "
            f"{len(train_examples)} train / {len(val_examples)} val -> {output_dir}"
        )

    return cluster_data


def save_assignments(
    assignments: dict[str, int],
    cluster_names: dict[int, str] | None = None,
    output_dir: str | Path = CLUSTER_DIR,
) -> Path:
    """Save cluster metadata to data/cluster/cluster_assignments.json."""
    ensure_project_dirs()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_names = cluster_names or CLUSTER_NAMES
    payload = {
        "method": "vad_quadrant",
        "thresholds": {"valence": 0.5, "arousal": 0.5},
        "emotion_to_cluster": assignments,
        "cluster_to_name": {str(k): v for k, v in cluster_names.items()},
        "cluster_to_emotions": {
            str(cid): sorted([emotion for emotion, assigned in assignments.items() if assigned == cid])
            for cid in range(4)
        },
    }
    out_path = output_dir / "cluster_assignments.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved cluster assignments to {out_path}")
    return out_path


if __name__ == "__main__":
    assignments, _, df = cluster_emotions(k=4)
    cluster_names = label_clusters(assignments, df)
    print("\nEmotion -> cluster assignments:")
    for emotion, cid in sorted(assignments.items(), key=lambda x: (x[1], x[0])):
        print(f"  {emotion:20s} -> {cid} ({cluster_names[cid]})")
    save_assignments(assignments, cluster_names)
    split_dataset_by_cluster(assignments)
