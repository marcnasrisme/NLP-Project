import re
from collections import defaultdict

from datasets import load_dataset


def clean_text(text: str) -> str:
    """replace _comma_ tokens and normalize whitespace"""
    text = text.replace("_comma_", ",")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_splits(
    dataset_name: str = "facebook/empathetic_dialogues",
) -> dict[str, list[dict]]:
    """load EmpatheticDialogues and group into conversations per split"""
    ds = load_dataset(dataset_name)
    splits = {}
    for split_name in ("train", "validation", "test"):
        splits[split_name] = group_into_conversations(ds[split_name])
    return splits


def group_into_conversations(rows) -> list[dict]:
    """group flat utterance rows by conv_id into conversation dicts"""
    buckets: dict[str, list[dict]] = defaultdict(list)
    metadata: dict[str, dict] = {}

    for row in rows:
        cid = row["conv_id"]
        buckets[cid].append(row)
        if cid not in metadata:
            metadata[cid] = {
                "emotion": row["context"],
                "situation": clean_text(row["prompt"]),
            }

    conversations = []
    for cid, utterances in buckets.items():
        utterances.sort(key=lambda r: r["utterance_idx"])

        # the first utterance is always the emotion-sharer (speaker)
        speaker_val = utterances[0]["speaker_idx"]

        turns = []
        for row in utterances:
            role = "speaker" if row["speaker_idx"] == speaker_val else "listener"
            turns.append({
                "utterance_idx": row["utterance_idx"],
                "speaker": role,
                "text": clean_text(row["utterance"]),
            })

        conversations.append({
            "conv_id": cid,
            "emotion": metadata[cid]["emotion"],
            "situation": metadata[cid]["situation"],
            "turns": turns,
        })

    return conversations
