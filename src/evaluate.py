"""Evaluation suite for DESA systems."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline

from inference import build_messages_from_utterances, build_prompt
from paths import OUTPUTS_DIR

EMOTION_TO_BROAD = {
    "afraid": "fear",
    "angry": "anger",
    "annoyed": "anger",
    "anticipating": "neutral",
    "anxious": "fear",
    "apprehensive": "fear",
    "ashamed": "sadness",
    "caring": "joy",
    "confident": "joy",
    "content": "joy",
    "devastated": "sadness",
    "disappointed": "sadness",
    "disgusted": "disgust",
    "embarrassed": "sadness",
    "excited": "joy",
    "faithful": "joy",
    "furious": "anger",
    "grateful": "joy",
    "guilty": "sadness",
    "hopeful": "joy",
    "impressed": "surprise",
    "jealous": "anger",
    "joyful": "joy",
    "lonely": "sadness",
    "nostalgic": "sadness",
    "prepared": "neutral",
    "proud": "joy",
    "sad": "sadness",
    "sentimental": "sadness",
    "surprised": "surprise",
    "terrified": "fear",
    "trusting": "joy",
}

_EMOTION_CLASSIFIER = None


def load_test_examples(n_eval: int | None = None) -> list[dict]:
    """Load test examples from the same mirror used for training."""
    dataset = load_dataset("Estwld/empathetic_dialogues_llm", split="test")
    examples = [dict(example) for example in dataset]
    if n_eval:
        examples = examples[:n_eval]
    return examples


def gold_response(example: dict) -> str:
    utterances = example.get("utterances") or []
    if utterances:
        return str(utterances[-1])
    conversations = example.get("conversations") or example.get("messages") or []
    if conversations:
        return str(conversations[-1].get("content", ""))
    return ""


def distinct_n(texts: Iterable[str], n: int) -> float:
    ngrams = []
    for text in texts:
        tokens = str(text).split()
        ngrams.extend(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def _prompt_and_gold(example: dict, tokenizer) -> tuple[str, str]:
    history = build_messages_from_utterances(example.get("utterances", []), include_last=False)
    prompt = build_prompt(history, tokenizer=tokenizer, add_generation_prompt=True)
    return prompt, gold_response(example)


def _encode_prompt_response(tokenizer, prompt: str, response: str, device: str, max_length: int):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if not response_ids:
        return None

    if len(response_ids) >= max_length:
        input_ids = response_ids[:max_length]
        labels = input_ids.copy()
    else:
        prompt_budget = max_length - len(response_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": torch.tensor([input_ids], device=device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
        "labels": torch.tensor([labels], device=device),
    }


def compute_gold_perplexity(
    model,
    tokenizer,
    examples: list[dict],
    device: str | None = None,
    max_length: int = 512,
    pre_forward=None,
    context_manager=None,
    prompt_builder=None,
) -> float:
    """Perplexity of gold assistant response conditioned on conversation prompt."""
    model.eval()
    device = device or str(next(model.parameters()).device)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for example in examples:
            if pre_forward is not None:
                pre_forward(example)
            if prompt_builder is None:
                prompt, response = _prompt_and_gold(example, tokenizer)
            else:
                prompt = prompt_builder(example, tokenizer)
                response = gold_response(example)
            encoded = _encode_prompt_response(tokenizer, prompt, response, device, max_length)
            if encoded is None:
                continue
            labels = encoded.pop("labels")
            valid_tokens = int((labels != -100).sum().item())
            if valid_tokens == 0:
                continue
            if context_manager is not None:
                with context_manager(model):
                    outputs = model(**encoded, labels=labels)
            else:
                outputs = model(**encoded, labels=labels)
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens
    return math.exp(total_loss / total_tokens) if total_tokens else float("nan")


def get_emotion_classifier(model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
    global _EMOTION_CLASSIFIER
    if _EMOTION_CLASSIFIER is None:
        device = 0 if torch.cuda.is_available() else -1
        _EMOTION_CLASSIFIER = pipeline("text-classification", model=model_name, device=device)
    return _EMOTION_CLASSIFIER


def emotion_accuracy(generated_texts: list[str], gold_emotions: list[str]) -> float:
    """Compare generated broad emotion to ED-32 mapped broad gold label."""
    clf = get_emotion_classifier()
    correct = 0
    total = 0
    for text, gold in zip(generated_texts, gold_emotions):
        target = EMOTION_TO_BROAD.get(str(gold).lower().strip())
        if not target:
            continue
        pred = clf(str(text)[:512])[0]["label"].lower()
        correct += int(pred == target)
        total += 1
    return correct / total if total else 0.0


def _alpha_to_vector(alpha) -> np.ndarray:
    arr = np.asarray(alpha, dtype=float)
    if arr.size == 0:
        return np.zeros(4)
    if arr.ndim == 1:
        vector = arr
    else:
        vector = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    denom = vector.sum()
    return vector / denom if denom > 0 else vector


def gating_entropy(alpha_list: list[np.ndarray]) -> dict[str, float]:
    vectors = [_alpha_to_vector(alpha) for alpha in alpha_list if alpha is not None]
    if not vectors:
        return {"mean_entropy": float("nan"), "std_entropy": float("nan"), "max_entropy": math.log(4)}
    entropies = []
    for vector in vectors:
        clipped = np.clip(vector, 1e-9, 1.0)
        entropies.append(float(-np.sum(clipped * np.log(clipped))))
    return {
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "max_entropy": float(math.log(len(vectors[0]))),
    }


def gating_alignment(
    alpha_list: list[np.ndarray],
    gold_emotions: list[str],
    cluster_assignments: dict[str, int],
) -> dict[str, float]:
    """How much probability mass the gate assigns to the gold emotion cluster."""
    masses = []
    by_cluster: dict[int, list[np.ndarray]] = {idx: [] for idx in range(4)}
    for alpha, emotion in zip(alpha_list, gold_emotions):
        if alpha is None:
            continue
        cid = cluster_assignments.get(str(emotion).lower().strip())
        if cid is None:
            continue
        vector = _alpha_to_vector(alpha)
        masses.append(float(vector[cid]))
        by_cluster[cid].append(vector)
    out = {"gold_cluster_mass": float(np.mean(masses)) if masses else float("nan")}
    for cid, vectors in by_cluster.items():
        if vectors:
            mean_vec = np.mean(vectors, axis=0)
            for adapter_id, value in enumerate(mean_vec):
                out[f"cluster_{cid}_mean_alpha_{adapter_id}"] = float(value)
    return out


def unpack_generated(system_outputs: list) -> tuple[list[str], list[np.ndarray] | None]:
    texts = []
    alphas = []
    saw_alpha = False
    for item in system_outputs:
        if isinstance(item, tuple):
            texts.append(item[0])
            alphas.append(item[1])
            saw_alpha = True
        else:
            texts.append(item)
    return texts, alphas if saw_alpha else None


def evaluate_system(
    system_name: str,
    generated_outputs: list,
    gold_emotions: list[str],
    examples: list[dict],
    model=None,
    tokenizer=None,
    cluster_assignments: dict[str, int] | None = None,
    pre_forward=None,
    context_manager=None,
    prompt_builder=None,
) -> dict:
    print(f"\nEvaluating: {system_name}")
    generated_texts, alpha_list = unpack_generated(generated_outputs)
    results = {
        "system": system_name,
        "distinct_1": distinct_n(generated_texts, 1),
        "distinct_2": distinct_n(generated_texts, 2),
        "emotion_accuracy": emotion_accuracy(generated_texts, gold_emotions),
    }
    if model is not None and tokenizer is not None:
        results["perplexity"] = compute_gold_perplexity(
            model,
            tokenizer,
            examples,
            pre_forward=pre_forward,
            context_manager=context_manager,
            prompt_builder=prompt_builder,
        )
    if alpha_list is not None:
        results.update(gating_entropy(alpha_list))
        if cluster_assignments:
            results.update(gating_alignment(alpha_list, gold_emotions, cluster_assignments))
    print(json.dumps(results, indent=2))
    return results


def save_results(all_results: list[dict], output_dir: str | Path = OUTPUTS_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_results.json"
    csv_path = output_dir / "eval_results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    try:
        import pandas as pd

        pd.DataFrame(all_results).to_csv(csv_path, index=False)
    except Exception as exc:
        print(f"Could not write CSV: {exc}")
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    print("Run evaluation through notebooks/04_evaluation.ipynb")
