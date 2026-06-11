"""QLoRA adapter training for one DESA emotion cluster."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from paths import CONFIG_PATH, OUTPUTS_DIR, SPLITS_DIR, ensure_project_dirs


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with Path(config_path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cluster_data(cluster_id: int, split: str = "train") -> list[dict]:
    """Load cluster_i_train.jsonl or cluster_i_val.jsonl."""
    suffix = "train" if split == "train" else "val"
    path = SPLITS_DIR / f"cluster_{cluster_id}_{suffix}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebook 01 first.")
    return load_jsonl(path)


def load_pooled_data(split: str = "train", seed: int = 42) -> list[dict]:
    """Cluster-balanced interleaving of all four clusters' conversations.

    Used to train the *pooled* control adapter for the specialization matrix:
    one generalist LoRA trained on data from all quadrants under the SAME
    example budget as a single expert (the budget cap is applied downstream by
    `build_sft_dataset`, so balance matters — round-robin interleaving makes
    the capped prefix draw ~equally from every cluster).
    """
    import random as _random

    per_cluster = []
    for cid in range(4):
        records = load_cluster_data(cid, split=split)
        _random.Random(seed + cid).shuffle(records)
        per_cluster.append(records)

    pooled: list[dict] = []
    for group in zip(*per_cluster):  # round-robin: 0,1,2,3,0,1,2,3,...
        pooled.extend(group)
    longest = max(len(records) for records in per_cluster)
    for idx in range(min(len(r) for r in per_cluster), longest):
        for records in per_cluster:
            if idx < len(records):
                pooled.append(records[idx])
    return pooled


def conversation_to_sft_texts(example: dict, tokenizer) -> list[str]:
    """Turn one conversation into one SFT example per assistant response."""
    utterances = [str(utt).strip() for utt in example.get("utterances", []) if str(utt).strip()]
    if len(utterances) < 2:
        return []

    texts: list[str] = []
    for idx in range(1, len(utterances), 2):
        messages = []
        for turn_idx, utterance in enumerate(utterances[: idx + 1]):
            role = "user" if turn_idx % 2 == 0 else "assistant"
            messages.append({"role": role, "content": utterance})

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Conservative fallback for environments with older tokenizer metadata.
            history = []
            for turn_idx, utterance in enumerate(utterances[:idx]):
                label = "User" if turn_idx % 2 == 0 else "Assistant"
                history.append(f"{label}: {utterance}")
            joined_history = "\n".join(history)
            text = f"<s>[INST] {joined_history} [/INST] {utterances[idx]}</s>"
        texts.append(text)
    return texts


def build_sft_dataset(
    records: list[dict],
    tokenizer,
    max_examples: int | None,
) -> Dataset:
    texts: list[str] = []
    for record in records:
        texts.extend(conversation_to_sft_texts(record, tokenizer))
        if max_examples and len(texts) >= max_examples:
            texts = texts[:max_examples]
            break
    return Dataset.from_dict({"text": texts})


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def summarize_parameter_dtypes(model, trainable_only: bool = False) -> dict[str, int]:
    counts = Counter()
    for parameter in model.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        counts[_dtype_name(parameter.dtype)] += parameter.numel()
    return dict(sorted(counts.items()))


def resolve_compute_dtype(config: dict) -> torch.dtype:
    dtype_name = config["model"].get("bnb_4bit_compute_dtype", "float16")
    if dtype_name == "bfloat16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            print("BF16 is not supported on this GPU; falling back to FP16.")
            return torch.float16
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(config: dict):
    compute_dtype = resolve_compute_dtype(config)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"BitsAndBytes compute dtype: {_dtype_name(compute_dtype)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["load_in_4bit"],
        bnb_4bit_quant_type=config["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["model"]["use_nested_quant"],
    )
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "dtype": compute_dtype,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(config["model"]["base_model"], **model_kwargs)
    except TypeError:
        # Older Transformers versions have not renamed torch_dtype to dtype yet.
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(config["model"]["base_model"], **model_kwargs)
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def build_lora_model(model, config: dict):
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        target_modules=config["lora"]["target_modules"],
        task_type=config["lora"]["task_type"],
    )
    return get_peft_model(model, lora_config)


def cast_trainable_parameters_to_fp32(model) -> None:
    """Keep QLoRA adapter weights in FP32 so AMP scaling never sees BF16 grads."""
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)


def build_sft_config(config: dict, output_dir: Path) -> SFTConfig:
    sft_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": config["training"]["num_train_epochs"],
        "per_device_train_batch_size": config["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "learning_rate": float(config["training"]["learning_rate"]),
        "warmup_ratio": float(config["training"]["warmup_ratio"]),
        "lr_scheduler_type": config["training"]["lr_scheduler_type"],
        "max_seq_length": config["training"]["max_seq_length"],
        # Do not enable Trainer AMP here. With QLoRA, bitsandbytes already controls
        # matmul compute dtype; Trainer fp16=True creates a GradScaler, and PyTorch
        # cannot unscale BF16 gradients in this CUDA path.
        "fp16": False,
        "bf16": False,
        "logging_steps": config["training"]["logging_steps"],
        "save_steps": config["training"]["save_steps"],
        "save_total_limit": config["training"]["save_total_limit"],
        "eval_strategy": config["training"]["eval_strategy"],
        "eval_steps": config["training"]["eval_steps"],
        "dataset_text_field": "text",
        "packing": False,
        "report_to": "none",
    }
    try:
        return SFTConfig(**sft_kwargs)
    except TypeError:
        sft_kwargs["max_length"] = sft_kwargs.pop("max_seq_length")
        return SFTConfig(**sft_kwargs)


def latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
    )
    return str(checkpoints[-1]) if checkpoints else None


def train(cluster_id: int, fresh: bool = False) -> Path:
    """Train one cluster adapter and return the final adapter path."""
    if cluster_id not in {0, 1, 2, 3}:
        raise ValueError("cluster_id must be one of 0, 1, 2, 3")

    ensure_project_dirs()
    set_seed(42 + cluster_id)
    config = load_config()
    output_dir = OUTPUTS_DIR / f"adapter_cluster_{cluster_id}"
    final_path = output_dir / "final"

    if fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and not fresh:
        print(f"Final adapter already exists at {final_path}. Use fresh=True to retrain.")
        return final_path

    print(f"\nTraining adapter for cluster {cluster_id}")
    print(f"Checkpoints -> {output_dir}")

    model, tokenizer = load_model_and_tokenizer(config)
    train_records = load_cluster_data(cluster_id, split="train")
    eval_records = load_cluster_data(cluster_id, split="val")
    train_dataset = build_sft_dataset(
        train_records,
        tokenizer,
        max_examples=config["training"].get("max_train_examples"),
    )
    eval_dataset = build_sft_dataset(
        eval_records,
        tokenizer,
        max_examples=config["training"].get("max_eval_examples"),
    )
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Eval examples:  {len(eval_dataset):,}")

    model = build_lora_model(model, config)
    cast_trainable_parameters_to_fp32(model)
    model.print_trainable_parameters()
    print("Trainable parameter dtypes:", summarize_parameter_dtypes(model, trainable_only=True))

    training_args = build_sft_config(config, output_dir)
    print(f"Trainer AMP flags: fp16={training_args.fp16}, bf16={training_args.bf16}")

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    checkpoint = latest_checkpoint(output_dir)
    print(f"Resume checkpoint: {checkpoint or 'none'}")
    trainer.train(resume_from_checkpoint=checkpoint)

    try:
        model.save_pretrained(str(final_path), save_embedding_layers=False)
    except TypeError:
        model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Adapter saved to {final_path}")
    return final_path


def train_pooled(fresh: bool = False) -> Path:
    """Train the pooled control adapter: all clusters' data, one expert's budget.

    Identical hyperparameters, LoRA rank, schedule, and `max_train_examples`
    cap as a single cluster expert — the ONLY difference is that the training
    data spans all four quadrants (balanced via `load_pooled_data`). This makes
    "pooled vs expert" a clean test of whether quadrant specialization buys
    anything at equal training cost.

    Saves to outputs/adapter_pooled/final/.
    """
    ensure_project_dirs()
    set_seed(46)  # distinct from the per-cluster seeds 42..45
    config = load_config()
    output_dir = OUTPUTS_DIR / "adapter_pooled"
    final_path = output_dir / "final"

    if fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and not fresh:
        print(f"Final pooled adapter already exists at {final_path}. Use fresh=True to retrain.")
        return final_path

    print("\nTraining POOLED adapter (all four clusters, single-expert budget)")
    print(f"Checkpoints -> {output_dir}")

    model, tokenizer = load_model_and_tokenizer(config)
    train_dataset = build_sft_dataset(
        load_pooled_data("train"),
        tokenizer,
        max_examples=config["training"].get("max_train_examples"),
    )
    eval_dataset = build_sft_dataset(
        load_pooled_data("val"),
        tokenizer,
        max_examples=config["training"].get("max_eval_examples"),
    )
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Eval examples:  {len(eval_dataset):,}")

    model = build_lora_model(model, config)
    cast_trainable_parameters_to_fp32(model)
    model.print_trainable_parameters()

    training_args = build_sft_config(config, output_dir)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    checkpoint = latest_checkpoint(output_dir)
    print(f"Resume checkpoint: {checkpoint or 'none'}")
    trainer.train(resume_from_checkpoint=checkpoint)

    try:
        model.save_pretrained(str(final_path), save_embedding_layers=False)
    except TypeError:
        model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Pooled adapter saved to {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_id", type=str, required=True, help="0..3, or 'pooled'")
    parser.add_argument("--fresh", action="store_true", help="Delete existing output dir before training.")
    args = parser.parse_args()
    if args.cluster_id == "pooled":
        train_pooled(fresh=args.fresh)
    else:
        train(int(args.cluster_id), fresh=args.fresh)
