from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


def load_config(path: str = "configs/adapter_training.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_base_model(
    config: dict,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """load the base model in 4-bit quantization with its tokenizer"""
    model_name = config["model"]["name"]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def make_lora_config(config: dict) -> LoraConfig:
    """build a LoraConfig from the yaml config"""
    lora = config["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        task_type=lora["task_type"],
        bias="none",
    )


def attach_adapter(
    model: AutoModelForCausalLM, config: dict
) -> AutoModelForCausalLM:
    """attach a fresh LoRA adapter to the base model"""
    lora_config = make_lora_config(config)
    return get_peft_model(model, lora_config)


def make_training_args(config: dict, output_dir: str) -> SFTConfig:
    """build SFTConfig from the yaml config"""
    train = config["training"]
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train["epochs"],
        per_device_train_batch_size=train["batch_size"],
        gradient_accumulation_steps=train["gradient_accumulation_steps"],
        learning_rate=train["learning_rate"],
        warmup_ratio=train["warmup_ratio"],
        max_length=train["max_seq_length"],
        save_steps=train["save_steps"],
        logging_steps=train["logging_steps"],
        fp16=train.get("fp16", False),
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )


def train_adapter(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_examples: list[dict],
    output_dir: str,
    config: dict,
    resume_from_checkpoint: bool = True,
) -> None:
    """train a LoRA adapter using SFTTrainer with checkpoint/resume support"""
    train_dataset = Dataset.from_list(train_examples)
    training_args = make_training_args(config, output_dir)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # resume from last checkpoint if one exists
    checkpoint = None
    if resume_from_checkpoint:
        checkpoints = sorted(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            checkpoint = str(checkpoints[-1])

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_trained_adapter(
    base_model: AutoModelForCausalLM, adapter_path: str
) -> AutoModelForCausalLM:
    """load a saved LoRA adapter onto the base model"""
    return PeftModel.from_pretrained(base_model, adapter_path)
