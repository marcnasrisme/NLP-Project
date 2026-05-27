"""Generation utilities for the four DESA systems."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftConfig, PeftModel, XLoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gating import TurnLevelGatingHead
from paths import CLUSTER_DIR, OUTPUTS_DIR

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_NAMES = [f"cluster_{idx}" for idx in range(4)]


def load_base_model(
    model_id: str = BASE_MODEL_ID,
    device_map: str = "auto",
    quantized: bool | str = True,
    torch_dtype=torch.bfloat16,
):
    """Load Mistral and tokenizer for Colab inference/evaluation."""
    kwargs = {"device_map": device_map, "torch_dtype": torch_dtype}
    if quantized:
        quantization_mode = "4bit" if quantized is True else str(quantized).lower()
        if quantization_mode == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        elif quantization_mode == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("quantized must be False, True, '4bit', or '8bit'")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def adapter_path(cluster_id: int) -> Path:
    return OUTPUTS_DIR / f"adapter_cluster_{cluster_id}" / "final"


def adapter_is_complete(path: Path) -> bool:
    return (
        path.exists()
        and (path / "adapter_config.json").exists()
        and ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists())
    )


def assert_adapters_exist(cluster_ids: list[int] | None = None) -> None:
    cluster_ids = cluster_ids or [0, 1, 2, 3]
    missing = [adapter_path(cid) for cid in cluster_ids if not adapter_is_complete(adapter_path(cid))]
    if missing:
        raise FileNotFoundError(f"Missing or incomplete adapters: {missing}")


def model_has_bnb_linear(model) -> bool:
    return any(module.__class__.__name__ in {"Linear4bit", "Linear8bitLt"} for module in model.modules())


def load_multi_adapter_model(base_model, cluster_ids: list[int] | None = None):
    """Register all LoRA adapters on a single PeftModel."""
    cluster_ids = cluster_ids or [0, 1, 2, 3]
    assert_adapters_exist(cluster_ids)
    first = cluster_ids[0]
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path(first)),
        adapter_name=f"cluster_{first}",
        is_trainable=False,
    )
    for cid in cluster_ids[1:]:
        model.load_adapter(str(adapter_path(cid)), adapter_name=f"cluster_{cid}", is_trainable=False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def load_xlora_model(
    base_model,
    model_id: str = BASE_MODEL_ID,
    classifier_path: Path = OUTPUTS_DIR / "xlora_classifier.pt",
):
    """Load PEFT's native X-LoRA wrapper and optional trained classifier state."""
    assert_adapters_exist([0, 1, 2, 3])
    if model_has_bnb_linear(base_model):
        raise ValueError(
            "PEFT X-LoRA is not compatible with this bitsandbytes Linear base path. "
            "Load the X-LoRA base with load_base_model(..., quantized=False, torch_dtype=torch.float16)."
        )
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    model_config = AutoConfig.from_pretrained(model_id)
    xcfg = XLoraConfig(
        task_type="CAUSAL_LM",
        hidden_size=model_config.hidden_size,
        adapters={name: str(adapter_path(idx)) for idx, name in enumerate(ADAPTER_NAMES)},
        xlora_depth=4,
        xlora_size=2048,
        xlora_dropout_p=0.1,
        layerwise_scalings=True,
        enable_softmax=True,
        use_trainable_adapters=False,
    )
    model = get_peft_model(base_model, xcfg)
    # PEFT X-LoRA activates the named LoRA experts during forward passes. Some
    # versions do not mirror those expert configs into peft_config, which makes
    # disable_adapter_layers() raise KeyError for names like "cluster_0".
    for idx, name in enumerate(ADAPTER_NAMES):
        if hasattr(model, "peft_config") and name not in model.peft_config:
            model.peft_config[name] = PeftConfig.from_pretrained(str(adapter_path(idx)))
        if hasattr(model, "base_model") and hasattr(model.base_model, "peft_config") and name not in model.base_model.peft_config:
            model.base_model.peft_config[name] = PeftConfig.from_pretrained(str(adapter_path(idx)))
    if classifier_path.exists():
        state = torch.load(classifier_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded X-LoRA classifier. Missing={len(missing)} unexpected={len(unexpected)}")
    return model


def load_turn_gate(path: Path = OUTPUTS_DIR / "turn_gate.pt", hidden_dim: int = 4096):
    if not path.exists():
        raise FileNotFoundError(f"Turn gate not found: {path}")
    gate = TurnLevelGatingHead(hidden_dim=hidden_dim, num_adapters=4)
    state = torch.load(path, map_location="cpu")
    gate.load_state_dict(state)
    gate.eval()
    return gate


def load_cluster_assignments(path: Path = CLUSTER_DIR / "cluster_assignments.json") -> dict[str, int]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {emotion: int(cid) for emotion, cid in data["emotion_to_cluster"].items()}


def build_messages_from_utterances(utterances: list[str], include_last: bool = False) -> list[dict]:
    if not include_last and utterances:
        utterances = utterances[:-1]
    return [
        {"role": "user" if idx % 2 == 0 else "assistant", "content": str(utterance)}
        for idx, utterance in enumerate(utterances)
        if str(utterance).strip()
    ]


def build_prompt(conversation_history: list[dict], tokenizer=None, add_generation_prompt: bool = True) -> str:
    """Build a Mistral chat prompt from [{role, content}, ...]."""
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass

    labeled = []
    for turn in conversation_history:
        role = "User" if turn["role"] == "user" else "Assistant"
        labeled.append(f"{role}: {turn['content']}")
    return f"<s>[INST] {' '.join(labeled)} [/INST]"


def build_static_emotion_prompt(conversation_history: list[dict], emotion: str, tokenizer=None) -> str:
    """Build the same emotion-conditioned prompt used by the static baseline."""
    system_turn = {
        "role": "user",
        "content": (
            "You are an empathetic conversational assistant. "
            f"Respond with a tone aligned to the emotion: {emotion}.\n\n"
            + "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation_history)
        ),
    }
    return build_prompt([system_turn], tokenizer=tokenizer, add_generation_prompt=True)


@contextlib.contextmanager
def adapters_disabled(model):
    """Compatibility wrapper around PEFT adapter disabling APIs."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return
    yield


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def generate_static_prompt(
    model,
    tokenizer,
    conversation_history: list[dict],
    emotion: str,
    max_new_tokens: int = 100,
) -> str:
    prompt = build_static_emotion_prompt(conversation_history, emotion, tokenizer=tokenizer)
    with adapters_disabled(model):
        return _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)


def generate_argmax_adapter(
    model,
    tokenizer,
    conversation_history: list[dict],
    emotion: str,
    cluster_assignments: dict[str, int],
    max_new_tokens: int = 100,
) -> str:
    cluster_id = cluster_assignments.get(emotion.lower().strip(), 0)
    model.set_adapter(f"cluster_{cluster_id}")
    prompt = build_prompt(conversation_history, tokenizer=tokenizer, add_generation_prompt=True)
    return _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)


def compute_turn_alpha(base_like_model, tokenizer, gate, conversation_history: list[dict]) -> np.ndarray:
    prompt = build_prompt(conversation_history, tokenizer=tokenizer, add_generation_prompt=True)
    device = next(base_like_model.parameters()).device
    gate = gate.to(device).eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad(), adapters_disabled(base_like_model):
        outputs = base_like_model(**inputs, output_hidden_states=True)
        alpha = gate(outputs.hidden_states[-1], inputs["attention_mask"])
    return alpha.squeeze(0).detach().float().cpu().numpy()


def apply_weighted_adapters(model, weights: np.ndarray, adapter_name: str = "turn_blend") -> None:
    """Apply PEFT LoRA blending for System 3."""
    weights = [float(w) for w in weights]
    if hasattr(model, "set_adapters"):
        try:
            model.set_adapters(ADAPTER_NAMES, weights=weights)
            return
        except TypeError:
            pass

    if hasattr(model, "delete_adapter"):
        try:
            model.delete_adapter(adapter_name)
        except Exception:
            pass
    if hasattr(model, "add_weighted_adapter"):
        model.add_weighted_adapter(
            adapters=ADAPTER_NAMES,
            weights=weights,
            adapter_name=adapter_name,
            combination_type="linear",
        )
        model.set_adapter(adapter_name)
        return

    raise RuntimeError("This PEFT version does not expose weighted LoRA blending.")


def generate_turn_level(
    model,
    tokenizer,
    turn_gate: TurnLevelGatingHead,
    conversation_history: list[dict],
    max_new_tokens: int = 100,
) -> tuple[str, np.ndarray]:
    alpha = compute_turn_alpha(model, tokenizer, turn_gate, conversation_history)
    apply_weighted_adapters(model, alpha)
    prompt = build_prompt(conversation_history, tokenizer=tokenizer, add_generation_prompt=True)
    text = _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    return text, alpha


def generate_token_level(
    xlora_model,
    tokenizer,
    conversation_history: list[dict],
    max_new_tokens: int = 100,
) -> tuple[str, np.ndarray | None]:
    if hasattr(xlora_model, "enable_scalings_logging"):
        xlora_model.clear_scalings_log()
        xlora_model.enable_scalings_logging()
    prompt = build_prompt(conversation_history, tokenizer=tokenizer, add_generation_prompt=True)
    text = _generate(xlora_model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    alpha = None
    if hasattr(xlora_model, "get_latest_scalings"):
        latest = xlora_model.get_latest_scalings()
        if latest is not None:
            alpha = latest.detach().float().cpu().numpy()
    return text, alpha


def example_to_history(example: dict) -> tuple[list[dict], str]:
    utterances = example.get("utterances", [])
    return build_messages_from_utterances(utterances, include_last=False), example.get("emotion", "")


def run_all_systems(
    examples: list[dict],
    multi_adapter_model,
    xlora_model,
    tokenizer,
    turn_gate: TurnLevelGatingHead,
    cluster_assignments: dict[str, int],
    max_new_tokens: int = 100,
) -> dict:
    results = {
        "static_prompt": [],
        "argmax_adapter": [],
        "turn_level": [],
        "token_level": [],
    }
    for idx, example in enumerate(examples):
        history, emotion = example_to_history(example)
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(examples)}")
        results["static_prompt"].append(
            generate_static_prompt(multi_adapter_model, tokenizer, history, emotion, max_new_tokens)
        )
        results["argmax_adapter"].append(
            generate_argmax_adapter(
                multi_adapter_model,
                tokenizer,
                history,
                emotion,
                cluster_assignments,
                max_new_tokens,
            )
        )
        turn_text, turn_alpha = generate_turn_level(
            multi_adapter_model,
            tokenizer,
            turn_gate,
            history,
            max_new_tokens,
        )
        results["turn_level"].append((turn_text, turn_alpha))
        token_text, token_alpha = generate_token_level(xlora_model, tokenizer, history, max_new_tokens)
        results["token_level"].append((token_text, token_alpha))
    return results


if __name__ == "__main__":
    print("Run inference through notebooks/04_evaluation.ipynb")
