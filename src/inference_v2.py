"""Inference helpers for the v2 (corrected) gating heads.

Additive companion to `inference.py`. Provides:

- `load_turn_gate_v2`: load a `LastTokenGatingHead` from disk.
- `compute_turn_alpha_v2`: compute mixing weights using last-token pooling.
- `load_xlora_model_v2`: load X-LoRA with strict-mode classifier loading and
  before/after L2-norm reporting, so silent state-dict mismatches surface.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from peft import PeftConfig, XLoraConfig, get_peft_model
from transformers import AutoConfig

from gating_v2 import LastTokenGatingHead
from inference import (
    ADAPTER_NAMES,
    BASE_MODEL_ID,
    adapter_path,
    adapters_disabled,
    assert_adapters_exist,
    build_prompt,
    model_has_bnb_linear,
)
from paths import OUTPUTS_DIR


def _classifier_l2_norm(model) -> float:
    total = 0.0
    n = 0
    for name, param in model.named_parameters():
        if "xlora" in name.lower():
            total += float(param.detach().float().pow(2).sum().item())
            n += 1
    return math.sqrt(total) if n > 0 else float("nan")


def load_turn_gate_v2(
    path: Path = OUTPUTS_DIR / "turn_gate_v2.pt",
    hidden_dim: int = 4096,
) -> LastTokenGatingHead:
    if not path.exists():
        raise FileNotFoundError(f"v2 turn gate not found: {path}")
    gate = LastTokenGatingHead(hidden_dim=hidden_dim, num_adapters=4)
    state = torch.load(path, map_location="cpu")
    gate.load_state_dict(state)
    gate.eval()
    return gate


def compute_turn_alpha_v2(
    base_like_model,
    tokenizer,
    gate: LastTokenGatingHead,
    conversation_history: list[dict],
) -> np.ndarray:
    """Predict turn-level alpha from a conversation, using last-token pooling.

    Mirrors `inference.compute_turn_alpha` but uses the v2 head's pooling.
    """
    prompt = build_prompt(conversation_history, tokenizer=tokenizer, add_generation_prompt=True)
    device = next(base_like_model.parameters()).device
    gate = gate.to(device).eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad(), adapters_disabled(base_like_model):
        outputs = base_like_model(**inputs, output_hidden_states=True)
        alpha = gate(outputs.hidden_states[-1], inputs["attention_mask"])
    return alpha.squeeze(0).detach().float().cpu().numpy()


def load_xlora_model_v2(
    base_model,
    model_id: str = BASE_MODEL_ID,
    classifier_path: Path = OUTPUTS_DIR / "xlora_classifier_v2.pt",
    *,
    require_classifier: bool = True,
    min_relative_delta: float = 1e-3,
):
    """Load PEFT X-LoRA with strict, verified classifier loading.

    Differences vs `inference.load_xlora_model`:
    - `require_classifier=True` raises if `classifier_path` does not exist.
    - After load, asserts no unexpected keys (every saved key was consumed).
    - Reports L2 norm before/after; warns if the classifier weights look
      unchanged (would indicate the saved file matched random init or was
      ignored).
    """
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
    for idx, name in enumerate(ADAPTER_NAMES):
        if hasattr(model, "peft_config") and name not in model.peft_config:
            model.peft_config[name] = PeftConfig.from_pretrained(str(adapter_path(idx)))
        if hasattr(model, "base_model") and hasattr(model.base_model, "peft_config") and name not in model.base_model.peft_config:
            model.base_model.peft_config[name] = PeftConfig.from_pretrained(str(adapter_path(idx)))

    pre_norm = _classifier_l2_norm(model)
    print(f"v2 X-LoRA pre-load classifier L2 norm: {pre_norm:.4f}")

    if not classifier_path.exists():
        if require_classifier:
            raise FileNotFoundError(
                f"v2 X-LoRA classifier not found: {classifier_path}. Run notebook 05 to train it."
            )
        print(f"WARN: {classifier_path} not found; X-LoRA will run with random classifier init.")
        return model

    state = torch.load(classifier_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"v2 X-LoRA load: keys_in_file={len(state)} missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        # `unexpected` means the file contained keys the model does not expect, which
        # is a real mismatch (typo, wrong save format, version skew). Fail fast.
        raise RuntimeError(
            f"v2 X-LoRA classifier file has {len(unexpected)} unexpected key(s); "
            f"first few: {unexpected[:5]}"
        )

    post_norm = _classifier_l2_norm(model)
    rel_delta = abs(post_norm - pre_norm) / max(pre_norm, 1e-9)
    print(f"v2 X-LoRA post-load classifier L2 norm: {post_norm:.4f} (relative delta {rel_delta:.4%})")
    if rel_delta < min_relative_delta:
        print(
            f"WARNING: post-load classifier L2 norm barely changed (rel_delta={rel_delta:.4%}). "
            "This usually means the saved file matched random init, which is bad."
        )
    return model
