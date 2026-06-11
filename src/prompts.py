"""Canonical prompt construction for DESA — single source of truth.

Why this module exists
----------------------
The v1 evaluation produced a perplexity of ~22,866 for the `static_prompt`
system because *generation* used an instruction-wrapped prompt while
*perplexity* was computed against a plain chat-template prompt (and with an
undefined adapter state). The lesson: every system must use ONE prompt format,
defined in ONE place, for both generation and likelihood scoring.

All new code (specialization matrix, probes, routing objectives, final eval)
builds prompts exclusively through this module. The format matches what the
adapters were trained on (`train_adapter.conversation_to_sft_texts` applies the
tokenizer chat template to alternating user/assistant turns), so likelihoods
are measured in-distribution for the fine-tuned experts.

EmpatheticDialogues conventions used throughout:
- `utterances[0]` is the *speaker* (the person with the emotional situation)
  -> role "user". Odd indices are the *listener* -> role "assistant".
- The gold response for evaluation is the final utterance, which must be a
  listener (assistant) turn. `is_valid_eval_example` enforces this.

This module deliberately has no torch / peft imports so it can be used in
lightweight analysis contexts too.
"""

from __future__ import annotations


def history_messages(utterances: list, include_last: bool = False) -> list[dict]:
    """Convert raw ED utterances to chat messages, optionally dropping the gold turn.

    With `include_last=False` (the default for evaluation), the final utterance —
    the gold assistant response — is removed, leaving the conversation context
    the model must respond to.
    """
    cleaned = [str(utt).strip() for utt in utterances if str(utt).strip()]
    if not include_last and cleaned:
        cleaned = cleaned[:-1]
    return [
        {"role": "user" if idx % 2 == 0 else "assistant", "content": utterance}
        for idx, utterance in enumerate(cleaned)
    ]


def is_valid_eval_example(example: dict) -> bool:
    """True when the example ends on an assistant turn we can score.

    Requires at least one user turn plus the gold assistant reply, and an even
    utterance count so the final utterance is a listener (assistant) turn.
    Odd-length conversations end on a *user* turn; scoring that as the "gold
    response" would be measuring the wrong distribution entirely.
    """
    utterances = [str(u).strip() for u in example.get("utterances", []) if str(u).strip()]
    return len(utterances) >= 2 and len(utterances) % 2 == 0


def gold_response(example: dict) -> str:
    """The final (assistant) utterance — the reference the systems are scored against."""
    utterances = [str(u).strip() for u in example.get("utterances", []) if str(u).strip()]
    return utterances[-1] if utterances else ""


def chat_prompt(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Apply the model's chat template, with a Mistral-style fallback."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception:
        labeled = " ".join(
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
            for turn in messages
        )
        return f"<s>[INST] {labeled} [/INST]"


def _prepend_to_first_user_turn(messages: list[dict], instruction: str) -> list[dict]:
    """Inject an instruction while keeping the multi-turn chat structure intact.

    Mistral-Instruct has no system role. The v1 static baseline instead crammed
    the whole conversation into one giant user message, which is far outside the
    chat format the model was tuned on — that is what inflated its perplexity.
    Prepending the instruction to the first user turn keeps every other turn in
    its native position.
    """
    if not messages:
        return [{"role": "user", "content": instruction}]
    out = [dict(turn) for turn in messages]
    first = out[0]
    if first["role"] == "user":
        first["content"] = f"{instruction}\n\n{first['content']}"
    else:
        out.insert(0, {"role": "user", "content": instruction})
    return out


# ---------------------------------------------------------------------------
# The three prompt variants used by the final evaluation
# ---------------------------------------------------------------------------


def build_vanilla_prompt(example: dict, tokenizer) -> str:
    """Plain chat history, no instruction. Used by all adapter-routed systems.

    This is the *uninformed* condition: the model sees only the conversation,
    exactly as during adapter training.
    """
    return chat_prompt(tokenizer, history_messages(example.get("utterances", [])))


def build_generic_empathy_prompt(example: dict, tokenizer) -> str:
    """Generic empathy instruction WITHOUT the gold emotion. Uninformed.

    This is the fair prompt-engineering baseline for the adapters: it gets the
    same information the routed systems get (none beyond the conversation), so
    any gap between it and the adapter systems is attributable to fine-tuning,
    not to leaked labels.
    """
    instruction = (
        "You are an empathetic conversational partner. Read the conversation "
        "carefully, acknowledge how the other person feels, and respond warmly "
        "and supportively in one or two sentences."
    )
    messages = _prepend_to_first_user_turn(
        history_messages(example.get("utterances", [])), instruction
    )
    return chat_prompt(tokenizer, messages)


def build_emotion_informed_prompt(example: dict, tokenizer) -> str:
    """Empathy instruction that NAMES the gold emotion. Informed (label leak).

    Kept as a reference point only: it sees the gold label, so it competes in a
    different class from the uninformed systems and must be reported separately.
    """
    emotion = str(example.get("emotion", "")).strip() or "the speaker's emotion"
    instruction = (
        "You are an empathetic conversational partner. The other person is "
        f"feeling {emotion}. Respond with a tone aligned to that emotion, "
        "warmly and supportively, in one or two sentences."
    )
    messages = _prepend_to_first_user_turn(
        history_messages(example.get("utterances", [])), instruction
    )
    return chat_prompt(tokenizer, messages)
