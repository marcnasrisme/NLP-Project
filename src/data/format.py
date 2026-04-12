def build_context(conversation: dict, up_to_turn: int) -> str:
    """build the conversation context string up to (not including) the target turn"""
    parts = [
        f"Situation: {conversation['situation']}",
        f"Emotion: {conversation['emotion']}",
        "",
    ]
    for turn in conversation["turns"][:up_to_turn]:
        role = "Speaker" if turn["speaker"] == "speaker" else "Listener"
        parts.append(f"{role}: {turn['text']}")
    return "\n".join(parts)


def apply_chat_template(context: str, response: str) -> str:
    """wrap context and response in Mistral-Instruct format"""
    return f"<s>[INST] {context} [/INST]{response}</s>"


def format_for_sft(
    conversations: list[dict],
    emotion_to_cluster: dict[str, int],
) -> list[dict]:
    """create one training example per listener turn in each conversation"""
    examples = []

    for conv in conversations:
        cluster_id = emotion_to_cluster.get(conv["emotion"])
        if cluster_id is None:
            continue

        for i, turn in enumerate(conv["turns"]):
            if turn["speaker"] != "listener":
                continue

            context = build_context(conv, up_to_turn=i)
            response = turn["text"]

            examples.append({
                "text": apply_chat_template(context, response),
                "context": context,
                "response": response,
                "emotion": conv["emotion"],
                "cluster_id": cluster_id,
                "conv_id": conv["conv_id"],
            })

    return examples
