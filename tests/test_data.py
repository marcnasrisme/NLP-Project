from unittest.mock import patch, MagicMock

import pytest

from src.data.load import clean_text, group_into_conversations
from src.data.cluster import get_vad_mappings, build_emotion_clusters, save_clusters, load_clusters
from src.data.format import format_for_sft, build_context, apply_chat_template


# ---- fixtures ----

MOCK_ROWS = [
    {
        "conv_id": "conv_1", "utterance_idx": 1, "context": "joyful",
        "prompt": "I got a promotion", "speaker_idx": 0,
        "utterance": "I just got promoted_comma_ I'm thrilled!",
        "selfeval": "", "tags": "",
    },
    {
        "conv_id": "conv_1", "utterance_idx": 2, "context": "joyful",
        "prompt": "I got a promotion", "speaker_idx": 1,
        "utterance": "That's wonderful news!",
        "selfeval": "", "tags": "",
    },
    {
        "conv_id": "conv_1", "utterance_idx": 3, "context": "joyful",
        "prompt": "I got a promotion", "speaker_idx": 0,
        "utterance": "Thanks_comma_ I worked really hard",
        "selfeval": "", "tags": "",
    },
    {
        "conv_id": "conv_1", "utterance_idx": 4, "context": "joyful",
        "prompt": "I got a promotion", "speaker_idx": 1,
        "utterance": "You deserve it!",
        "selfeval": "", "tags": "",
    },
    {
        "conv_id": "conv_2", "utterance_idx": 1, "context": "sad",
        "prompt": "My dog passed away", "speaker_idx": 0,
        "utterance": "I lost my dog yesterday",
        "selfeval": "", "tags": "",
    },
    {
        "conv_id": "conv_2", "utterance_idx": 2, "context": "sad",
        "prompt": "My dog passed away", "speaker_idx": 1,
        "utterance": "I'm so sorry to hear that",
        "selfeval": "", "tags": "",
    },
]


def _make_mock_conversations():
    """return grouped conversations from mock rows"""
    return group_into_conversations(MOCK_ROWS)


# ---- clean_text ----

class TestCleanText:
    def test_comma_replacement(self):
        assert clean_text("hello_comma_ world") == "hello, world"

    def test_whitespace_collapsing(self):
        assert clean_text("  too   many   spaces  ") == "too many spaces"

    def test_combined(self):
        assert clean_text("hi_comma_  how   are you") == "hi, how are you"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_no_changes_needed(self):
        assert clean_text("already clean") == "already clean"


# ---- load and group ----

class TestLoadAndGroup:
    def test_groups_by_conv_id(self):
        convs = _make_mock_conversations()
        conv_ids = {c["conv_id"] for c in convs}
        assert conv_ids == {"conv_1", "conv_2"}

    def test_sorts_by_utterance_idx(self):
        # feed rows in scrambled order
        scrambled = [MOCK_ROWS[3], MOCK_ROWS[0], MOCK_ROWS[2], MOCK_ROWS[1],
                     MOCK_ROWS[5], MOCK_ROWS[4]]
        convs = group_into_conversations(scrambled)
        for conv in convs:
            idxs = [t["utterance_idx"] for t in conv["turns"]]
            assert idxs == sorted(idxs)

    def test_speaker_assignment(self):
        convs = _make_mock_conversations()
        conv1 = [c for c in convs if c["conv_id"] == "conv_1"][0]
        roles = [t["speaker"] for t in conv1["turns"]]
        assert roles == ["speaker", "listener", "speaker", "listener"]

    def test_cleans_utterance_text(self):
        convs = _make_mock_conversations()
        conv1 = [c for c in convs if c["conv_id"] == "conv_1"][0]
        assert conv1["turns"][0]["text"] == "I just got promoted, I'm thrilled!"

    def test_cleans_situation(self):
        convs = _make_mock_conversations()
        conv1 = [c for c in convs if c["conv_id"] == "conv_1"][0]
        assert conv1["situation"] == "I got a promotion"

    def test_emotion_extracted(self):
        convs = _make_mock_conversations()
        emotions = {c["emotion"] for c in convs}
        assert emotions == {"joyful", "sad"}

    @patch("src.data.load.load_dataset")
    def test_load_splits_returns_all_splits(self, mock_load):
        mock_ds = {
            "train": MOCK_ROWS,
            "validation": MOCK_ROWS[:2],
            "test": MOCK_ROWS[:2],
        }
        mock_load.return_value = mock_ds
        from src.data.load import load_splits
        splits = load_splits()
        assert set(splits.keys()) == {"train", "validation", "test"}
        assert len(splits["train"]) == 2  # 2 conversations


# ---- vad mapping ----

class TestVADMapping:
    def test_all_32_emotions_present(self):
        vad = get_vad_mappings()
        assert len(vad) == 32

    def test_known_emotions_present(self):
        vad = get_vad_mappings()
        expected = {
            "afraid", "angry", "annoyed", "anticipating", "apprehensive",
            "ashamed", "caring", "confident", "content", "devastated",
            "disappointed", "disgusted", "embarrassed", "excited", "faithful",
            "furious", "grateful", "guilty", "hopeful", "impressed",
            "jealous", "joyful", "lonely", "nostalgic", "prepared",
            "proud", "sad", "sentimental", "surprised", "terrified",
            "trusting", "wistful",
        }
        assert set(vad.keys()) == expected

    def test_vad_values_in_range(self):
        vad = get_vad_mappings()
        for emotion, (v, a, d) in vad.items():
            assert 0.0 <= v <= 1.0, f"{emotion} valence out of range: {v}"
            assert 0.0 <= a <= 1.0, f"{emotion} arousal out of range: {a}"
            assert 0.0 <= d <= 1.0, f"{emotion} dominance out of range: {d}"

    def test_vad_tuples_have_three_elements(self):
        vad = get_vad_mappings()
        for emotion, coords in vad.items():
            assert len(coords) == 3, f"{emotion} has {len(coords)} elements"


# ---- clustering ----

class TestClustering:
    def test_produces_four_clusters(self):
        result = build_emotion_clusters()
        assert result["n_clusters"] == 4

    def test_all_emotions_assigned(self):
        result = build_emotion_clusters()
        assert len(result["emotion_to_cluster"]) == 32

    def test_no_empty_clusters(self):
        result = build_emotion_clusters()
        cluster_ids = set(result["emotion_to_cluster"].values())
        assert cluster_ids == {0, 1, 2, 3}

    def test_cluster_labels_are_meaningful(self):
        result = build_emotion_clusters()
        names = set(result["cluster_names"].values())
        # each name should contain positive/negative and high/low arousal
        for name in names:
            assert "positive" in name or "negative" in name
            assert "arousal" in name

    def test_deterministic_with_seed(self):
        r1 = build_emotion_clusters(random_state=42)
        r2 = build_emotion_clusters(random_state=42)
        assert r1["emotion_to_cluster"] == r2["emotion_to_cluster"]

    def test_positive_emotions_cluster_together(self):
        result = build_emotion_clusters()
        mapping = result["emotion_to_cluster"]
        # excited and joyful should be in the same cluster
        assert mapping["excited"] == mapping["joyful"]
        # sad and lonely should be in the same cluster
        assert mapping["sad"] == mapping["lonely"]
        # excited and sad should NOT be in the same cluster
        assert mapping["excited"] != mapping["sad"]

    def test_save_and_load_roundtrip(self, tmp_path):
        result = build_emotion_clusters()
        yaml_path = str(tmp_path / "clusters.yaml")
        save_clusters(result, yaml_path)
        loaded = load_clusters(yaml_path)
        assert loaded == result["emotion_to_cluster"]


# ---- format ----

class TestFormat:
    @pytest.fixture()
    def sample_data(self):
        convs = _make_mock_conversations()
        cluster_map = {"joyful": 0, "sad": 3}
        return convs, cluster_map

    def test_has_required_fields(self, sample_data):
        convs, cluster_map = sample_data
        examples = format_for_sft(convs, cluster_map)
        required = {"text", "context", "response", "emotion", "cluster_id", "conv_id"}
        for ex in examples:
            assert required.issubset(ex.keys())

    def test_only_listener_turns_become_examples(self, sample_data):
        convs, cluster_map = sample_data
        examples = format_for_sft(convs, cluster_map)
        # conv_1 has 2 listener turns, conv_2 has 1 listener turn
        assert len(examples) == 3

    def test_text_contains_inst_tokens(self, sample_data):
        convs, cluster_map = sample_data
        examples = format_for_sft(convs, cluster_map)
        for ex in examples:
            assert "[INST]" in ex["text"]
            assert "[/INST]" in ex["text"]

    def test_context_includes_situation_and_emotion(self, sample_data):
        convs, cluster_map = sample_data
        examples = format_for_sft(convs, cluster_map)
        for ex in examples:
            assert "Situation:" in ex["context"]
            assert "Emotion:" in ex["context"]

    def test_context_grows_with_conversation(self, sample_data):
        convs, cluster_map = sample_data
        examples = format_for_sft(convs, cluster_map)
        # get examples from conv_1 (which has 2 listener turns)
        conv1_examples = [e for e in examples if e["conv_id"] == "conv_1"]
        assert len(conv1_examples) == 2
        # second example should have longer context than first
        assert len(conv1_examples[1]["context"]) > len(conv1_examples[0]["context"])

    def test_cluster_id_matches_emotion(self, sample_data):
        convs, cluster_map = sample_data
        examples = format_for_sft(convs, cluster_map)
        for ex in examples:
            assert ex["cluster_id"] == cluster_map[ex["emotion"]]

    def test_response_is_listener_text(self, sample_data):
        convs, cluster_map = sample_data
        examples = format_for_sft(convs, cluster_map)
        conv1_examples = [e for e in examples if e["conv_id"] == "conv_1"]
        assert conv1_examples[0]["response"] == "That's wonderful news!"
        assert conv1_examples[1]["response"] == "You deserve it!"

    def test_skips_unknown_emotions(self):
        convs = [{
            "conv_id": "conv_x",
            "emotion": "unknown_emotion",
            "situation": "test",
            "turns": [
                {"utterance_idx": 1, "speaker": "speaker", "text": "hi"},
                {"utterance_idx": 2, "speaker": "listener", "text": "hello"},
            ],
        }]
        examples = format_for_sft(convs, {"joyful": 0})
        assert len(examples) == 0


class TestBuildContext:
    def test_includes_prior_turns_only(self):
        conv = {
            "situation": "test situation",
            "emotion": "joyful",
            "turns": [
                {"utterance_idx": 1, "speaker": "speaker", "text": "hello"},
                {"utterance_idx": 2, "speaker": "listener", "text": "hi there"},
                {"utterance_idx": 3, "speaker": "speaker", "text": "how are you"},
            ],
        }
        ctx = build_context(conv, up_to_turn=2)
        assert "Speaker: hello" in ctx
        assert "Listener: hi there" in ctx
        assert "how are you" not in ctx


class TestApplyChatTemplate:
    def test_format(self):
        result = apply_chat_template("my context", "my response")
        assert result == "<s>[INST] my context [/INST]my response</s>"
