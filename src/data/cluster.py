from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans


# nrc vad lexicon values (valence, arousal, dominance) for the 32
# empathetic dialogues emotions. scale 0-1.
VAD_LEXICON: dict[str, tuple[float, float, float]] = {
    "afraid":       (0.100, 0.840, 0.149),
    "angry":        (0.167, 0.865, 0.497),
    "annoyed":      (0.192, 0.673, 0.372),
    "anticipating": (0.672, 0.710, 0.534),
    "apprehensive": (0.225, 0.590, 0.257),
    "ashamed":      (0.105, 0.473, 0.133),
    "caring":       (0.830, 0.510, 0.590),
    "confident":    (0.900, 0.650, 0.860),
    "content":      (0.870, 0.370, 0.700),
    "devastated":   (0.061, 0.750, 0.097),
    "disappointed": (0.150, 0.475, 0.251),
    "disgusted":    (0.080, 0.690, 0.377),
    "embarrassed":  (0.145, 0.595, 0.160),
    "excited":      (0.890, 0.850, 0.700),
    "faithful":     (0.820, 0.460, 0.700),
    "furious":      (0.103, 0.910, 0.477),
    "grateful":     (0.870, 0.550, 0.630),
    "guilty":       (0.110, 0.540, 0.170),
    "hopeful":      (0.890, 0.600, 0.660),
    "impressed":    (0.850, 0.680, 0.560),
    "jealous":      (0.120, 0.696, 0.337),
    "joyful":       (0.960, 0.780, 0.740),
    "lonely":       (0.087, 0.390, 0.145),
    "nostalgic":    (0.500, 0.400, 0.400),
    "prepared":     (0.730, 0.560, 0.760),
    "proud":        (0.900, 0.680, 0.850),
    "sad":          (0.052, 0.370, 0.149),
    "sentimental":  (0.590, 0.400, 0.440),
    "surprised":    (0.653, 0.850, 0.407),
    "terrified":    (0.062, 0.900, 0.084),
    "trusting":     (0.840, 0.470, 0.700),
    "wistful":      (0.460, 0.350, 0.380),
}


def get_vad_mappings() -> dict[str, tuple[float, float, float]]:
    return dict(VAD_LEXICON)


def _label_clusters(centroids: np.ndarray) -> dict[int, str]:
    """assign unique names by ranking centroids on valence then arousal"""
    n = len(centroids)
    # split by valence: top half = positive, bottom half = negative
    valence_order = np.argsort(centroids[:, 0])
    mid = n // 2
    negative_ids = set(valence_order[:mid].tolist())
    positive_ids = set(valence_order[mid:].tolist())

    names = {}
    # within each valence group, split by arousal
    for ids, val_label in [(positive_ids, "positive"), (negative_ids, "negative")]:
        sorted_by_arousal = sorted(ids, key=lambda i: centroids[i, 1])
        for rank, cid in enumerate(sorted_by_arousal):
            ar_label = "low_arousal" if rank < len(sorted_by_arousal) / 2 else "high_arousal"
            names[cid] = f"{val_label}_{ar_label}"
    return names


def build_emotion_clusters(
    n_clusters: int = 4, random_state: int = 42
) -> dict:
    """cluster the 32 emotions by VAD coordinates using k-means"""
    emotions = list(VAD_LEXICON.keys())
    vad_array = np.array([VAD_LEXICON[e] for e in emotions])

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(vad_array)

    # rank-based naming so each cluster gets a unique label
    cluster_names = _label_clusters(km.cluster_centers_)

    # map each emotion to its cluster id
    emotion_to_cluster = {
        emotions[i]: int(km.labels_[i]) for i in range(len(emotions))
    }

    # group emotions by cluster
    clusters = {}
    for cid, name in cluster_names.items():
        member_emotions = [e for e, c in emotion_to_cluster.items() if c == cid]
        centroid = km.cluster_centers_[cid]
        clusters[name] = {
            "id": cid,
            "emotions": sorted(member_emotions),
            "centroid": {
                "valence": round(float(centroid[0]), 3),
                "arousal": round(float(centroid[1]), 3),
                "dominance": round(float(centroid[2]), 3),
            },
        }

    # vad mappings for reference
    vad_mappings = {
        e: {"valence": v[0], "arousal": v[1], "dominance": v[2]}
        for e, v in VAD_LEXICON.items()
    }

    return {
        "n_clusters": n_clusters,
        "emotion_to_cluster": emotion_to_cluster,
        "cluster_names": cluster_names,
        "clusters": clusters,
        "vad_mappings": vad_mappings,
    }


def save_clusters(cluster_data: dict, path: str) -> None:
    """write cluster assignments to yaml"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_clusters": cluster_data["n_clusters"],
        "emotion_to_cluster": cluster_data["emotion_to_cluster"],
        "cluster_names": cluster_data["cluster_names"],
        "clusters": cluster_data["clusters"],
    }
    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


def load_clusters(path: str) -> dict[str, int]:
    """load emotion-to-cluster mapping from yaml"""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["emotion_to_cluster"]
