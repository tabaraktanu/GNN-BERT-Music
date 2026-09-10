import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

from audio_features import extract_audio_segment_features


GENRE_TO_LABEL = {
    "Electronic": 0,
    "Experimental": 1,
    "Folk": 2,
    "Hip-Hop": 3,
    "Instrumental": 4,
    "International": 5,
    "Pop": 6,
    "Rock": 7,
}


def build_temporal_edges(num_nodes):
    """Create bidirectional edges between consecutive segments."""
    edges = []

    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    return edges


def build_similarity_edges(features, k_similarity=2):
    """Create bidirectional top-k cosine similarity edges."""
    num_nodes = len(features)

    if num_nodes <= 1:
        return []

    similarity_matrix = cosine_similarity(features)

    edges = []

    for i in range(num_nodes):
        similarity_scores = similarity_matrix[i].copy()
        similarity_scores[i] = -np.inf

        k = min(k_similarity, num_nodes - 1)

        nearest_nodes = np.argsort(similarity_scores)[-k:]

        for j in nearest_nodes:
            edges.append((i, int(j)))
            edges.append((int(j), i))

    return edges


def create_audio_graph(
    audio_path,
    track_id,
    genre,
    title,
    scaler=None,
    sr=22050,
    segment_duration=5,
    n_mels=128,
    k_similarity=2
):
    """Create a PyTorch Geometric audio graph from one track."""
    features = extract_audio_segment_features(
        audio_path,
        sr=sr,
        segment_duration=segment_duration,
        n_mels=n_mels
    )

    if features is None:
        return None

    if scaler is not None:
        features = scaler.transform(features)

    num_nodes = features.shape[0]

    temporal_edges = build_temporal_edges(num_nodes)

    similarity_edges = build_similarity_edges(
        features,
        k_similarity=k_similarity
    )

    all_edges = sorted(
        set(temporal_edges + similarity_edges)
    )

    if all_edges:
        edge_index = torch.tensor(
            all_edges,
            dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    if genre not in GENRE_TO_LABEL:
        raise ValueError(
            f"Unknown genre: {genre}"
        )

    graph = Data(
        x=torch.tensor(
            features,
            dtype=torch.float32
        ),
        edge_index=edge_index,
        y=torch.tensor(
            [GENRE_TO_LABEL[genre]],
            dtype=torch.long
        ),
    )

    graph.track_id = int(track_id)
    graph.genre = str(genre)
    graph.title = str(title)

    return graph
