import glob
import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class AudioGraphDataset(Dataset):
    """Dataset for loading saved PyTorch Geometric audio graphs."""

    def __init__(self, graph_dir):
        self.graph_dir = graph_dir
        self.graph_files = sorted(
            glob.glob(
                os.path.join(graph_dir, "*.pt")
            )
        )

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph_path = self.graph_files[idx]

        graph = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False
        )

        return graph


class BERTEmbeddingDataset(Dataset):
    """Dataset for loading cached BERT embeddings."""

    def __init__(self, cache):
        self.embeddings = cache["embeddings"]
        self.labels = cache["labels"]
        self.track_ids = cache["track_ids"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.embeddings[idx],
            self.labels[idx],
            self.track_ids[idx]
        )


class FusionData(Data):
    """PyG Data object with a graph-level BERT embedding."""

    def __cat_dim__(
        self,
        key,
        value,
        *args,
        **kwargs
    ):
        if key == "bert_embedding":
            return None

        return super().__cat_dim__(
            key,
            value,
            *args,
            **kwargs
        )
