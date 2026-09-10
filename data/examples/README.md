# Example Audio Graphs

This directory contains 24 representative audio graph examples used for demonstrating the graph data format of the project.

## Dataset Coverage

The examples contain 3 graphs from each of the 8 FMA-small genre classes:

- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

## Graph Representation

Each `.pt` file contains a PyTorch Geometric `Data` object with:

- `x`: 128-dimensional log-Mel node features
- `edge_index`: temporal and similarity-based graph connections
- `y`: numeric genre label
- `track_id`: original FMA track identifier
- `genre`: genre name
- `title`: track title

Each node represents a complete 5-second audio segment.

The graph edges combine:

1. Bidirectional temporal adjacency between consecutive segments.
2. Bidirectional top-k cosine similarity connections between acoustically similar segments.

## Purpose

These files are representative examples for repository inspection and reproducibility. The complete processed graph dataset is not included in GitHub because of repository size constraints.

The original FMA-small audio dataset and the full processed graph dataset must be generated or obtained separately.
