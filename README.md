# GNN-BERT Music

Multimodal music genre prediction using Graph Neural Networks and DistilBERT.

## Author

**Tabarak Tasnim Tanu**  
BRAC University  
Student ID: 24341067

## Project Overview

This project investigates multimodal music genre prediction by combining audio representations, Graph Neural Networks (GNNs), and DistilBERT-based text representations.

The project uses the FMA-small dataset and evaluates unimodal and multimodal models on an official train/validation/test split.

## Implemented Tasks

1. DistilBERT title-based genre classification
2. CNN audio baseline
3. GraphSAGE audio graph classification
4. Simple GNN + DistilBERT fusion
5. Cross-attention GNN + DistilBERT fusion
6. t-SNE representation analysis
7. Qualitative case studies

## Dataset

The experiments use the FMA-small dataset.

- Train: 6,400 tracks
- Validation: 800 tracks
- Test: 800 tracks
- Number of genre classes: 8

The eight genre classes are:

- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

The original FMA audio dataset is not included in this repository.

## Audio Graph Construction

Each music track is represented as a graph where each node corresponds to one complete 5-second audio segment.

Node features are 128-dimensional log-Mel spectrogram representations.

The graph contains:

- Temporal edges between neighboring segments
- Similarity edges based on cosine similarity
- Top-2 similarity neighbors for each segment

Audio is loaded at 22,050 Hz.

Training-only feature normalization is performed using a StandardScaler fitted on the training tracks.

## Models

### DistilBERT Title Baseline

The text baseline uses distilbert-base-uncased with a frozen DistilBERT encoder, CLS-token embeddings, and a trainable classification head.

The textual input is the track title.

### CNN Baseline

A lightweight CNN is used as an audio baseline.

The CNN operates on the first complete 5-second log-Mel segment of each valid track.

### GraphSAGE

The audio graph model uses two GraphSAGE convolution layers followed by global mean pooling and a classification head.

GraphSAGE uses all complete 5-second segments available for each valid track.

### GNN + DistilBERT Fusion

Two multimodal fusion approaches are evaluated:

- Simple feature fusion
- Cross-attention fusion

The cross-attention model uses the text representation as a query over the audio graph node representations.

## Final Results

| Model | Accuracy | Macro-F1 | Micro-F1 | Macro AUC-PR |
|---|---:|---:|---:|---:|
| DistilBERT Title | 26.62% | 24.99% | 26.62% | 25.51% |
| CNN | 31.50% | 29.40% | 31.50% | 32.48% |
| GraphSAGE | 37.13% | 35.88% | 37.13% | 38.72% |
| Simple GNN+BERT | 40.38% | 40.78% | 40.38% | 42.71% |
| Cross-Attention GNN+BERT | 40.50% | 41.18% | 40.50% | 43.25% |

The cross-attention fusion model achieved the best overall performance among the evaluated models.

## Methodological Notes

The available FMA metadata provides track titles and top-level genres rather than the richer natural-language music tags described in the general project formulation.

Therefore, the text experiment should be interpreted as a title-based genre classification baseline rather than a full natural-language music tag prediction system.

The CNN baseline uses the first complete 5-second segment, while GraphSAGE uses all complete 5-second segments. Therefore, improvements from GraphSAGE cannot be attributed exclusively to graph structure.

Six training tracks were excluded from graph generation because their audio files were missing, unreadable, or shorter than one complete 5-second segment. The official split was not changed.

## Example Graphs

Representative preprocessed graph samples are available in:

data/examples/

The repository contains 24 example graph files, with three examples for each of the eight genres.

The complete graph dataset is excluded from GitHub because of its size.

## Results and Analysis

The results directory contains:

- Model comparison tables
- Ablation results
- Evaluation metrics
- Confusion matrices and related result data
- t-SNE visualizations
- Qualitative case studies

Three case studies compare text-only, graph-only, simple-fusion, and cross-attention predictions.

## Reproducibility

The main experiment notebook contains the data preparation, graph construction, model training, evaluation, and analysis workflow.

The demo notebook provides a compact end-to-end inference example.

Large raw audio data, complete graph datasets, cached embeddings, and model checkpoints are excluded from version control.

## Project Structure

GNN_BERT_Music/
├── data/
│   ├── examples/
│   ├── processed/
│   └── splits/
├── models/
│   ├── bert/
│   └── gnn/
├── notebooks/
├── report/
├── results/
├── src/
├── .gitignore
├── config.yaml
├── requirements.txt
└── README.md

## Academic Project

Multimodal Music Genre Prediction using Graph Neural Networks and DistilBERT.