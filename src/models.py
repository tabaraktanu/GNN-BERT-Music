
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


class GraphSAGEClassifier(nn.Module):
    def __init__(
        self,
        input_dim=128,
        hidden_dim=128,
        num_classes=8,
        dropout=0.3
    ):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        logits = self.classifier(x)

        return logits


class BERTClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_classes=8,
        dropout=0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)


class SimpleFusionClassifier(nn.Module):
    def __init__(
        self,
        audio_dim=128,
        bert_dim=768,
        hidden_dim=128,
        num_classes=8,
        dropout=0.3
    ):
        super().__init__()

        self.audio_conv1 = SAGEConv(
            audio_dim,
            hidden_dim
        )

        self.audio_conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.bert_projection = nn.Linear(
            bert_dim,
            hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                128
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        x,
        edge_index,
        batch,
        bert_embedding
    ):
        x = self.audio_conv1(
            x,
            edge_index
        )
        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )

        x = self.audio_conv2(
            x,
            edge_index
        )
        x = F.relu(x)

        audio_global = global_mean_pool(
            x,
            batch
        )

        text_features = self.bert_projection(
            bert_embedding
        )

        fused_features = torch.cat(
            [
                audio_global,
                text_features
            ],
            dim=1
        )

        logits = self.classifier(
            fused_features
        )

        return logits


class GNNBERTCrossAttention(nn.Module):
    def __init__(
        self,
        audio_input_dim=128,
        gnn_hidden_dim=128,
        bert_dim=768,
        num_heads=4,
        num_classes=8,
        dropout=0.3
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            audio_input_dim,
            gnn_hidden_dim
        )

        self.conv2 = SAGEConv(
            gnn_hidden_dim,
            gnn_hidden_dim
        )

        self.bert_projection = nn.Linear(
            bert_dim,
            gnn_hidden_dim
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=gnn_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(
            gnn_hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                gnn_hidden_dim * 2,
                128
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        x,
        edge_index,
        batch,
        bert_embedding
    ):
        x = self.conv1(
            x,
            edge_index
        )
        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )
        x = F.relu(x)

        audio_dense, node_mask = to_dense_batch(
            x,
            batch
        )

        text_features = self.bert_projection(
            bert_embedding
        )

        text_query = text_features.unsqueeze(1)

        attended_text, attention_weights = self.cross_attention(
            query=text_query,
            key=audio_dense,
            value=audio_dense,
            key_padding_mask=~node_mask
        )

        attended_text = attended_text.squeeze(1)

        attended_text = self.attention_norm(
            attended_text + text_features
        )

        audio_global = global_mean_pool(
            x,
            batch
        )

        fused_features = torch.cat(
            [
                attended_text,
                audio_global
            ],
            dim=1
        )

        logits = self.classifier(
            fused_features
        )

        return logits, attention_weights
