"""GCN graph encoder with global pooling (one embedding per frame) followed by a temporal Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, in_node_dim=2, hidden_dim=128, n_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_node_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, data_batch):
        """data_batch: PyG batch of the graphs of one time step -> (batch_size, hidden_dim)."""
        x, edge_index = data_batch.x, data_batch.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return global_mean_pool(x, data_batch.batch)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (seq_len, batch, d_model)"""
        return x + self.pe[: x.size(0)].unsqueeze(1)


class GNNTransformerPredictor(nn.Module):
    def __init__(self, gnn_hidden=128, transformer_dim=128, nhead=8, num_layers=3, m=8, L=5, C=6):
        """
        gnn_hidden: size of the graph embedding (equal to transformer_dim)
        m: observation window, L: prediction horizon, C: number of objects
        """
        super().__init__()
        assert gnn_hidden == transformer_dim
        self.gnn = GraphEncoder(in_node_dim=2, hidden_dim=gnn_hidden, n_layers=2)
        self.pos_enc = PositionalEncoding(transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=nhead, dim_feedforward=transformer_dim * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.m = m
        self.L = L
        self.C = C
        self.transformer_dim = transformer_dim

        # the last output of the Transformer is projected to L future embeddings
        self.proj_to_future = nn.Linear(transformer_dim, L * transformer_dim)

        # each future embedding is decoded into the C object positions
        self.decoder = nn.Sequential(
            nn.Linear(transformer_dim, 2 * transformer_dim),
            nn.ReLU(),
            nn.Linear(2 * transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, self.C * 2),
        )

    def forward(self, batched_inputs):
        """batched_inputs: list of m PyG batches -> predicted positions (B, L, C, 2)."""
        B = batched_inputs[0].num_graphs

        seq = torch.stack([self.gnn(batched_inputs[t]) for t in range(self.m)], dim=0)  # (m, B, D)
        seq = self.pos_enc(seq)
        out = self.transformer(seq)  # (m, B, D)

        future = self.proj_to_future(out[-1]).view(B, self.L, self.transformer_dim)  # (B, L, D)

        preds = [self.decoder(future[:, k, :]).view(B, self.C, 2) for k in range(self.L)]
        return torch.stack(preds, dim=1)  # (B, L, C, 2)
