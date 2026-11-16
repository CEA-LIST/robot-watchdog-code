"""GnnT: edge-aware GATv2 spatial encoder followed by a spatio-temporal Transformer.

Each frame of the observation window is a fully connected graph of the C objects. The GATv2
encoder produces one embedding per object and per frame; the m x C tokens are processed jointly by
a Transformer encoder and the C tokens of the last frame are decoded into the next L positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class InputNoise(nn.Module):
    """Gaussian noise on the input coordinates, during training only."""

    def __init__(self, std=0.01):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class SpatialEGAT(nn.Module):
    """Two GATv2 layers using the edge features [distance, cos(theta), sin(theta)]."""

    def __init__(self, in_dim=2, edge_dim=3, hidden_dim=128, heads=4):
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels=in_dim, out_channels=hidden_dim // heads, heads=heads, edge_dim=edge_dim, concat=True
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, edge_dim=edge_dim, concat=True
        )
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        return x  # (num_nodes, hidden_dim)


class TemporalEncoding(nn.Module):
    """Sinusoidal encoding of the position of a frame in the observation window."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x, t):
        return x + self.pe[t]


class STGraphTransformer_3(nn.Module):
    def __init__(self, hidden_dim=128, nhead=8, num_layers=4, L=5, C=6, noise_std=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.C = C
        self.L = L

        self.noise = InputNoise(noise_std)

        # embedding of the object identity
        self.label_dim = 16
        self.obj_enc = nn.Embedding(C, self.label_dim)

        # node features: position (2) + identity embedding (16)
        self.gnn = SpatialEGAT(in_dim=2 + self.label_dim, edge_dim=3, hidden_dim=hidden_dim)

        self.temp_enc = TemporalEncoding(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # regression of the future absolute positions
        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, L * 2),
        )

    def forward(self, batched_inputs):
        """batched_inputs: list of m PyG batches -> predicted positions (B, L, C, 2)."""
        m = len(batched_inputs)
        B = batched_inputs[0].num_graphs
        C = self.C

        seq_tokens = []
        for t in range(m):
            data = batched_inputs[t]
            x_kinematics = self.noise(data.x)
            x_semantic = self.obj_enc(data.category)  # (B*C, 16)
            x_combined = torch.cat([x_kinematics, x_semantic], dim=-1)  # (B*C, 18)

            node_emb = self.gnn(x_combined, data.edge_index, data.edge_attr)
            node_emb = self.temp_enc(node_emb, t)
            seq_tokens.append(node_emb.view(B, C, self.hidden_dim))

        # (B, m, C, D) -> (B, m*C, D): one token per object and per frame
        seq = torch.stack(seq_tokens, dim=1).view(B, m * C, self.hidden_dim)
        out = self.transformer(seq)

        # the C last tokens correspond to the last observed frame
        last_nodes = out[:, (m - 1) * C : m * C, :]

        fut_pos = self.future_head(last_nodes).view(B, C, self.L, 2)
        return fut_pos.permute(0, 2, 1, 3)  # (B, L, C, 2)
