"""GnnT with a global positional encoding: the progress of the frames in the episode (in [0, 1])
is embedded and added to every object token, on top of the local (window) temporal encoding.
"""

import torch
import torch.nn as nn

from robot_watchdog.anomaly_detection.gnn_transformer_3 import SpatialEGAT, TemporalEncoding


class STGraphTransformer_PE_global(nn.Module):
    def __init__(self, hidden_dim=128, nhead=8, num_layers=4, L=5, C=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.C = C
        self.L = L

        # embedding of the object identity
        self.label_dim = 16
        self.obj_enc = nn.Embedding(C, self.label_dim)

        # node features: position (2) + identity embedding (16)
        self.gnn = SpatialEGAT(in_dim=2 + self.label_dim, edge_dim=3, hidden_dim=hidden_dim)

        # local time: position of the frame in the observation window
        self.temp_enc = TemporalEncoding(hidden_dim)

        # global time: task completion percentage, embedded from a scalar
        self.global_temp_enc = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, hidden_dim))

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # regression of the future absolute positions
        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, L * 2),
        )

    def forward(self, batched_inputs, progress):
        """
        batched_inputs: list of m PyG batches
        progress: (B, m) completion percentage of each input frame
        Returns the predicted positions (B, L, C, 2).
        """
        m = len(batched_inputs)
        B = batched_inputs[0].num_graphs
        C = self.C

        seq_tokens = []
        for t in range(m):
            data = batched_inputs[t]
            x_semantic = self.obj_enc(data.category)  # (B*C, 16)
            x_combined = torch.cat([data.x, x_semantic], dim=-1)  # (B*C, 18)

            node_emb = self.gnn(x_combined, data.edge_index, data.edge_attr)  # (B*C, D)
            node_emb = self.temp_enc(node_emb, t)
            node_emb = node_emb.view(B, C, self.hidden_dim)

            # the global time embedding is added to every object of the frame
            global_emb = self.global_temp_enc(progress[:, t].view(B, 1, 1))  # (B, 1, D)
            seq_tokens.append(node_emb + global_emb)

        seq = torch.stack(seq_tokens, dim=1).view(B, m * C, self.hidden_dim)
        out = self.transformer(seq)

        # the C last tokens correspond to the last observed frame
        last_nodes = out[:, (m - 1) * C : m * C, :]

        fut_pos = self.future_head(last_nodes).view(B, C, self.L, 2)
        return fut_pos.permute(0, 2, 1, 3)  # (B, L, C, 2)
