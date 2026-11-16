"""Object-centric spatio-temporal graphs and sliding windows for the predictive models."""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


def build_graph_from_positions(pos):
    """Fully connected graph of the C objects of one frame.

    Node features: (x, y) position; node category: object index (0 .. C-1);
    edge features (i -> j): [distance, cos(theta), sin(theta)] of the displacement p_j - p_i.
    """
    pos = torch.as_tensor(pos, dtype=torch.float32)  # (C, 2)
    C = pos.shape[0]

    x = pos.clone()
    category = torch.arange(C, dtype=torch.long)

    src, dst, edge_attrs = [], [], []
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            src.append(i)
            dst.append(j)

            diff = pos[j] - pos[i]
            dx, dy = diff[0], diff[1]
            dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
            angle = torch.atan2(dy, dx)
            edge_attrs.append(torch.stack([dist, torch.cos(angle), torch.sin(angle)], dim=0))

    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.stack(edge_attrs, dim=0)

    return Data(x=x, category=category, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


class GraphCustomDataset(Dataset):
    """Sliding windows over episodes (T, C, 2): m input graphs and the next L positions as target.

    Positions are standardised with the mean (per coordinate) and the global standard deviation of
    all the episodes given to the dataset.
    """

    def __init__(self, episodes, m, L, normalize=True):
        self.episodes = [torch.as_tensor(ep, dtype=torch.float32) for ep in episodes]
        self.m = int(m)
        self.L = int(L)

        if normalize:
            all_pos = torch.cat(self.episodes, dim=0)
            self.mean = all_pos.mean(dim=(0, 1))  # (2,)
            # a single (global) standard deviation, so that x and y keep the same scale
            self.std = all_pos.std()
            if self.std < 1e-6:
                self.std = torch.tensor(1e-6)
            self.episodes = [(ep - self.mean) / self.std for ep in self.episodes]
        else:
            self.mean = None
            self.std = None

        # graphs are built once
        self.graph_episodes = [[build_graph_from_positions(frame) for frame in ep] for ep in self.episodes]

        self.indices = []
        for ei, ep in enumerate(self.episodes):
            for s in range(0, ep.shape[0] - (self.m + self.L) + 1):
                self.indices.append((ei, s))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ei, s = self.indices[idx]
        T = self.episodes[ei].shape[0]

        # global progress of the input frames in the episode, in [0, 1] (global positional encoding)
        progress = torch.arange(s, s + self.m, dtype=torch.float32) / T

        # clone() prevents side effects of the PyG batching
        input_graphs = [g.clone() for g in self.graph_episodes[ei][s : s + self.m]]
        target_pos = self.episodes[ei][s + self.m : s + self.m + self.L]

        return {"input_graphs": input_graphs, "target_pos": target_pos, "progress": progress}


def collate_seq(batch):
    """-> input_graphs: list of m PyG batches, target_pos: (B, L, C, 2), progress: (B, m)."""
    m = len(batch[0]["input_graphs"])
    batched_inputs = [Batch.from_data_list([sample["input_graphs"][t] for sample in batch]) for t in range(m)]
    targets = torch.stack([sample["target_pos"] for sample in batch], dim=0)
    progress = torch.stack([sample["progress"] for sample in batch], dim=0)
    return {"input_graphs": batched_inputs, "target_pos": targets, "progress": progress}
