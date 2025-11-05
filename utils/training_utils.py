import torch
from torch_geometric.data import HeteroData, Data
from torch.utils.data import DataLoader, Dataset
import random

def train_val_test_split(hetero_data, edge_type, message_p=0.7, train_p=0.1, val_p=0.1):
    """
    Split a HeteroData dataset into message passing, training supervision, validation and testing datasets.

    Args:
        hetero_data: HeteroData object to split
        edge_type: 3-tuple, the relation over which to split, ex. ("user", "rates", "problem").
                The split will also occur over the reverse relation, ex. ("problem", "rev_rates", "user")
        message_p: fraction of edges for message passing
        train_p: fraction of edges for training supervision
        val_p: fraction of edges for validation supervision

    Returns:
    tuple of four HeteroData objects:
        message_edges: edges used for message passing
        train_edges: edges used for training supervision
        val_edges: edges used for validation supervision
        test_edges: edges used for testing
    """
    # get the edge indexes of the relations and permute them 
    edge_index = hetero_data[edge_type].edge_index
    rev_type = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
    rev_edge_index = hetero_data[rev_type].edge_index
    num_edges = edge_index.size(1)

    perm = torch.randperm(num_edges)
    edge_index = edge_index[:, perm]
    rev_edge_index = rev_edge_index[:, perm]

    # permute attributes
    edge_attrs = {k: v[perm] for k, v in hetero_data[edge_type].items() if k != 'edge_index'}
    rev_edge_attrs = {k: v[perm] for k, v in hetero_data[rev_type].items() if k != 'edge_index'}

    # compute breaks between splits
    message_end = int(message_p * num_edges)
    train_end = int((message_p + train_p) * num_edges)
    val_end = int((message_p + train_p + val_p) * num_edges)

    datasets = []
    starts = [0, message_end, train_end, val_end]
    ends = [message_end, train_end, val_end, num_edges]
    for s, e in zip(starts, ends):
        data = hetero_data.clone()
        # split edge index 
        data[edge_type].edge_index = edge_index[:, s:e]
        data[rev_type].edge_index = rev_edge_index[:, s:e]
        # split edge attributes
        for k, v in edge_attrs.items():
            if k != "edge_index":
                data[edge_type][k] = v[s:e] 
        for k, v in rev_edge_attrs.items():
            if k != "edge_index":
                data[rev_type][k] = v[s:e] 
        datasets.append(data)
    
    return datasets

class EdgeBatchDataset(Dataset):
    """Dataset of positive edges for batching."""
    def __init__(self, edge_index):
        """
        Args:
            edge_index: [2, num_edges] tensor of positive edges
        """
        self.edge_index = edge_index

    def __len__(self):
        return self.edge_index.size(1)

    def __getitem__(self, idx):
        return self.edge_index[:, idx]


def create_edge_loader(message_data, supervision_data, edge_type, batch_size=1024):
    """
    Returns a DataLoader that yields batches of positive edges and their corresponding negative edges.

    Args:
        message_data: HeteroData object of message edges
        supervision_data: HeteroData object of supervision edges
        edge_type: 3-tuple representing the type of edge to load, ex. ("user", "rates", "problem")
        batch_size: number of positive edges per batch
        num_neg_per_pos: number of negative edges to generate per positive edge

    Returns:
        DataLoader yielding dicts with:
            'pos_edge_index': [2, batch_size]
            'neg_edge_index': [2, batch_size] (for now, need to add hard negatives)
    """
    dataset = EdgeBatchDataset(supervision_data[edge_type].edge_index)
    edge_index = torch.cat([message_data[edge_type].edge_index, supervision_data[edge_type].edge_index], dim=1).tolist()
    existing_edges = {(u, p) for u, p in zip(edge_index[0], edge_index[1])}
    num_problems = message_data[edge_type[2]].x.shape[0]

    def sample_easy_negative(pos_batch):
        """
        Generate simple negative edges by changing the target node (problem node).
        """
        neg_edges = []
        for u, p in pos_batch.t():
            new_p = random.randint(0, num_problems-1)
            while (u.item(), new_p) in existing_edges:
                new_p = random.randint(0, num_problems-1)
            neg_edges.append([u.item(), new_p])
        return torch.tensor(neg_edges, dtype=torch.long).t()  # shape [2, num_neg]

    def collate_fn(batch):
        pos_batch = torch.stack(batch, dim=1)  # [2, batch_size]

        neg_batch = sample_easy_negative(pos_batch)
        # TODO: add hard negative sampling

        return {
            'pos_edge_index': pos_batch,
            'neg_edge_index': neg_batch
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader
