import torch
from torch_geometric.data import HeteroData, Data
from torch.utils.data import DataLoader, Dataset
import random
from torch_geometric.utils import to_networkx
import networkx as nx

def hetero_to_undirected_nx(hetero_data):
    """Convert hetero graph to undirected NetworkX graph."""
    G = nx.Graph()
    for etype in hetero_data.edge_types:
        src_type, rel, dst_type = etype
        edge_index = hetero_data[etype].edge_index
        for src, dst in edge_index.t().tolist():
            G.add_edge((src_type, src), (dst_type, dst))
            G.add_edge((dst_type, dst), (src_type, src))  # manually add reverse
    return G


def train_val_test_split(hetero_data, edge_type, message_p=0.7, train_p=0.1, val_p=0.1, by_user=True):
    """
    Split a HeteroData dataset into message passing, training supervision, validation and testing datasets.

    Args:
        hetero_data: HeteroData object to split
        edge_type: 3-tuple, the relation over which to split, ex. ("user", "rates", "problem").
                The split will also occur over the reverse relation, ex. ("problem", "rev_rates", "user")
        message_p: fraction of edges for message passing
        train_p: fraction of edges for training supervision
        val_p: fraction of edges for validation supervision
        by_user: boolean, if True split each user by time, if False split the whole dataset by time

    Returns:
    tuple of four HeteroData objects:
        message_edges: edges used for message passing
        train_edges: edges used for training supervision
        val_edges: edges used for validation supervision
        test_edges: edges used for testing
    """
    # get the edge indexes of the relations
    edge_index = hetero_data[edge_type].edge_index
    rev_type = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
    rev_edge_index = hetero_data[rev_type].edge_index
    num_edges = edge_index.size(1)

    # sort edges by time
    perm = torch.argsort(hetero_data[edge_type].edge_time)
    edge_index_tmp = edge_index[:, perm]
    if by_user:
        # sort edges by user by time
        edges = [[], [], [], []]
        unique_src = torch.unique(edge_index_tmp[0])
        idx = {s.item():[] for s in unique_src}
        for i, s in enumerate(edge_index_tmp[0]):
            idx[s.item()].append(i)
        for s in unique_src:
            mask = idx[s.item()]
            l = len(mask)
            for i, n in enumerate(mask):
                if i < message_p * l:
                    edges[0].append(n)
                elif i < (message_p + train_p) * l:
                    edges[1].append(n)
                elif i < (message_p + train_p + val_p) * l:
                    edges[2].append(n)
                else:
                    edges[3].append(n)

        # compute breaks between splits
        message_end = len(edges[0])
        train_end = message_end + len(edges[1])
        val_end = train_end + len(edges[2])

        # modify permutation
        p2 = torch.tensor([x for sublist in edges for x in sublist])
        perm = perm[p2]
    else:
        # compute breaks between splits
        message_end = int(message_p * num_edges)
        train_end = int((message_p + train_p) * num_edges)
        val_end = int((message_p + train_p + val_p) * num_edges)
    
    edge_index = edge_index[:, perm]
    rev_edge_index = rev_edge_index[:, perm]

    # permute attributes
    edge_attrs = {k: v[perm] for k, v in hetero_data[edge_type].items() if k != 'edge_index'}
    rev_edge_attrs = {k: v[perm] for k, v in hetero_data[rev_type].items() if k != 'edge_index'}

    # make splits
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
    
    
    # Build a graph from message data
    G = hetero_to_undirected_nx(message_data)


    pr_cache = {}

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
    
    
    def sample_hard_negative(pos_batch, skip_top=50, window_size=450):
        """
        PageRank-based hard negatives:
        - For each user u in pos_batch, compute personalized PageRank (or reuse cache).
        - Take problems ranked after the very top (skip_top) and sample from the next window.
        - If no candidate found, fall back to easy negative.
        """
        neg_edges = []

        for u, _ in pos_batch.t():
            u_idx = u.item()

            # Compute or reuse personalized PageRank vector for this user
            if u_idx in pr_cache:
                pr = pr_cache[u_idx]
            else:
                # personalization key must match node names in the NetworkX graph:
                # Hetero nodes created by to_networkx are tuples like ('user', idx), ('problem', idx)
                start_node = ('user', u_idx)
                # If the node doesn't exist in G, fallback to empty dict
                if start_node not in G:
                    pr = {}
                else:
                    pr = nx.pagerank(G, alpha=0.85, personalization={start_node: 1.0})
                pr_cache[u_idx] = pr

            # collect problem nodes and sort by pagerank score desc
            if not pr:
                # fallback
                neg_edges.append([u_idx, random.randint(0, num_problems - 1)])
                continue

            problem_scores = [(node, score) for node, score in pr.items()
                              if isinstance(node, tuple) and node[0] == edge_type[2]]
            problem_scores.sort(key=lambda x: x[1], reverse=True)

            # choose candidates from a middle window to avoid the very top-most nodes
            start = skip_top
            end = min(skip_top + window_size, len(problem_scores))
            candidates = [node for node, _ in problem_scores[start:end]
                          if (u_idx, node[1]) not in existing_edges]

            if len(candidates) == 0:
                # fallback to easier negatives if no hard candidate found
                new_p = random.randint(0, num_problems - 1)
                while (u_idx, new_p) in existing_edges:
                    new_p = random.randint(0, num_problems - 1)
                neg_edges.append([u_idx, new_p])
            else:
                chosen = random.choice(candidates)
                neg_edges.append([u_idx, chosen[1]])  # chosen is ('problem', problem_idx)

        return torch.tensor(neg_edges, dtype=torch.long).t()  # [2, B]


    def collate_fn(batch):
        pos_batch = torch.stack(batch, dim=1)  # [2, batch_size]
        # Try to sample hard negatives, if it fails, sample easy negatives
        try: 
            hard_neg = sample_hard_negative(pos_batch)
            neg_batch = hard_neg
        except Exception:
            neg_batch = sample_easy_negative(pos_batch)

        return {
            'pos_edge_index': pos_batch,
            'neg_edge_index': neg_batch
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader


def train_hetero(model, message_data, train_data, edge_type, optimizer, device='cuda', num_epochs=10, batch_size=1024):
    """
    Train a heterogeneous GNN for link prediction using a custom edge loader.

    Args:
        model: Hetero GNN producing embeddings per node type
        message_data: HeteroData object for message passing
        train_data: HeteroData object for train supervision
        loader: DataLoader yielding batches with 'pos_edge_index' and 'neg_edge_index'
        edge_type: 3-tuple ('src_type', 'relation', 'dst_type')
        optimizer: torch optimizer
        device: 'cuda' or 'cpu'
        num_epochs: number of training epochs
    """
    model = model.to(device)
    model.train()

    x_dict = {node_type: message_data[node_type].x for node_type in message_data.node_types}
    edge_index_dict = {edge_type: message_data[edge_type].edge_index for edge_type in message_data.edge_types}

    for epoch in range(num_epochs):
        total_loss = 0

        loader = create_edge_loader(message_data, train_data, edge_type, batch_size=batch_size)

        for batch in loader:
            pos_edge_index = batch['pos_edge_index'].to(device)  # [2, batch_size]
            neg_edge_index = batch['neg_edge_index'].to(device)  # [2, batch_size]

            # how many times more negative than positive samples we have
            k = neg_edge_index.shape[1] // pos_edge_index.shape[1]

            optimizer.zero_grad()

            # Forward pass /// maybe need to change this depending on what kind of models we build
            z_dict = model(x_dict, edge_index_dict)

            # Positive edge scores (dot product)
            src_pos = z_dict[edge_type[0]][pos_edge_index[0]]
            dst_pos = z_dict[edge_type[2]][pos_edge_index[1]]
            pos_scores = (src_pos * dst_pos).sum(dim=-1)

            # Negative edge scores (dot product)
            src_neg = z_dict[edge_type[0]][neg_edge_index[0]]
            dst_neg = z_dict[edge_type[2]][neg_edge_index[1]]
            neg_scores = (src_neg * dst_neg).sum(dim=-1)

            # BPR loss 
            loss = -torch.log(torch.sigmoid(pos_scores.repeat(k) - neg_scores)).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size * k
            total_edges += batch_size * k

        avg_loss = total_loss / total_edges
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")