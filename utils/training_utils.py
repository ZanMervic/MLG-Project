import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
from torch.utils.data import DataLoader, Dataset
import random
from torch_geometric.utils import to_networkx
import networkx as nx
from collections import Counter
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import get_ppr



def hetero_to_undirected_nx(data):
    """Convert a HeteroData or Data graph to undirected NetworkX graph."""
    if isinstance(data, HeteroData):
        hetero = True
    elif isinstance(data, Data):
        hetero = False 
    else:
        raise TypeError("message_data should be of type HeteroData or Data")
    G = nx.Graph()
    if hetero:
        for etype in data.edge_types:
            src_type, rel, dst_type = etype
            edge_index = data[etype].edge_index
            for src, dst in edge_index.t().tolist():
                G.add_edge((src_type, src), (dst_type, dst))
                G.add_edge((dst_type, dst), (src_type, src))  # manually add reverse
    else:
        for src, dst in data.edge_index.t().tolist():
            # types don't matter, just set as user for easier hard neg sampling
            G.add_edge(("user", src), ("user", dst))
            G.add_edge(("user", dst), ("user", src))  # manually add reverse
    return G


def train_val_test_split(hetero_data, edge_type, message_p=0.7, train_p=0.1, val_p=0.1, by_user=True):
    """
    Split a HeteroData dataset into message passing, training supervision, validation and testing HeteroData datasets 
    over a given edge type.

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

def train_val_test_split_homogeneous(hetero_data, edge_type, message_p=0.7, train_p=0.1, val_p=0.1, by_user=True):
    """
    Split a HeteroData dataset into message passing, training supervision, validation and testing Data datasets 
    over a given edge type.
    This is used for LightGCN.
    """
    datasets = train_val_test_split(hetero_data, edge_type, message_p, train_p, val_p, by_user)
    return [ds.to_homogeneous() for ds in datasets]

def approximate_ppr_pyg(message_data, supervision_data, include_holds=True, eps=1e-5, alpha=0.1):
    """
    Approximate personalized pagerank scores with pyG for all user nodes of a graph and include only 
    non-existing user-problem edges.

    Args:
        message_data: HeteroData / Data object of message edges
        supervision_data: HeteroData / Data object of supervision edges
        include_holds: boolean, whether to include hold nodes in the computation graph, only relevant with HeteroData

    Returns:
        edge_index, ppr_values; a sparse representation of PPR scores
    """
    if isinstance(message_data, HeteroData):
        hetero = True
    elif isinstance(message_data, Data):
        hetero = False 
    else:
        raise TypeError("message_data should be of type HeteroData or Data")
    
    if hetero:
        num_users = message_data["user"].x.shape[0]
        num_problems = message_data["problem"].x.shape[0]
        # transform edge indexes into a single homogeneous graph
        user_problem = message_data[("user", "rates", "problem")].edge_index
        user_problem[1] += num_users
        existing_edges = torch.cat([user_problem, user_problem.flip(0)], dim=1)
        N = num_problems + num_users
        if include_holds:
            hold_problem = message_data[("problem", "contains", "hold")].edge_index
            hold_problem[1] += num_problems + num_users 
            existing_edges = torch.cat([existing_edges, hold_problem, hold_problem.flip(0)], dim=1)
            N += message_data["hold"].x.shape[0]
        # compute ppr
        ppr_edge_index, ppr_values = get_ppr(existing_edges, target=torch.tensor(range(num_users)), alpha=alpha, eps=eps, num_nodes=N)
        # update edges to include supervision
        user_problem = supervision_data[("user", "rates", "problem")].edge_index
        user_problem[1] += num_users
        existing_edges = torch.cat([existing_edges, user_problem, user_problem.flip(0)], dim=1)
    else:
        c = Counter(message_data.node_type.tolist())
        num_users, num_problems = c[0], c[1]
        existing_edges = message_data.edge_index
        N = num_problems + num_users
        # compute ppr
        ppr_edge_index, ppr_values = get_ppr(existing_edges, target=torch.tensor(range(num_users)), alpha=alpha, eps=eps, num_nodes=N)
        # update edges to include supervision
        existing_edges = torch.cat([message_data.edge_index, supervision_data.edge_index], dim=1)

    # filter ppr scores to exclude existing edges and scores user-user, user-hold 
    existing_edges = {(x, y) for x, y in existing_edges.t().tolist()}
    new_index = []
    new_scores = []
    for (src, dst), score in zip(ppr_edge_index.t().tolist(), ppr_values.tolist()):
        if num_users <= dst < num_users + num_problems and (src, dst) not in existing_edges:
            new_index.append((src, dst))
            new_scores.append(score)

    ppr_edge_index, ppr_values = torch.tensor(new_index).t(), torch.tensor(new_scores)
    if hetero:
        ppr_edge_index[1] -= num_users 
    return ppr_edge_index, ppr_values

def ppr_to_hard_negatives(ppr_edge_index, ppr_values, start=10, end=100):
    """
    Returns a dict of user - hard negative candidate list pairs, calculated with PPR.
    """
    # get all candidates for each user
    negative_candidates = {}
    for (src, dst), score in zip(ppr_edge_index.t().tolist(), ppr_values.tolist()):
        if src in negative_candidates:
            negative_candidates[src].append((dst, score))
        else:
            negative_candidates[src] = [(dst, score)]
    # sort by ppr score
    negative_candidates = {k: 
                           [dst for dst, _ in sorted(v, key=lambda x:x[1], reverse=True)] 
                           for k, v in negative_candidates.items()}
    # subset negatives
    negative_candidates = {k: v[start:min(end, len(v))] for k, v in negative_candidates.items() if len(v) > start}
    return negative_candidates

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


def create_edge_loader(message_data, supervision_data, edge_type=None, hard_negatives=None, batch_size=1024, n_hard=1):
    """
    Returns a DataLoader that yields batches of positive edges and their corresponding negative edges.

    Args:
        message_data: HeteroData / Data object of message edges
        supervision_data: HeteroData / Data object of supervision edges
        edge_type: 3-tuple representing the type of edge to load, ex. ("user", "rates", "problem"), or None if data is homogeneous
        batch_size: number of positive edges per batch
        hard_negatives: a dict with user keys, representing hard negative candidates for each user
        n_hard: int, how many hard negatives to sample per positive

    Returns:
        DataLoader yielding dicts with:
            'pos_edge_index': [2, batch_size]
            'neg_edge_index': [2, batch_size] (for now, need to add hard negatives)
    """
    if isinstance(message_data, HeteroData):
        hetero = True
    elif isinstance(message_data, Data):
        hetero = False 
    else:
        raise TypeError("message_data should be of type HeteroData or Data")
    
    if hard_negatives is None:
        ppr_edge_index, ppr_values = approximate_ppr_pyg(message_data, supervision_data, include_holds=hetero)
        hard_negatives = ppr_to_hard_negatives(ppr_edge_index, ppr_values)

    if hetero:
        dataset = EdgeBatchDataset(supervision_data[edge_type].edge_index)
        edge_index = torch.cat([message_data[edge_type].edge_index, supervision_data[edge_type].edge_index], dim=1).tolist()
        existing_edges = {(u, p) for u, p in zip(edge_index[0], edge_index[1])}
        num_problems = message_data[edge_type[2]].x.shape[0]
        num_users = 0
    else:
        # include only edges that start at users
        dataset = EdgeBatchDataset(supervision_data.edge_index[:, message_data.node_type[supervision_data.edge_index[0]] == 0])
        edge_index = torch.cat([message_data.edge_index, supervision_data.edge_index], dim=1).tolist()
        existing_edges = {(u, p) for u, p in zip(edge_index[0], edge_index[1]) if message_data.node_type[u] == 0}
        c = Counter(message_data.node_type.tolist())
        num_users, num_problems = c[0], c[1]

    def sample_easy_negative(pos_batch):
        """
        Generate simple negative edges by changing the target node (problem node).
        """
        neg_edges = []
        for u, p in pos_batch.t():
            new_p = num_users + random.randint(0, num_problems-1)
            while (u.item(), new_p) in existing_edges:
                new_p = num_users + random.randint(0, num_problems-1)
            neg_edges.append([u.item(), new_p])
        return torch.tensor(neg_edges, dtype=torch.long).t()  # shape [2, num_neg]
    
    
    def sample_hard_negative(pos_batch):
        """
        PageRank-based hard negatives:
        - For each user u in pos_batch, compute personalized PageRank (or reuse cache).
        - Take problems ranked after the very top (skip_top) and sample from the next window.
        - If no candidate found, fall back to easy negative.
        """
        neg_edges = []

        for u, _ in pos_batch.t():
            u_idx = u.item()

            # if we have hard negative candidates
            if u_idx in hard_negatives:
                neg_edges.append([u_idx, random.choice(hard_negatives[u_idx])])
            else:
                # use easy negatives
                new_p = num_users + random.randint(0, num_problems-1)
                while (u_idx, new_p) in existing_edges:
                    new_p = num_users + random.randint(0, num_problems-1)
                neg_edges.append([u_idx, new_p])

        return torch.tensor(neg_edges, dtype=torch.long).t()


    def collate_fn(batch):
        pos_batch = torch.stack(batch, dim=1)  # [2, batch_size]
        # Try to sample hard negatives, if it fails, sample easy negatives
        neg_batch = sample_easy_negative(pos_batch)
        for _ in range(n_hard):
            hard_neg = sample_hard_negative(pos_batch)
            neg_batch = torch.cat([neg_batch, hard_neg], dim=1)

        return {
            'pos_edge_index': pos_batch,
            'neg_edge_index': neg_batch
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader

def recall_at_k(z_dict, edge_index_val, edge_type, k=20):
    """
    Compute Recall@K for user–item validation edges using precomputed embeddings.

    Args:
        z_dict (dict): Dictionary of node embeddings from model.forward().
        edge_index_val (torch.Tensor): [2, num_val_edges] tensor of validation edges 
            (user->item relation).
        user_type (str): Node type for users.
        item_type (str): Node type for items.
        k (int): Cutoff for Recall@K.

    Returns:
        float: Mean Recall@K across all users with at least one validation edge.
    """
    user_emb = z_dict[edge_type[0]]   # shape [num_users, d]
    problem_emb = z_dict[edge_type[2]]   # shape [num_problems, d]

    users, problems = edge_index_val

    # Group validation positives by user
    val_dict = {}
    for u, i in zip(users.tolist(), problems.tolist()):
        val_dict.setdefault(u, []).append(i)

    recalls = []
    for u, pos_items in val_dict.items():
        # Compute scores for all items
        scores = (user_emb[u] @ problem_emb.t())  # shape [num_problems]
        topk = torch.topk(scores, k=k).indices.tolist()

        # Compute recall@k for this user
        hit_count = len(set(pos_items) & set(topk))
        recall = hit_count / len(pos_items)
        recalls.append(recall)

    return sum(recalls) / len(recalls)


def train_hetero(model, message_data, train_data, val_data, edge_type, optimizer, device='cpu', num_epochs=10, batch_size=1024):
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
    val_edge_index_dict = {edge_type: torch.unique(torch.cat(
        [message_data[edge_type].edge_index, train_data[edge_type].edge_index],
        dim=1).t(), dim=0).t() for edge_type in message_data.edge_types}

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
        print(f"Epoch {epoch+1}, average training loss: {avg_loss:.4f}")

        with torch.no_grad():
            pos_edge_index = val_data[edge_type].edge_index
            # forward pass
            z_dict = model(x_dict, val_edge_index_dict)

            print(f"Validation Recall@20: {recall_at_k(z_dict, pos_edge_index, edge_type, k=20)}")
        
        
        
        
        
################################ PinSAGE ##########################################

class PinSAGEHetero(nn.Module):
    def __init__(self, user_in, problem_in, hidden_channels=128, out_channels=64, num_layers=2):
        super().__init__()

        # Linear projections to align feature dimensions
        self.user_lin = nn.Linear(user_in, hidden_channels)
        self.problem_lin = nn.Linear(problem_in, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv({
                    ('user', 'rates', 'problem'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                    ('problem', 'rev_rates', 'user'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                }, aggr='mean')
            )

        self.lin_user_out = nn.Linear(hidden_channels, out_channels)
        self.lin_problem_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {#align feature dimensions
            'user': self.user_lin(x_dict['user']),
            'problem': self.problem_lin(x_dict['problem'])
        }

        # message passing
        h_dict = x_dict
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        # output embeddings
        h_dict['user'] = self.lin_user_out(h_dict['user'])
        h_dict['problem'] = self.lin_problem_out(h_dict['problem'])
        return h_dict
        
        

def train_pinsage_hetero(model, message_data, train_data, edge_type, optimizer, num_epochs=5, device='cpu'):
    model = model.to(device)
    model.train()

    x_dict = {
        'user': message_data['user'].x.to(device),
        'problem': message_data['problem'].x.to(device)
    }

    edge_index_dict = {
        ('user', 'rates', 'problem'): message_data['user', 'rates', 'problem'].edge_index.to(device),
        ('problem', 'rev_rates', 'user'): message_data['problem', 'rev_rates', 'user'].edge_index.to(device)
    }

    edge_index = train_data[edge_type].edge_index.to(device)
    num_problems = message_data['problem'].x.size(0)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        h_dict = model(x_dict, edge_index_dict)

        user_emb = h_dict['user']
        problem_emb = h_dict['problem']

        # BPR loss with random negatives
        users, pos_items = edge_index
        neg_items = torch.randint(0, num_problems, (users.size(0),), device=device)

        pos_scores = (user_emb[users] * problem_emb[pos_items]).sum(dim=1)
        neg_scores = (user_emb[users] * problem_emb[neg_items]).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        loss.backward()
        optimizer.step()

        print(f"[PinSAGE-Hetero] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        


