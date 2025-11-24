from collections import Counter
import random
import networkx as nx
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData, Data
from .ppr_utils import ppr_to_hard_negatives, approximate_ppr_pyg, approximate_ppr_rw
from .graph_creation import standardize_columns


def hetero_to_undirected_nx(data) -> nx.Graph:
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
            src_type, _, dst_type = etype
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


def train_val_test_split(
    hetero_data,
    edge_type,
    message_p=0.7,
    train_p=0.1,
    val_p=0.1,
    by_user=True,
    standardize=True
):
    """
    Split a HeteroData dataset into message passing, training supervision, validation and testing HeteroData datasets
    over a given edge type. Input should not be standardized.

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
        idx = {s.item(): [] for s in unique_src}
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
    edge_attrs = {
        k: v[perm] for k, v in hetero_data[edge_type].items() if k != "edge_index"
    }
    rev_edge_attrs = {
        k: v[perm] for k, v in hetero_data[rev_type].items() if k != "edge_index"
    }

    # make splits
    datasets = []
    user_ratings = {}
    problem_ratings = {}
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
        
        # update user features
        rs = torch.clone(data[("user", "rates", "problem")].edge_index)
        rs[1] = data["problem"].x.t()[0][rs[1]]

        for u, r in rs.t().tolist():
            user_ratings.setdefault(u, []).append(r)
        max_grade = {k: (max(v) if max(v) > -1 else 0) for k, v in user_ratings.items()}
        sends = {k: len(v) for k, v in user_ratings.items()}
        idx = torch.tensor(list(max_grade.keys()), dtype=torch.long)
        max_grade = torch.tensor(list(max_grade.values()), dtype=torch.float)
        sends = torch.tensor(list(sends.values()), dtype=torch.float)
        data["user"].x.t()[0][idx] = max_grade
        data["user"].x.t()[3][idx] = sends

        # update problem features
        rs = torch.clone(data[("user", "rates", "problem")].edge_index)
        rs[0] = data[("user", "rates", "problem")].edge_attr.t()[1]

        for r, p in rs.t().tolist():
            problem_ratings.setdefault(p, []).append(r)
        avg_rating = {k: sum(v) / len(v) for k, v in problem_ratings.items()}
        sends = {k: len(v) for k, v in problem_ratings.items()}
        idx = torch.tensor(list(avg_rating.keys()), dtype=torch.long)
        avg_rating = torch.tensor(list(avg_rating.values()), dtype=torch.float)
        sends = torch.tensor(list(sends.values()), dtype=torch.float)
        data["problem"].x.t()[1][idx] = avg_rating
        data["problem"].x.t()[2][idx] = sends

        datasets.append(data)

    # Feature standardization
    if standardize:
        for i, data in enumerate(datasets):
            # User feature layout: highest_grade_idx, height, weight, problems_sent
            user_x = data["user"].x
            user_cont_cols = [0, 1, 2, 3] # Normalize all columns
            if i == 0:
                user_x, user_data = standardize_columns(user_x, user_cont_cols)
            else:
                mean = user_data["mean"]
                std = user_data["std"]
                user_x[:, user_cont_cols] = (user_x[:, user_cont_cols] - mean) / std
            data["user"].x = user_x

            # Problem feature layout: grade_idx, rating, num_sends, foot_rules (onehot)
            problem_x = data["problem"].x
            problem_cont_cols = [0, 1, 2] # We skip the one hot encoded features
            problem_x, _ = standardize_columns(problem_x, problem_cont_cols)
            if i == 0:
                problem_x, problem_data = standardize_columns(problem_x, problem_cont_cols)
            else:
                mean = problem_data["mean"]
                std = problem_data["std"]
                problem_x[:, problem_cont_cols] = (problem_x[:, problem_cont_cols] - mean) / std
            data["problem"].x = problem_x

    return datasets


def train_val_test_split_homogeneous(
    hetero_data,
    edge_type,
    message_p=0.7,
    train_p=0.1,
    val_p=0.1,
    by_user=True,
):
    """
    Split a HeteroData dataset into message passing, training supervision, validation and testing Data datasets
    over a given edge type.
    This is used for LightGCN.
    """
    datasets = train_val_test_split(
        hetero_data, edge_type, message_p, train_p, val_p, by_user
    )
    return [ds.to_homogeneous() for ds in datasets]


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


def create_edge_loader(
    message_data,
    supervision_data,
    edge_type=None,
    hard_negatives=None,
    batch_size=1024,
    n_hard=1,
):
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
        ppr_edge_index, ppr_values = approximate_ppr_rw(
            message_data, supervision_data, include_holds=hetero
        )
        hard_negatives = ppr_to_hard_negatives(ppr_edge_index, ppr_values)

    if hetero:
        dataset = EdgeBatchDataset(supervision_data[edge_type].edge_index)
        edge_index = torch.cat(
            [
                message_data[edge_type].edge_index,
                supervision_data[edge_type].edge_index,
            ],
            dim=1,
        ).tolist()
        existing_edges = {(u, p) for u, p in zip(edge_index[0], edge_index[1])}
        num_problems = message_data[edge_type[2]].x.shape[0]
        num_users = 0
    else:
        # include only edges that start at users
        dataset = EdgeBatchDataset(
            supervision_data.edge_index[
                :, message_data.node_type[supervision_data.edge_index[0]] == 0
            ]
        )
        edge_index = torch.cat(
            [message_data.edge_index, supervision_data.edge_index], dim=1
        ).tolist()
        existing_edges = {
            (u, p)
            for u, p in zip(edge_index[0], edge_index[1])
            if message_data.node_type[u] == 0
        }
        c = Counter(message_data.node_type.tolist())
        num_users, num_problems = c[0], c[1]

    def sample_easy_negative(pos_batch):
        """
        Generate simple negative edges by changing the target node (problem node).
        """
        neg_edges = []
        for u, p in pos_batch.t():
            new_p = num_users + random.randint(0, num_problems - 1)
            while (u.item(), new_p) in existing_edges:
                new_p = num_users + random.randint(0, num_problems - 1)
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
                new_p = num_users + random.randint(0, num_problems - 1)
                while (u_idx, new_p) in existing_edges:
                    new_p = num_users + random.randint(0, num_problems - 1)
                neg_edges.append([u_idx, new_p])

        return torch.tensor(neg_edges, dtype=torch.long).t()

    def collate_fn(batch):
        pos_batch = torch.stack(batch, dim=1)  # [2, batch_size]
        # Try to sample hard negatives, if it fails, sample easy negatives
        neg_batch = sample_easy_negative(pos_batch)
        for _ in range(n_hard):
            hard_neg = sample_hard_negative(pos_batch)
            neg_batch = torch.cat([neg_batch, hard_neg], dim=1)

        return {"pos_edge_index": pos_batch, "neg_edge_index": neg_batch}

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return loader


def recall_at_k(embed, edge_index_val, edge_type, k=20, hetero=True, num_users=None):
    """
    Compute Recall@K for user–item validation edges using precomputed embeddings.

    Args:
        embed (dict): Dictionary of node embeddings from model.forward().
        edge_index_val (torch.Tensor): [2, num_val_edges] tensor of validation edges
            (user->item relation).
        user_type (str): Node type for users.
        item_type (str): Node type for items.
        k (int): Cutoff for Recall@K.
        hetero: True if the model is heterogeneous, False if homogeneous

    Returns:
        float: Mean Recall@K across all users with at least one validation edge.
    """
    if hetero:
        user_emb = embed[edge_type[0]]  # shape [num_users, d]
        problem_emb = embed[edge_type[2]]  # shape [num_problems, d]
    else:
        user_emb = embed[:num_users]
        problem_emb = embed[num_users:]

    users, problems = edge_index_val

    if not hetero:
        problems -= num_users

    # Group validation positives by user
    val_dict = {}
    for u, i in zip(users.tolist(), problems.tolist()):
        val_dict.setdefault(u, []).append(i if hetero else i - num_users)

    recalls = []
    for u, pos_items in val_dict.items():
        # Compute scores for all items
        scores = user_emb[u] @ problem_emb.t()  # shape [num_problems]
        topk = torch.topk(scores, k=k).indices.tolist()

        # Compute recall@k for this user
        hit_count = len(set(pos_items) & set(topk))
        recall = hit_count / len(pos_items)
        recalls.append(recall)

    return sum(recalls) / len(recalls)


def train(
    model,
    message_data,
    train_data,
    val_data,
    edge_type,
    optimizer,
    hetero=True,
    features=True,
    device="cpu",
    num_epochs=10,
    batch_size=1024,
    hn_increase_rate=1,
    max_hn=None,
    ppr_start=10,
    ppr_end=100,
    early_stopping_patience=10,
    early_stopping_min_delta=0.0,
):
    """
    Train a heterogeneous GNN for link prediction using a custom edge loader.

    Args:
        model: Hetero GNN producing embeddings per node type
        message_data: HeteroData object for message passing
        train_data: HeteroData object for train supervision
        loader: DataLoader yielding batches with 'pos_edge_index' and 'neg_edge_index'
        edge_type: 3-tuple ('src_type', 'relation', 'dst_type')
        optimizer: torch optimizer
        hetero: True if the model is heterogeneous, False if homogeneous
        features: True to include features, False to not (LightGCN)
        device: 'cuda' or 'cpu'
        num_epochs: number of training epochs
        batch_size: number of positive edges per batch
        hn_increase_rate: int, how many epochs before increasing number of hard negatives by 1
        max_hn: int, the maximum number of hard negatives, after which we don't increase anymore
        ppr_start: int, start range for PPR-based hard negative sampling
        ppr_end: int, end range for PPR-based hard negative sampling
    """
    model = model.to(device)
    model.train()

    if hetero:
        x = {
            node_type: message_data[node_type].x.to(device)
            for node_type in message_data.node_types
        }
        x_val = {
            node_type: train_data[node_type].x.to(device)
            for node_type in train_data.node_types
        }
        edge_index = {
            edge_type: message_data[edge_type].edge_index.to(device)
            for edge_type in message_data.edge_types
        }
        val_edge_index = {
            edge_type: torch.unique(
                torch.cat(
                    [
                        message_data[edge_type].edge_index,
                        train_data[edge_type].edge_index,
                    ],
                    dim=1,
                ).t(),
                dim=0,
            )
            .t()
            .to(device)
            for edge_type in message_data.edge_types
        }
    else:
        x = torch.clone(message_data.x).to(device)
        x_val = torch.clone(train_data.x).to(device)
        edge_index = torch.clone(message_data.edge_index).to(device)
        val_edge_index = torch.cat(
            [message_data.edge_index, train_data.edge_index], dim=1
        ).to(device)

    # precompute hard negative candidates
    print("Computing hard negative candidates")
    ppr_edge_index, ppr_values = approximate_ppr_rw(
        message_data, train_data, include_holds=hetero
    )
    hard_negatives = ppr_to_hard_negatives(ppr_edge_index, ppr_values, start=ppr_start, end=ppr_end)

    # training loop
    best_recall = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_state_dict = None

    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        total_edges = 0

        loader = create_edge_loader(
            message_data,
            train_data,
            edge_type,
            batch_size=batch_size,
            hard_negatives=hard_negatives,
            n_hard=(
                min(epoch // hn_increase_rate, max_hn)
                if max_hn is not None
                else epoch // hn_increase_rate
            ),
        )

        for batch in loader:
            pos_edge_index = batch["pos_edge_index"].to(device)  # [2, batch_size]
            neg_edge_index = batch["neg_edge_index"].to(device)  # [2, batch_size]

            # how many times more negative than positive samples we have
            k = neg_edge_index.shape[1] // pos_edge_index.shape[1]

            optimizer.zero_grad()

            # Forward pass
            if features:
                embed = model(x, edge_index)
            else:
                embed = model(edge_index)

            if hetero:
                src_embed = embed[edge_type[0]]
                dst_embed = embed[edge_type[2]]
            else:
                src_embed = embed
                dst_embed = embed
            # Positive edge scores (dot product)
            src_pos = src_embed[pos_edge_index[0]]
            dst_pos = dst_embed[pos_edge_index[1]]
            pos_scores = (src_pos * dst_pos).sum(dim=-1)

            # Negative edge scores (dot product)
            src_neg = src_embed[neg_edge_index[0]]
            dst_neg = dst_embed[neg_edge_index[1]]
            neg_scores = (src_neg * dst_neg).sum(dim=-1)

            # BPR loss
            # loss = -torch.log(torch.sigmoid(pos_scores.repeat(k) - neg_scores)).mean()
            # Numerically stable BPR loss:
            loss = F.softplus(-(pos_scores.repeat(k) - neg_scores)).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size * k
            total_edges += batch_size * k

        avg_loss = total_loss / total_edges
        print(f"Epoch {epoch+1}, average training loss: {avg_loss:.4f}")

        with torch.no_grad():
            # forward pass
            if features:
                embed_val = model(x, val_edge_index)
            else:
                embed_val = model(val_edge_index)
            if hetero:
                pos_edge_index_val = val_data[edge_type].edge_index.to(device)
                val_recall = recall_at_k(
                    embed_val, pos_edge_index_val, edge_type, k=20
                )
            else:
                pos_edge_index_val = torch.clone(val_data.edge_index)
                pos_edge_index_val = pos_edge_index_val[
                    :, val_data.node_type[pos_edge_index_val[0]] == 0
                ]
                num_users = message_data.node_type.tolist().count(0)
                val_recall = recall_at_k(
                    embed_val,
                    pos_edge_index_val,
                    edge_type,
                    k=20,
                    hetero=False,
                    num_users=num_users,
                )

        print(f"Validation Recall@20: {val_recall:.4f}")

        # ---- early stopping check ----
        if val_recall > best_recall + early_stopping_min_delta:
            best_recall = val_recall
            best_epoch = epoch
            epochs_no_improve = 0
            # store best weights on CPU so we can reload later
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Best Recall@20={best_recall:.4f} at epoch {best_epoch+1}."
                )
                break

    # restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "best_recall": float(best_recall),
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
    }
