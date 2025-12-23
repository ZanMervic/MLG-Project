import torch
from torch_cluster import random_walk
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import get_ppr
from collections import Counter


def preprocess_data(
    message_data,
    supervision_data,
    include_holds=True,
):
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
        user_problem = torch.clone(
            message_data[("user", "rates", "problem")].edge_index
        )
        user_problem[1] += num_users
        message_edges = torch.cat([user_problem, user_problem.flip(0)], dim=1)
        N = num_problems + num_users
        if include_holds:
            hold_problem = torch.clone(
                message_data[("problem", "contains", "hold")].edge_index
            )
            hold_problem[1] += num_problems + num_users
            hold_problem[0] += num_users
            message_edges = torch.cat(
                [message_edges, hold_problem, hold_problem.flip(0)], dim=1
            )
            N += message_data["hold"].x.shape[0]
        # update edges to include supervision
        user_problem_sup = torch.clone(
            supervision_data[("user", "rates", "problem")].edge_index
        )
        user_problem_sup[1] += num_users
        existing_edges = torch.cat(
            [message_edges, user_problem_sup, user_problem_sup.flip(0)], dim=1
        )
    else:
        c = Counter(message_data.node_type.tolist())
        num_users, num_problems = c[0], c[1]
        message_edges = torch.clone(message_data.edge_index)
        N = num_problems + num_users
        # update edges to include supervision
        existing_edges = torch.cat(
            [message_data.edge_index, supervision_data.edge_index], dim=1
        )
    return message_edges, existing_edges, hetero, num_users, num_problems, N


def filter_edges(
    ppr_edge_index, ppr_values, existing_edges, num_users, num_problems, hetero
):
    existing_edges = {(x, y) for x, y in existing_edges.t().tolist()}
    new_index = []
    new_scores = []
    for (src, dst), score in zip(ppr_edge_index.t().tolist(), ppr_values.tolist()):
        if (
            num_users <= dst < num_users + num_problems
            and (src, dst) not in existing_edges
        ):
            new_index.append((src, dst))
            new_scores.append(score)

    ppr_edge_index, ppr_values = torch.tensor(new_index).t(), torch.tensor(new_scores)
    if hetero:
        ppr_edge_index[1] -= num_users

    return ppr_edge_index, ppr_values


def approximate_ppr_pyg(
    message_data,
    supervision_data,
    include_holds=True,
    eps=1e-5,
    alpha=0.1,
):
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
    message_edges, existing_edges, hetero, num_users, num_problems, N = preprocess_data(
        message_data, supervision_data, include_holds
    )
    # compute ppr
    ppr_edge_index, ppr_values = get_ppr(
        message_edges,
        target=torch.tensor(range(num_users)),
        alpha=alpha,
        eps=eps,
        num_nodes=N,
    )
    # filter ppr scores to exclude existing edges and scores user-user, user-hold
    ppr_edge_index, ppr_values = filter_edges(
        ppr_edge_index, ppr_values, existing_edges, num_users, num_problems, hetero
    )
    return ppr_edge_index, ppr_values


def simulate_random_walks(
    edges: torch.tensor, num_users: int, walks_per_user=10, walk_length=100
):
    """
    Simulate random walks on a graph given by edges starting at the first num_users nodes.
    """
    users = torch.tensor(range(num_users))
    batch_size = 500000 // (walks_per_user * walk_length)
    ppr_edge_index, ppr_values = [], []
    for batch in users.split(batch_size):
        start_users = batch.repeat_interleave(walks_per_user)
        rw = random_walk(edges[0], edges[1], start=start_users, walk_length=walk_length)
        # Flatten to get (source, target) pairs
        user_ids = start_users.repeat_interleave(rw.size(1))
        visited = rw.flatten()

        # get counts
        uniq, counts = torch.unique(
            torch.stack([user_ids, visited], dim=0), dim=1, return_counts=True
        )
        ppr_edge_index.append(uniq)
        ppr_values.append(counts)
    return torch.cat(ppr_edge_index, dim=1), torch.cat(ppr_values)


def approximate_ppr_rw(
    message_data,
    supervision_data,
    include_holds=True,
    walks_per_user=50,
    walk_length=50,
):
    """
    Approximate personalized pagerank scores by simulating random walks for all user nodes of a graph and include only
    non-existing user-problem edges.

    Args:
        message_data: HeteroData / Data object of message edges
        supervision_data: HeteroData / Data object of supervision edges
        include_holds: boolean, whether to include hold nodes in the computation graph, only relevant with HeteroData

    Returns:
        edge_index, ppr_values; a sparse representation of PPR scores
    """
    message_edges, existing_edges, hetero, num_users, num_problems, N = preprocess_data(
        message_data, supervision_data, include_holds
    )
    # simulate random walks
    ppr_edge_index, ppr_values = simulate_random_walks(
        message_edges, num_users, walk_length=walk_length, walks_per_user=walks_per_user
    )
    # filter ppr scores to exclude existing edges and scores user-user, user-hold
    ppr_edge_index, ppr_values = filter_edges(
        ppr_edge_index, ppr_values, existing_edges, num_users, num_problems, hetero
    )
    return ppr_edge_index, ppr_values


def ppr_to_hard_negatives(
    ppr_edge_index: torch.Tensor, ppr_values: torch.Tensor, start=10, end=100
):
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
    negative_candidates = {
        k: [dst for dst, _ in sorted(v, key=lambda x: x[1], reverse=True)]
        for k, v in negative_candidates.items()
    }
    # subset negatives
    negative_candidates = {
        k: v[start : min(end, len(v))]
        for k, v in negative_candidates.items()
        if len(v) > start
    }
    return negative_candidates
