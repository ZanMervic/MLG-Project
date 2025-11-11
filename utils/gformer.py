import sys, os
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from models.gformer.Params import args          # your GFormer args module
from models.gformer.Model import Model, GTLayer # original GFormer implementation

def build_up_adj_from_hetero(message_data: HeteroData, edge_type, device):
    """
    Build a normalized symmetric adjacency matrix using only the user–problem edges
    (and their reverse) from a HeteroData graph.

    - Users:  0 .. num_users-1
    - Items:  num_users .. num_users+num_items-1
    """
    # user->problem edge index in type-local coords
    up_edge_index = message_data[edge_type].edge_index  # [2, E]

    num_users = message_data["user"].x.size(0)
    num_items = message_data["problem"].x.size(0)
    num_nodes = num_users + num_items

    # convert hetero indices to one index space
    u = up_edge_index[0]
    p = up_edge_index[1] + num_users  # shift items

    # undirected: add reverse edges
    row = torch.cat([u, p])
    col = torch.cat([p, u])

    row = row.to(device)
    col = col.to(device)

    # add self-loops
    all_nodes = torch.arange(num_nodes, device=device)
    row = torch.cat([all_nodes, row])
    col = torch.cat([all_nodes, col])

    values = torch.ones(row.size(0), device=device)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        values,
        size=(num_nodes, num_nodes),
    ).coalesce()

    # symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    r, c = adj.indices()
    v = adj.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]

    adj_norm = torch.sparse_coo_tensor(
        torch.stack([r, c], dim=0),
        v,
        size=(num_nodes, num_nodes),
    ).coalesce()

    return adj_norm, num_users, num_items


class GFormerWrapper(nn.Module):
    """
    Thin wrapper so GFormer fits your generic train() function.

    - Treats the graph as heterogeneous (user/problem types).
    - Ignores node features and edge_index dict that train() passes in;
      uses a precomputed bipartite adjacency internally.
    - Returns z_dict = {'user': ..., 'problem': ...}.
    """

    def __init__(self, message_data: HeteroData, edge_type, device="cuda:0"):
        super().__init__()
        self.device = device

        # build adjacency and get counts
        self.adj, num_users, num_items = build_up_adj_from_hetero(
            message_data, edge_type, device
        )
        self.num_users = num_users
        self.num_items = num_items

        # tell GFormer how many users/items to create embeddings for
        args.user = num_users
        args.item = num_items

        # optional: disable PNN layers for now (simpler)
        args.pnn_layer = 0

        # create GFormer core
        gt_layer = GTLayer().to(device)
        self.gformer = Model(gtLayer=gt_layer).to(device)

    def forward(self, *args, **kwargs):
        """
        Training loop (features=False) calls model(edge_index),
        where edge_index is a dict of hetero edge indices.
        We ignore it and just use self.adj.

        Return a dict so recall_at_k() and train() work as-is.
        """
        user_emb, item_emb, _, _ = self.gformer(
            handler=None,
            is_test=False,        # training mode; contrastive parts can run if pnn_layer>0
            sub=self.adj,
            cmp=self.adj,
            encoderAdj=self.adj,
            decoderAdj=None,
        )
        return {"user": user_emb, "problem": item_emb}

    @torch.no_grad()
    def infer(self):
        """
        Convenience method for test-time inference:
        returns (user_emb, item_emb) with is_test=True.
        """
        self.gformer.eval()
        user_emb, item_emb, _, _ = self.gformer(
            handler=None,
            is_test=True,
            sub=self.adj,
            cmp=self.adj,
            encoderAdj=self.adj,
            decoderAdj=None,
        )
        return user_emb, item_emb
