import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class Custom(nn.Module):
    def __init__(self, hetero_data, hidden_channels=64, output_lin=False, num_layers=2, dropout=0.1):
        super().__init__()

        # This gives you ([node_types], [edge_type tuples...])
        # e.g., node_types = ['user', 'problem', 'hold']
        # edge_types = [('user', 'rates', 'problem'), ('problem', 'rated_by', 'user'), ...]
        node_types, edge_types = hetero_data.metadata()
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout

        # 1) Linear "input" layer per node type: feature_dim -> hidden_channels
        # This layers will project raw node features to a common hidden dimension
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            # Get feature dim for this node type
            in_channels = hetero_data[node_type].x.size(-1) 
            # Create linear layer and store in dict
            self.lin_dict[node_type] = nn.Linear(in_channels, hidden_channels)

        # 2) Several HeteroConv layers for message passing
        # HeteroConv allows us to define different conv layers per edge type
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                # edge_type is a tuple: (src_type, rel_name, dst_type)
                conv_dict[edge_type] = SAGEConv(
                    (-1, -1),  # infer input dims from x_dict at runtime
                    hidden_channels,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # 3) Final linear layers per node type (optional)
        if output_lin:
            self.lin_dict_out = nn.ModuleDict()
            for node_type in node_types:
                self.lin_dict_out[node_type] = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {"user": user_x, "problem": problem_x, "hold": hold_x}
        # edge_index_dict: {("user","rates","problem"): edge_index, ...}

        # 1) Project raw features to hidden dim
        h_dict = {}
        for node_type, x in x_dict.items():
            h = self.lin_dict[node_type](x)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_dict[node_type] = h

        # 2) Message passing
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            # Apply non-linearity + dropout after each layer
            for node_type in h_dict:
                h = F.relu(h_dict[node_type])
                h = F.dropout(h, p=self.dropout, training=self.training)
                h_dict[node_type] = h

        # 3) Final linear layer per node type (if defined)
        if hasattr(self, 'lin_dict_out'):
            for node_type in h_dict:
                h = self.lin_dict_out[node_type](h_dict[node_type])
                h_dict[node_type] = h

        # 4) Return final node embeddings per type
        return h_dict
