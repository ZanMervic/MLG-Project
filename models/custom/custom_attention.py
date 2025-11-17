import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GAT, GATConv


class CustomAttention(nn.Module):
    def __init__(self, hetero_data, hidden_channels=64, heads=4, num_layers=3, dropout=0.2):
        super().__init__()

        # This gives you ([node_types], [edge_type tuples...])
        # e.g., node_types = ['user', 'problem', 'hold']
        # edge_types = [('user', 'rates', 'problem'), ('problem', 'rated_by', 'user'), ...]
        node_types, edge_types = hetero_data.metadata()
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout
        self.heads = heads
        
        # preprocessing layer
        self.pre_pm = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hetero_data[node_type].x.size(-1) , hidden_channels),
                nn.ReLU()
            )
            for node_type in node_types
        })

        # HeteroConv layers for message passing
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                # edge_type is a tuple: (src_type, rel_name, dst_type)
                if i == 0:
                    conv_dict[edge_type] = GATConv(
                        hidden_channels, 
                        hidden_channels,
                        heads=self.heads,
                        add_self_loops=False
                    )
                else:
                    conv_dict[edge_type] = GATConv(
                        self.heads * hidden_channels, 
                        hidden_channels,
                        heads=self.heads,
                        add_self_loops=False
                    )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # post-processing linear layers 
        self.post_pm = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(self.heads * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            for node_type in node_types
        })

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {"user": user_x, "problem": problem_x, "hold": hold_x}
        # edge_index_dict: {("user","rates","problem"): edge_index, ...}

        # Pre processing
        h_dict = {}
        for nt in x_dict:
            h_dict[nt] = self.pre_pm[nt](x_dict[nt])

        # Message passing
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            # Apply non-linearity + dropout after each layer
            for node_type in h_dict:
                h = F.relu(h_dict[node_type])
                h = F.dropout(h, p=self.dropout, training=self.training)
                h_dict[node_type] = h

        # Post processing
        out_dict = {}
        for nt in x_dict:
            out_dict[nt] = self.post_pm[nt](h_dict[nt])
        return out_dict
