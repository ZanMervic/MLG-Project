import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv


class PinSAGEHetero(nn.Module):
    def __init__(
        self, user_in, problem_in, hidden_channels=128, out_channels=64, num_layers=2
    ):
        super().__init__()

        # Linear projections to align feature dimensions
        self.user_lin = nn.Linear(user_in, hidden_channels)
        self.problem_lin = nn.Linear(problem_in, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        ("user", "rates", "problem"): SAGEConv(
                            (hidden_channels, hidden_channels), hidden_channels
                        ),
                        ("problem", "rev_rates", "user"): SAGEConv(
                            (hidden_channels, hidden_channels), hidden_channels
                        ),
                    },
                    aggr="mean",
                )
            )

        self.lin_user_out = nn.Linear(hidden_channels, out_channels)
        self.lin_problem_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {  # align feature dimensions
            "user": self.user_lin(x_dict["user"]),
            "problem": self.problem_lin(x_dict["problem"]),
        }

        # message passing
        h_dict = x_dict
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        # output embeddings
        h_dict["user"] = self.lin_user_out(h_dict["user"])
        h_dict["problem"] = self.lin_problem_out(h_dict["problem"])
        return h_dict


def train_pinsage_hetero(
    model, message_data, train_data, edge_type, optimizer, num_epochs=5, device="cpu"
):
    model = model.to(device)
    model.train()

    x_dict = {
        "user": message_data["user"].x.to(device),
        "problem": message_data["problem"].x.to(device),
    }

    edge_index_dict = {
        ("user", "rates", "problem"): message_data[
            "user", "rates", "problem"
        ].edge_index.to(device),
        ("problem", "rev_rates", "user"): message_data[
            "problem", "rev_rates", "user"
        ].edge_index.to(device),
    }

    edge_index = train_data[edge_type].edge_index.to(device)
    num_problems = message_data["problem"].x.size(0)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        h_dict = model(x_dict, edge_index_dict)

        user_emb = h_dict["user"]
        problem_emb = h_dict["problem"]

        # BPR loss with random negatives
        users, pos_items = edge_index
        neg_items = torch.randint(0, num_problems, (users.size(0),), device=device)

        pos_scores = (user_emb[users] * problem_emb[pos_items]).sum(dim=1)
        neg_scores = (user_emb[users] * problem_emb[neg_items]).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        loss.backward()
        optimizer.step()

        print(f"[PinSAGE-Hetero] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
