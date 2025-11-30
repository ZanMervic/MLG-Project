import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from .ppr_utils import ppr_to_hard_negatives, approximate_ppr_rw
from .training_utils import create_edge_loader
from .training_utils import recall_at_k
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
    model, 
    message_data, 
    train_data, 
    val_data, 
    edge_type, 
    optimizer, 
    num_epochs=5, 
    device="cpu",
    batch_size=1024,
    hn_increase_rate=1,
    max_hn=None,
    ppr_start=10,
    ppr_end=100,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
):
    """
    Train PinSAGE model with hard negative sampling.
    
    Args:
        model: PinSAGEHetero model instance
        message_data: HeteroData for message passing
        train_data: HeteroData for training supervision
        val_data: HeteroData for validation
        edge_type: 3-tuple edge type, e.g. ("user", "rates", "problem")
        optimizer: torch optimizer
        num_epochs: number of training epochs
        device: device to train on
        batch_size: number of positive edges per batch
        hn_increase_rate: epochs before increasing hard negatives by 1
        max_hn: maximum number of hard negatives (None = no limit)
        ppr_start: start rank for PPR-based hard negatives
        ppr_end: end rank for PPR-based hard negatives
    """
    model = model.to(device)
    model.train()

    # Prepare data for message passing
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

    # Prepare validation edge index (message + train edges for validation)
    val_edge_index_dict = {
        ("user", "rates", "problem"): torch.unique(
            torch.cat(
                [
                    message_data[("user", "rates", "problem")].edge_index,
                    train_data[("user", "rates", "problem")].edge_index,
                ],
                dim=1,
            ).t(),
            dim=0,
        )
        .t()
        .to(device),
        ("problem", "rev_rates", "user"): torch.unique(
            torch.cat(
                [
                    message_data[("problem", "rev_rates", "user")].edge_index,
                    train_data[("problem", "rev_rates", "user")].edge_index,
                ],
                dim=1,
            ).t(),
            dim=0,
        )
        .t()
        .to(device),
    }

    # Precompute hard negative candidates using PPR
    print("Computing hard negative candidates...")
    ppr_edge_index, ppr_values = approximate_ppr_rw(
        message_data, train_data, include_holds=True
    )
    hard_negatives = ppr_to_hard_negatives(
        ppr_edge_index, ppr_values, start=ppr_start, end=ppr_end
    )
    print(f"Found hard negatives for {len(hard_negatives)} users")

    # Training loop with early stopping support
    print("Starting training...")
    best_recall = -1.0
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = getattr(train_pinsage_hetero, 'early_stopping_patience', None)
    early_stopping_min_delta = getattr(train_pinsage_hetero, 'early_stopping_min_delta', 0.0)
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_edges = 0

        # Create data loader with progressive hard negatives
        n_hard = (
            min(epoch // hn_increase_rate, max_hn)
            if max_hn is not None
            else epoch // hn_increase_rate
        )
        
        loader = create_edge_loader(
            message_data,
            train_data,
            edge_type,
            batch_size=batch_size,
            hard_negatives=hard_negatives,
            n_hard=n_hard,
        )

        # Process batches
        for batch in loader:
            pos_edge_index = batch["pos_edge_index"].to(device)  # [2, batch_size]
            neg_edge_index = batch["neg_edge_index"].to(device)  # [2, batch_size * (1 + n_hard)]

            # Calculate how many negatives per positive
            k = neg_edge_index.shape[1] // pos_edge_index.shape[1]

            optimizer.zero_grad()

            # Forward pass
            h_dict = model(x_dict, edge_index_dict)

            user_emb = h_dict["user"]
            problem_emb = h_dict["problem"]

            # Positive edge scores (dot product)
            pos_scores = (user_emb[pos_edge_index[0]] * problem_emb[pos_edge_index[1]]).sum(dim=-1)

            # Negative edge scores (dot product)
            neg_scores = (user_emb[neg_edge_index[0]] * problem_emb[neg_edge_index[1]]).sum(dim=-1)

            # Numerically stable BPR loss
            loss = F.softplus(-(pos_scores.repeat(k) - neg_scores)).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pos_edge_index.shape[1] * k
            total_edges += pos_edge_index.shape[1] * k

        avg_loss = total_loss / total_edges

        # Validation
        with torch.no_grad():
            model.eval()
            h_dict = model(x_dict, val_edge_index_dict)
            recall_result = recall_at_k(
                embed=h_dict,
                edge_index_val=val_data[edge_type].edge_index,
                edge_type=edge_type,
                k=20,
                hetero=True
            )
            # Handle both dict and float returns
            if isinstance(recall_result, dict):
                recall20 = recall_result["mean"]
            else:
                recall20 = recall_result
            
            print(f"[PinSAGE-Hetero] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Recall@20: {recall20:.4f}, Hard negatives: {n_hard}")
            
            # Track best recall
            if recall20 > best_recall + early_stopping_min_delta:
                best_recall = recall20
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            model.train()
        
        # Early stopping
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} (patience: {early_stopping_patience})")
            break
    
    return {
        "best_recall": best_recall,
        "best_epoch": best_epoch,
    }
