import random
import json
import time
import sys

import torch
from torch_geometric.data import HeteroData

from utils.graph_creation import create_hetero_graph
from utils.training_utils import train_val_test_split, train, recall_at_k
from models.custom.custom_attention import CustomAttention  # adjust if your file/class name differs


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(message_data, hidden_channels, heads, num_layers, dropout):
    # Adjust this to match your HeteroRecommender __init__
    return CustomAttention(
        hetero_data=message_data,
        hidden_channels=hidden_channels,
        heads=heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def generate_random_configs(search_space, n_trials):
    keys = list(search_space.keys())
    for _ in range(n_trials):
        cfg = {k: random.choice(search_space[k]) for k in keys}
        yield cfg


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1) Build graph & splits once ---
    print("Building heterogeneous graph...")
    hetero_data = create_hetero_graph(holds_as_nodes=True)

    edge_type = ("user", "rates", "problem")

    print("Creating train/val/test splits...")
    message_data, train_data, val_data, test_data = train_val_test_split(
        hetero_data,
        edge_type=edge_type,
        message_p=0.7,
        train_p=0.1,
        val_p=0.1,
        by_user=True,
    )

    # --- 2) Define hyperparameter search space ---
    # Keep this small at first; expand once everything runs on HPC.
    search_space = {
        "hidden_channels": [16, 32, 64],
        "num_layers": [1, 2],
        "heads": [2, 4, 8],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [0.01, 1e-3, 5e-4],
        "weight_decay": [1e-5, 1e-4],
        "hn_increase_rate": [3, 5],
        "max_hn": [1, 2],
        "ppr_start": [10, 20],
        "ppr_end": [100, 200],
    }

    # How many random configs to try (you can bump this up on HPC)
    N_TRIALS = 30
    RESULTS_PATH = "hyperparam_results_attention.json"
    BEST_PARAMS_PATH = "hyperparam_best_attention.json"
    BEST_MODEL_PATH = "hyperparam_best_model_attention.pth"
    BEST_MODEL_EVAL_PATH = "best_model_eval_attention.json"

    best_config = None
    best_recall = -1.0
    results = []

    print(f"Starting hyperparameter search with {N_TRIALS} trials...")
    for trial_idx, cfg in enumerate(generate_random_configs(search_space, N_TRIALS), 1):
        print("\n" + "=" * 80)
        print(f"Trial {trial_idx}/{N_TRIALS}")
        print("Config:", cfg)
        sys.stdout.flush()

        # --- 3) Build model & optimizer for this config ---
        model = build_model(
            message_data=message_data,
            hidden_channels=cfg["hidden_channels"],
            heads=cfg["heads"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        start_time = time.time()

        # --- 4) Train with early stopping ---
        train_info = train(
            model=model,
            message_data=message_data,
            train_data=train_data,
            val_data=val_data,
            edge_type=edge_type,
            optimizer=optimizer,
            hetero=True,
            features=True,
            device=device,
            num_epochs=200,  # high upper bound; early stopping will cut it
            batch_size=1024,
            hn_increase_rate=cfg["hn_increase_rate"],
            max_hn=cfg["max_hn"],
            ppr_start=cfg["ppr_start"],
            ppr_end=cfg["ppr_end"],
            early_stopping_patience=10,
            early_stopping_min_delta=0.0,
        )

        elapsed = time.time() - start_time
        trial_best_recall = train_info["best_recall"]
        trial_best_epoch = train_info["best_epoch"]

        print(
            f"Trial {trial_idx} done. "
            f"Best val Recall@20={trial_best_recall:.4f} at epoch {trial_best_epoch}, "
            f"time={elapsed/60:.1f} min"
        )

        trial_result = {
            "config": cfg,
            "best_recall": trial_best_recall,
            "best_epoch": trial_best_epoch,
            "time_sec": elapsed,
        }
        results.append(trial_result)

        if trial_best_recall > best_recall:
            best_recall = trial_best_recall
            best_config = cfg
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"*** New best config with Recall@20={best_recall:.4f} ***")

        try:
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

            with open(BEST_PARAMS_PATH, "w") as f:
                json.dump(
                    {
                        "best_config": best_config,
                        "best_recall": best_recall,
                        "trials_completed": trial_idx,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            # Don't crash training just because writing logs failed
            print(f"Warning: failed to write JSON results: {e}")
            sys.stdout.flush()

        del model
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "#" * 80)
    print("Hyperparameter search finished.")
    print("Best config:")
    print(json.dumps(best_config, indent=2))
    print(f"Best validation Recall@20: {best_recall:.4f}")

    # save all results at the end as well
    with open("hyperparam_results_final_attention.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- 5) Final evaluation of best model on train/val/test ---
    if best_config is not None:
        print("\nEvaluating best model on train/val/test splits...")
        sys.stdout.flush()

        # Rebuild model with best hyperparameters
        best_model = build_model(
            message_data=message_data,
            hidden_channels=best_config["hidden_channels"],
            heads=best_config["heads"],
            num_layers=best_config["num_layers"],
            dropout=best_config["dropout"],
        )
        # Load best weights
        state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
        best_model.load_state_dict(state_dict)
        best_model.to(device)
        best_model.eval()

        # Use the same edge construction as in training for evaluation embeddings:
        # message + train edges for message passing (no leakage from val/test)
        if isinstance(message_data, HeteroData):
            x_eval = {nt: message_data[nt].x.to(device) for nt in message_data.node_types}
            edge_index_eval = {
                et: torch.unique(
                    torch.cat(
                        [message_data[et].edge_index, train_data[et].edge_index],
                        dim=1,
                    ).t(),
                    dim=0,
                ).t().to(device)
                for et in message_data.edge_types
            }
            hetero = True
        else:
            x_eval = message_data.x.to(device)
            edge_index_eval = torch.cat(
                [message_data.edge_index, train_data.edge_index], dim=1
            ).to(device)
            hetero = False

        with torch.no_grad():
            if hetero:
                embed_eval = best_model(x_eval, edge_index_eval)
            else:
                embed_eval = best_model(edge_index_eval)

        # Compute stats on each split
        train_edges = train_data[edge_type].edge_index.to(device)
        val_edges = val_data[edge_type].edge_index.to(device)
        test_edges = test_data[edge_type].edge_index.to(device)

        train_stats = recall_at_k(embed_eval, train_edges, edge_type, k=20, hetero=hetero)
        val_stats = recall_at_k(embed_eval, val_edges, edge_type, k=20, hetero=hetero)
        test_stats = recall_at_k(embed_eval, test_edges, edge_type, k=20, hetero=hetero)

        print(
            f"Train Recall@20: {train_stats['mean']:.4f} "
            f"(95% CI [{train_stats['ci_low']:.4f}, {train_stats['ci_high']:.4f}])"
        )
        print(
            f"Val   Recall@20: {val_stats['mean']:.4f} "
            f"(95% CI [{val_stats['ci_low']:.4f}, {val_stats['ci_high']:.4f}])"
        )
        print(
            f"Test  Recall@20: {test_stats['mean']:.4f} "
            f"(95% CI [{test_stats['ci_low']:.4f}, {test_stats['ci_high']:.4f}])"
        )

        # Store these results to disk
        best_eval = {
            "best_config": best_config,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "test_stats": test_stats,
        }
        with open(BEST_MODEL_EVAL_PATH, "w") as f:
            json.dump(best_eval, f, indent=2)

    sys.stdout.flush()

if __name__ == "__main__":
    main()
