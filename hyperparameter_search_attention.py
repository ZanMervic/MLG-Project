import itertools
import random
import json
import time

import torch

from utils.graph_creation import create_hetero_graph
from utils.training_utils import train_val_test_split, train
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
        "hidden_channels": [32, 64, 128],
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
    N_TRIALS = 20

    best_config = None
    best_recall = -1.0
    results = []

    print(f"Starting hyperparameter search with {N_TRIALS} trials...")
    for trial_idx, cfg in enumerate(generate_random_configs(search_space, N_TRIALS), 1):
        print("\n" + "=" * 80)
        print(f"Trial {trial_idx}/{N_TRIALS}")
        print("Config:", cfg)

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
            print(f"*** New best config with Recall@20={best_recall:.4f} ***")

    print("\n" + "#" * 80)
    print("Hyperparameter search finished.")
    print("Best config:")
    print(json.dumps(best_config, indent=2))
    print(f"Best validation Recall@20: {best_recall:.4f}")

    # Optionally dump all results to a JSON file (for later analysis)
    with open("hyperparam_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
