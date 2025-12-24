import random
import json
import time
import sys

import torch
from torch_geometric.data import HeteroData

from utils.graph_creation import create_hetero_graph
from utils.training_utils import train_val_test_split, recall_at_k
from utils.pinsage import PinSAGEHetero, train_pinsage_hetero


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(message_data, user_in, problem_in, hidden_channels, out_channels, num_layers):
    """Build PinSAGE model with given hyperparameters."""
    return PinSAGEHetero(
        user_in=user_in,
        problem_in=problem_in,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
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

    # Get input dimensions
    user_in = message_data["user"].x.size(1)
    problem_in = message_data["problem"].x.size(1)

    # --- 2) Define hyperparameter search space ---
    search_space = {
        "hidden_channels": [64, 128, 256],
        "out_channels": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "lr": [0.01, 1e-3, 5e-4],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "batch_size": [512, 1024, 2048],
        "hn_increase_rate": [1, 2, 3],
        "max_hn": [1, 2, 3],
        "ppr_start": [10, 20],
        "ppr_end": [100, 200],
    }

    # How many random configs to try
    N_TRIALS = 50
    RESULTS_PATH = "hyperparam_results_pinsage.json"
    BEST_PARAMS_PATH = "hyperparam_best_pinsage.json"
    BEST_MODEL_PATH = "hyperparam_best_model_pinsage.pt"
    BEST_MODEL_EVAL_PATH = "best_model_eval_pinsage.json"

    best_config = None
    best_recall = -1.0
    best_epoch_for_best_config = None
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
            user_in=user_in,
            problem_in=problem_in,
            hidden_channels=cfg["hidden_channels"],
            out_channels=cfg["out_channels"],
            num_layers=cfg["num_layers"],
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        start_time = time.time()

        # --- 4) Train with early stopping ---
        train_info = train_pinsage_hetero(
            model=model,
            message_data=message_data,
            train_data=train_data,
            val_data=val_data,
            edge_type=edge_type,
            optimizer=optimizer,
            num_epochs=50,  # high upper bound; early stopping will cut it
            device=device,
            batch_size=cfg["batch_size"],
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
            best_epoch_for_best_config = trial_best_epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"*** New best config with Recall@20={best_recall:.4f} ***")

        # --- write incremental results to disk after each trial ---
        try:
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

            with open(BEST_PARAMS_PATH, "w") as f:
                json.dump(
                    {
                        "best_config": best_config,
                        "best_recall": best_recall,
                        "best_epoch": best_epoch_for_best_config,
                        "trials_completed": trial_idx,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            # Don't crash training just because writing logs failed
            print(f"Warning: failed to write JSON results: {e}")
            sys.stdout.flush()

        # Clean up
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
    with open("hyperparam_results_final_pinsage.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- 5) Final evaluation of best model on train/val/test ---
    if best_config is not None:
        print("\nEvaluating best model on train/val/test splits...")
        sys.stdout.flush()

        # Build a combined train+val supervision graph for the rated edge type
        rev_type = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])

        trainval_data = train_data.clone()

        # Merge edge indices and all edge attributes for edge_type
        trainval_data[edge_type].edge_index = torch.cat(
            [train_data[edge_type].edge_index, val_data[edge_type].edge_index],
            dim=1,
        )
        for k, v in train_data[edge_type].items():
            if k == "edge_index":
                continue
            trainval_data[edge_type][k] = torch.cat(
                [train_data[edge_type][k], val_data[edge_type][k]], dim=0
            )

        # Merge reverse edges as well
        trainval_data[rev_type].edge_index = torch.cat(
            [train_data[rev_type].edge_index, val_data[rev_type].edge_index],
            dim=1,
        )
        for k, v in train_data[rev_type].items():
            if k == "edge_index":
                continue
            trainval_data[rev_type][k] = torch.cat(
                [train_data[rev_type][k], val_data[rev_type][k]], dim=0
            )

        # Rebuild model with best hyperparameters
        best_model = build_model(
            message_data=message_data,
            user_in=user_in,
            problem_in=problem_in,
            hidden_channels=best_config["hidden_channels"],
            out_channels=best_config["out_channels"],
            num_layers=best_config["num_layers"],
        )

        optimizer = torch.optim.Adam(
            best_model.parameters(),
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
        )

        # Retrain on train+val for exactly best_epoch_for_best_config epochs
        num_epochs_final = (
            best_epoch_for_best_config if best_epoch_for_best_config is not None else 50
        )

        _ = train_pinsage_hetero(
            model=best_model,
            message_data=message_data,
            train_data=trainval_data,
            val_data=trainval_data,
            edge_type=edge_type,
            optimizer=optimizer,
            num_epochs=num_epochs_final,
            device=device,
            batch_size=best_config["batch_size"],
            hn_increase_rate=best_config["hn_increase_rate"],
            max_hn=best_config["max_hn"],
            ppr_start=best_config["ppr_start"],
            ppr_end=best_config["ppr_end"],
            early_stopping_patience=num_epochs_final + 1,
            early_stopping_min_delta=0.0,
        )

        best_model.to(device)
        best_model.eval()

        # Use the same edge construction as in training for evaluation embeddings:
        # message + train edges for message passing (no leakage from val/test)
        x_eval_train = {nt: message_data[nt].x.to(device) for nt in message_data.node_types}
        x_eval_val = {nt: train_data[nt].x.to(device) for nt in message_data.node_types}
        x_eval_test = {nt: val_data[nt].x.to(device) for nt in message_data.node_types}
        
        edge_index_eval_train = {
            et: message_data[et].edge_index.to(device)
            for et in message_data.edge_types
        }
        edge_index_eval_val = {
            et: torch.unique(
                torch.cat(
                    [message_data[et].edge_index, train_data[et].edge_index],
                    dim=1,
                ).t(),
                dim=0,
            ).t().to(device)
            for et in message_data.edge_types
        }
        edge_index_eval_test = {
            et: torch.unique(
                torch.cat(
                    [message_data[et].edge_index, train_data[et].edge_index, val_data[et].edge_index],
                    dim=1,
                ).t(),
                dim=0,
            ).t().to(device)
            for et in message_data.edge_types
        }

        with torch.no_grad():
            embed_eval_train = best_model(x_eval_train, edge_index_eval_train)
            embed_eval_val = best_model(x_eval_val, edge_index_eval_val)
            embed_eval_test = best_model(x_eval_test, edge_index_eval_test)

        # Compute stats on each split
        train_edges = train_data[edge_type].edge_index.to(device)
        val_edges = val_data[edge_type].edge_index.to(device)
        test_edges = test_data[edge_type].edge_index.to(device)

        train_stats = recall_at_k(embed_eval_train, train_edges, edge_type, k=20, hetero=True)
        val_stats = recall_at_k(embed_eval_val, val_edges, edge_type, k=20, hetero=True)
        test_stats = recall_at_k(embed_eval_test, test_edges, edge_type, k=20, hetero=True)

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
