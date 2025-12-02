# hyperparameter_search_gformer.py

import random
import json
import time
import sys

import torch

from utils.graph_creation import create_hetero_graph
from utils.training_utils import train_val_test_split, train, recall_at_k
from utils.gformer import GFormerWrapper
from models.gformer.Params import args


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(message_data, edge_type, cfg, device: str):
    """
    Build a GFormerWrapper with hyperparameters applied through the global args.
    """
    # GFormer-specific hyperparameters (from Params.py)
    args.latdim = cfg["latdim"]
    args.head = cfg["head"]
    args.gcn_layer = cfg["gcn_layer"]
    args.gt_layer = cfg["gt_layer"]
    # This wrapper always disables PNN inside GFormer
    args.pnn_layer = 0

    # The wrapper itself will set args.user / args.item from message_data
    model = GFormerWrapper(message_data=message_data, edge_type=edge_type, device=device)
    return model


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
    # Combine GFormer (Params.py) hyperparams with RS hyperparams.
    search_space = {
        # GFormer-specific (via args.*)
        "latdim": [16, 32, 64],
        "head": [2, 4],
        "gcn_layer": [1, 2, 3],
        "gt_layer": [1, 2],
        # Optimizer & RS-specific
        "lr": [1e-3, 5e-4, 1e-4],
        "weight_decay": [1e-5, 1e-4],
        "hn_increase_rate": [3, 5],
        "max_hn": [1, 2],
        "ppr_start": [10, 20],
        "ppr_end": [100, 200],
    }

    # How many random configs to try
    N_TRIALS = 30
    RESULTS_PATH = "hyperparam_results_gformer.json"
    BEST_PARAMS_PATH = "hyperparam_best_gformer.json"
    BEST_MODEL_PATH = "hyperparam_best_model_gformer.pth"
    BEST_MODEL_EVAL_PATH = "best_model_eval_gformer.json"

    best_config = None
    best_recall = -1.0
    results = []

    print(f"Starting GFormer hyperparameter search with {N_TRIALS} trials...")
    sys.stdout.flush()

    for trial_idx, cfg in enumerate(generate_random_configs(search_space, N_TRIALS), 1):
        print("\n" + "=" * 80)
        print(f"[GFormer] Trial {trial_idx}/{N_TRIALS}")
        print("Config:", cfg)
        sys.stdout.flush()

        # --- 3) Build model & optimizer for this config ---
        model = build_model(
            message_data=message_data,
            edge_type=edge_type,
            cfg=cfg,
            device=device,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        start_time = time.time()

        # --- 4) Train with early stopping ---
        # IMPORTANT: GFormer does NOT use features -> features=False
        train_info = train(
            model=model,
            message_data=message_data,
            train_data=train_data,
            val_data=val_data,
            edge_type=edge_type,
            optimizer=optimizer,
            hetero=True,
            features=False,  # <- GFormer
            device=device,
            num_epochs=200,
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
            f"[GFormer] Trial {trial_idx} done. "
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

        # Global best across hyperparams
        if trial_best_recall > best_recall:
            best_recall = trial_best_recall
            best_config = cfg
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"*** [GFormer] New best config with Recall@20={best_recall:.4f} (saved model) ***")

        # --- write incremental results to disk after each trial ---
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
            print(f"Warning: failed to write GFormer JSON results: {e}")
            sys.stdout.flush()

        # Free GPU memory between trials
        del model
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "#" * 80)
    print("GFormer hyperparameter search finished.")
    print("Best config:")
    print(json.dumps(best_config, indent=2))
    print(f"Best validation Recall@20: {best_recall:.4f}")

    # Save final results snapshot too
    with open("hyperparam_results_final_gformer.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- 5) Final evaluation of best model on train/val/test ---
    if best_config is not None:
        print("\nEvaluating best GFormer model on train/val/test splits...")
        sys.stdout.flush()

        # Rebuild model with best hyperparameters
        best_model = build_model(
            message_data=message_data,
            edge_type=edge_type,
            cfg=best_config,
            device=device,
        )
        # Load best weights
        state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
        best_model.load_state_dict(state_dict)
        best_model.to(device)
        best_model.eval()

        # For GFormer, embeddings only depend on the bipartite adjacency,
        # not on train/val/test edges, so we can compute them once.
        with torch.no_grad():
            embed_eval = best_model()  # returns {"user": ..., "problem": ...}

        # Edges for each split
        train_edges = train_data[edge_type].edge_index.to(device)
        val_edges = val_data[edge_type].edge_index.to(device)
        test_edges = test_data[edge_type].edge_index.to(device)

        train_stats = recall_at_k(embed_eval, train_edges, edge_type, k=20, hetero=True)
        val_stats = recall_at_k(embed_eval, val_edges, edge_type, k=20, hetero=True)
        test_stats = recall_at_k(embed_eval, test_edges, edge_type, k=20, hetero=True)

        print(
            f"GFormer Train Recall@20: {train_stats['mean']:.4f} "
            f"(95% CI [{train_stats['ci_low']:.4f}, {train_stats['ci_high']:.4f}])"
        )
        print(
            f"GFormer Val   Recall@20: {val_stats['mean']:.4f} "
            f"(95% CI [{val_stats['ci_low']:.4f}, {val_stats['ci_high']:.4f}])"
        )
        print(
            f"GFormer Test  Recall@20: {test_stats['mean']:.4f} "
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
