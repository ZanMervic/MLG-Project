# MoonBoard Recommendation System 

End-to-end tooling to scrape MoonBoard climber activity, convert the raw data into graph-structured datasets, and train graph recommendation models.

## Repository Map

- `data/` – merged JSON dumps used as the default input to graph-building utilities (`all_users.json`, `all_problems.json`, `all_holds.json`).
- `Problems/` & `Users/` – paginated raw JSON exports straight from the scraper; kept for auditing or rebuilding the merged files.
- `utils/` – Python utilities and notebooks:
  - `moonboard_scraper.py` automates Selenium scraping of MoonBoard user pages, capturing profile metadata and logbook entries.
  - `merge_scraped_jsons.py` consolidates paginated `Problems/` and `Users/` files into the single datasets used downstream.
  - `graph_creation.py` cleans the merged JSON and converts it into a `torch_geometric.data.HeteroData` object, adding reverse relations and optional hold nodes.
  - `training_utils.py` contains shared helpers: dataset splits, edge loaders with hard-negative sampling, the `PinSAGEModel`, and training loops.
  - `testing_utils.py` evaluation helpers (e.g., recall@k).
  -  `graph_creation.ipynb` and `testing.ipynb`, capture experiments and interactive exploration.
- `screenshots/` – reference captures for the scraping UI.
- `requirements.txt` – Selenium-focused dependencies (install PyTorch/PyG separately as noted below).

## Environment Setup

1. **Python**: 3.10+ recommended.
2. **Virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. **Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Browser driver**: `moonboard_scraper.py` expects Chrome; ensure the matching ChromeDriver binary is available on `PATH`.

## Data Flow

1. **Scrape** (`utils/moonboard_scraper.py`):
   ```bash
   python utils/moonboard_scraper.py <email> <password> <first_page> <last_page>
   ```
   - Opens MoonBoard, iterates through paginated user lists, and writes `users_<start>_<page>.json` and `problems_<start>_<page>.json`.
   - Pauses system sleep on Windows to keep long scraping sessions alive.

2. **Merge** (`utils/merge_scraped_jsons.py`):
   - Normalises the paginated exports into `data/all_users.json`, `data/all_problems.json`, and `data/all_holds.json`.
   - Run whenever new pages are scraped.

3. **Graph build** (`utils/graph_creation.py`):
   ```python
   from utils.graph_creation import create_hetero_graph
   hetero = create_hetero_graph(holds_as_nodes=True)
   ```
   - Cleans user/problem dictionaries, drops empty entries, encodes grades and foot rules, and constructs:
     - `user` nodes with `[ranking, highest_grade_idx, height, weight, problems_sent]`.
     - `problem` nodes with grade, rating, num_sends, and one-hot foot rules.
     - Optional `hold` nodes (identity embedding) with `problem↔hold` relations.
   - Adds reverse edges and stores timestamps (`edge_time`) for temporal splits.

4. **Dataset splits** (`utils/training_utils.py`):
   - `train_val_test_split` partitions edges into message-passing, train, validation, and test subsets (per-user chronological by default).
   - `train_val_test_split_homogeneous` converts to homogeneous graphs for models like LightGCN.

5. **Training & evaluation**:
   ```python
   from utils.training_utils import PinSAGEModel, train_pinsage
   model = PinSAGEModel(in_channels=hetero["user"].num_node_features)
   train_pinsage(model, message_data, train_data, optimizer, device="cuda")
   ```
   - `create_edge_loader` yields batches with personalised PageRank-based hard negatives.
   - `train_pinsage` applies BPR loss; `testing_utils.recall_at_k` provides simple ranking metrics.
   - Notebooks (`testing.ipynb`, `graph_creation.ipynb`) demonstrate interactive workflows for experimentation and plotting.

## Current & Planned Models

- **PinSAGE (implemented)**: `training_utils.PinSAGEModel` uses `SAGEConv` pending direct `PinSAGEConv` integration.
- **LightGCN (baseline, upcoming)**: Will reuse homogeneous data conversion and the edge loader. Expect a simplified forward pass with layer-wise message averaging.
- **GFormer (planned)**: Targets hold-aware sequence modelling; will consume the same cleaned graph but likely require positional encodings or hold subsets.

## Workflow

1. Scrape missing user pages and merge them.
2. Run `create_hetero_graph()` to regenerate the training graph (check logs for dropped entities).
3. Persist split datasets or regenerate on the fly with `train_val_test_split`.
4. Train PinSAGE; log losses and save embeddings or state dicts.
5. Evaluate with recall@k and compare upcoming LightGCN baseline results.
