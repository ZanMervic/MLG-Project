# 🧗‍♂️ MoonBoard Recommendation System

End-to-end tooling to scrape MoonBoard data, convert it into graph datasets, and train heterogeneous GNN-based recommendation models such as **PinSAGE** and **GFormer**.

---

## 📁 Repository Structure

```
├── data/                     # Merged JSON datasets used for graph construction (not included in the repo)
│   ├── all_users.json
│   ├── all_problems.json
│   ├── all_holds.json
│   └── problem_names.json
│
├── data_utils/               # Data scraping and preprocessing scripts
│   ├── moonboard_scraper.py      # Automates scraping of MoonBoard users and problems
│   ├── merge_scraped_jsons.py    # Merges paginated JSON dumps into single dataset files
│   ├── screenshot_taker.py       # Captures UI screenshots for scraped problems
│   └── screenshot_processer.py   # Postprocesses screenshots (e.g. hold highlighting)
│
├── models/
│   └── gformer/              # Transformer-based recommendation model
│       ├── Model.py
│       ├── Params.py
│       └── note.txt
│
├── utils/                    # Core graph and training utilities
│   ├── graph_creation.py         # Builds a HeteroData graph from merged JSONs
│   ├── ppr_utils.py              # Personalized PageRank (PPR) & hard-negative sampling
│   ├── training_utils.py         # Dataset splits, dataloaders, training, evaluation
│   ├── pinsage.py                # Implementation of PinSAGE for heterogeneous graphs
│   ├── gformer.py                # Wrapper to use GFormer on the MoonBoard graph
│   └── __init__.py
│
├── other/
│   ├── gformer.ipynb             # Experiments and training runs with GFormer
│   └── testing.ipynb             # Testing and exploratory analysis
│
├── environment.yml           # Conda environment specification
├── README.md
└── .gitignore
```

---

## ⚙️ Environment Setup (via Conda)

1. **Create and activate the environment**

   ```bash
   conda env create -f environment.yml
   conda activate moonboard
   ```

2. **Install optional GPU support**
   If not already included in your environment file:

   ```bash
   conda install pytorch pytorch-cuda=12.1 pyg -c pytorch -c nvidia -c pyg
   ```

---

## 🧮 Data Flow

Note: Steps 1 and 2 can be skipped if you have pre-scraped data available in the `data/` directory.

### 1. Scrape Raw Data

`scraper` uses Selenium to log into MoonBoard and download user/problem logs.

```bash
python data_utils/moonboard_scraper.py <email> <password> <start_page> <end_page>
```

This produces paginated JSONs of user and problem data.

---

### 2. Merge JSONs

Combine all paginated exports into unified datasets:

```bash
python data_utils/merge_scraped_jsons.py
```

This outputs:

* `data/all_users.json`
* `data/all_problems.json`
* `data/all_holds.json`

---

### 3. Build the Graph

Generate a heterogeneous PyTorch Geometric graph:

```python
from utils.graph_creation import create_hetero_graph
hetero = create_hetero_graph(holds_as_nodes=True)
```

**Nodes**

* `user`: `[ranking, highest_grade, height, weight, problems_sent]`
* `problem`: `[grade, rating, num_sends, foot_rule_onehot]`
* `hold` *(optional)*: identity embeddings

**Edges**

* `user → problem` (`rates`) with grade, rating, attempts, and timestamps
* `problem → hold` (`contains`) if hold nodes are enabled
* Reverse edges are automatically added for message passing

---

### 4. Split the Dataset

Temporal (per-user) edge splits for message passing, training, validation, and testing:

```python
from utils.training_utils import train_val_test_split
message_data, train_data, val_data, test_data = train_val_test_split(
    hetero, ("user", "rates", "problem")
)
```

---

### 5. Train a Model

#### Example **GFormer**

Transformer-based model (Graph-Former hybrid):

```python
from utils.gformer import GFormerWrapper
from utils.training_utils import train
from torch import optim

model = GFormerWrapper(message_data, ("user", "rates", "problem"), device="cuda")
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train(model, message_data, train_data, val_data, ("user", "rates", "problem"), optimizer)
```

---

## 🔍 Negative Sampling and Evaluation

Hard-negative edges are generated automatically using **Personalized PageRank (PPR)** or **random walk approximations** (`ppr_utils.py`).

Evaluation uses **Recall@K**:

```python
from utils.training_utils import recall_at_k
recall = recall_at_k(embeddings, val_data[("user", "rates", "problem")].edge_index,
                     ("user", "rates", "problem"), k=20)
print("Recall@20:", recall)
```

---

## 📓 Notebooks

* **`other/gformer.ipynb`** – GFormer training and evaluation
* **`other/testing.ipynb`** – Experimenting and PinSAGE testing

---

## 🧠 Models Overview

| Model                  | Description                                                     | File                                            |
| ---------------------- | --------------------------------------------------------------- | ----------------------------------------------- |
| **PinSAGE**            | GraphSAGE-style message passing on bipartite user-problem graph | `utils/pinsage.py`                              |
| **GFormer**            | Transformer-based model operating on normalized adjacency       | `utils/gformer.py`, `models/gformer/`           |
| **(Planned)** LightGCN | Lightweight collaborative filtering baseline                    | `utils/training_utils.py` (homogeneous support) |

---

## 🪜 Recommended Workflow

1. Scrape (or download the data if available and move to step 3.) missing user/problem data.
2. Merge JSONs into `data/all_*.json`.
3. Run `create_hetero_graph()` to regenerate the PyG dataset.
4. Split edges with `train_val_test_split()`.
5. Train `PinSAGEHetero` or `GFormerWrapper`.
6. Evaluate with `recall_at_k`.
7. Compare models and export embeddings.

