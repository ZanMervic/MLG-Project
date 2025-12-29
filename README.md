# 🧗‍♂️ MoonBoard Recommendation System

End-to-end tooling to scrape MoonBoard climbing data, convert it into graph datasets, and train heterogeneous GNN-based recommendation models. This system recommends climbing problems to users based on their climbing history and preferences.

---

## 📁 Repository Structure

```
├── data/                         # Merged JSON datasets (not included in repo)
│   ├── all_users.json                # User profiles and climbing logs
│   ├── all_problems.json             # Problem metadata (grade, rating, holds)
│   ├── all_holds.json                # Hold positions per problem
│   └── problem_names.json            # Problem name mappings
│
├── data_utils/                   # Data scraping and preprocessing
│   ├── moonboard_scraper.py          # Selenium-based MoonBoard scraper
│   ├── merge_scraped_jsons.py        # Merges paginated JSON dumps
│   ├── screenshot_taker.py           # Captures problem screenshots
│   └── screenshot_processer.py       # Post-processes screenshots
│
├── models/                       # Model implementations
│   ├── custom/                       # Custom heterogeneous GNN models
│   │   ├── custom.py                     # SAGEConv-based heterogeneous GNN
│   │   └── custom_attention.py           # GATConv-based heterogeneous GNN with attention
│   └── gformer/                      # Graph Transformer model
│       ├── Model.py                      # GFormer architecture (GCN + GT layers)
│       └── Params.py                     # GFormer hyperparameters
│
├── utils/                        # Core utilities
│   ├── graph_creation.py             # Builds HeteroData graph from JSONs
│   ├── training_utils.py             # Splits, training loop, evaluation
│   ├── ppr_utils.py                  # PPR-based hard negative sampling
│   ├── pinsage.py                    # PinSAGE model and training
│   └── gformer.py                    # GFormer wrapper for HeteroData
│
├── hyper_tuning/                 # Hyperparameter search scripts
│   ├── hyperparameter_search.py          # Custom (SAGEConv) model search
│   ├── hyperparameter_search_attention.py# CustomAttention (GATConv) search
│   ├── hyperparameter_search_gformer.py  # GFormer search
│   ├── hyperparameter_search_pinsage.py  # PinSAGE search
│   ├── job.sh                            # SLURM job script (Custom)
│   ├── job_attention.sh                  # SLURM job script (Attention)
│   ├── job_gformer.sh                    # SLURM job script (GFormer)
│   └── job_pinsage.sh                    # SLURM job script (PinSAGE)
│
├── results/                      # Best model evaluation results
│   ├── best_model_eval.json          # Custom model results
│   ├── best_model_eval_attention.json# CustomAttention results
│   ├── best_model_eval_gformer.json  # GFormer results
│   └── best_model_eval_pinsage.json  # PinSAGE results
│
├── notebooks/                        # Notebooks for experimentation
│   ├── custom.ipynb                  # Custom model experiments
│   ├── custom_attention.ipynb        # Attention model experiments
│   ├── gformer.ipynb                 # GFormer experiments
│   ├── lightGCN.ipynb                # LightGCN experiments
│   ├── pinsage.ipynb                 # PinSAGE testing, exploration and debugging
│   └── data_graph_statistics.ipynb   # Data and graph statistics exploration
│
├── report/                           # Project report   
│   ├── images/                         # Figures used in the report
│   └── report.md                      # Project report document
├── project_proposal.pdf          # Project proposal document
├── environment.yml               # Conda environment (mlg-project)
├── README.md
└── .gitignore
```

---

## ⚙️ Environment Setup

### Via Conda

```bash
conda env create -f environment.yml
conda activate mlg-project
```

The environment includes PyTorch with CUDA 12.1 support and PyTorch Geometric.

---

## 🧮 Data Pipeline

### 1. Scrape Raw Data (Optional)

If you don't have pre-scraped data, use the Selenium-based scraper:

```bash
python data_utils/moonboard_scraper.py <email> <password> <start_page> <end_page>
```

### 2. Merge JSONs (Optional)

Combine paginated exports into unified datasets:

```bash
python data_utils/merge_scraped_jsons.py
```

Outputs:
- `data/all_users.json` – User profiles with climbing logs
- `data/all_problems.json` – Problem metadata
- `data/all_holds.json` – Hold configurations per problem

### 3. Build the Graph

```python
from utils.graph_creation import create_hetero_graph

hetero_data = create_hetero_graph(holds_as_nodes=True, standardize=False)
```

**Node Types & Features:**

| Node Type | Features |
|-----------|----------|
| `user` | `[highest_grade, height, weight, problems_sent]` |
| `problem` | `[grade, rating, num_sends, foot_rules_onehot...]` |
| `hold` *(optional)* | One-hot identity embeddings |

**Edge Types & Attributes:**

| Edge Type | Attributes |
|-----------|------------|
| `(user, rates, problem)` | `[grade, rating, attempts]` + `edge_time` (timestamp) |
| `(problem, rev_rates, user)` | Reverse edges (same attributes) |
| `(problem, contains, hold)` | `[is_start, is_middle, is_end]` (one-hot) |
| `(hold, rev_contains, problem)` | Reverse edges (same attributes) |

### 4. Split the Dataset

Temporal per-user splitting (70% message passing, 10% train, 10% val, 10% test):

```python
from utils.training_utils import train_val_test_split

message_data, train_data, val_data, test_data = train_val_test_split(
    hetero_data,
    edge_type=("user", "rates", "problem"),
    message_p=0.7,
    train_p=0.1,
    val_p=0.1,
    by_user=True,       # Split per-user by time
    standardize=True    # Standardize features after split
)
```

---

## 🧠 Models

### Custom (SAGEConv-based)

Heterogeneous GNN using `HeteroConv` with `SAGEConv` layers:

```python
from models.custom.custom import Custom

model = Custom(
    hetero_data=message_data,
    hidden_channels=128,
    num_layers=2,
    output_lin=True,
    dropout=0.05
)
```

### CustomAttention (GATConv-based)

Heterogeneous GNN with multi-head attention:

```python
from models.custom.custom_attention import CustomAttention

model = CustomAttention(
    hetero_data=message_data,
    hidden_channels=64,
    heads=4,
    num_layers=3,
    dropout=0.1
)
```

### PinSAGE

GraphSAGE-style model for bipartite user-problem graphs:

```python
from utils.pinsage import PinSAGEHetero

model = PinSAGEHetero(
    user_in=message_data["user"].x.size(1),
    problem_in=message_data["problem"].x.size(1),
    hidden_channels=128,
    out_channels=64,
    num_layers=3
)
```

### GFormer

Graph Transformer with GCN layers operating on normalized adjacency:

```python
from utils.gformer import GFormerWrapper

model = GFormerWrapper(
    message_data=message_data,
    edge_type=("user", "rates", "problem"),
    device="cuda"
)
```

---

## 🏋️ Training

### Generic Training Function

Works with Custom, CustomAttention, and GFormer models:

```python
from utils.training_utils import train
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

results = train(
    model=model,
    message_data=message_data,
    train_data=train_data,
    val_data=val_data,
    edge_type=("user", "rates", "problem"),
    optimizer=optimizer,
    device="cuda",
    num_epochs=50,
    batch_size=1024,
    hn_increase_rate=2,         # Increase hard negatives every N epochs
    max_hn=2,                   # Maximum hard negatives per positive
    ppr_start=10,               # PPR rank start for hard negatives
    ppr_end=200,                # PPR rank end for hard negatives
    early_stopping_patience=10,
    early_stopping_min_delta=0.0
)
```

### PinSAGE-specific Training

```python
from utils.pinsage import train_pinsage_hetero

results = train_pinsage_hetero(
    model=model,
    message_data=message_data,
    train_data=train_data,
    val_data=val_data,
    edge_type=("user", "rates", "problem"),
    optimizer=optimizer,
    device="cuda",
    num_epochs=50,
    batch_size=1024
)
```

---

## 🔍 Hard Negative Sampling

Hard negatives are sampled using **Personalized PageRank (PPR)** via random walks:

```python
from utils.ppr_utils import approximate_ppr_rw, ppr_to_hard_negatives

# Compute PPR scores
ppr_edge_index, ppr_values = approximate_ppr_rw(
    message_data, train_data,
    include_holds=True,
    walks_per_user=50,
    walk_length=50
)

# Convert to hard negative candidates (ranks 10-100)
hard_negatives = ppr_to_hard_negatives(ppr_edge_index, ppr_values, start=10, end=100)
```

---

## 📊 Evaluation

### Recall@K

```python
from utils.training_utils import recall_at_k

stats = recall_at_k(
    embed=embeddings,                              # Dict: {"user": ..., "problem": ...}
    edge_index_val=val_data[edge_type].edge_index,
    edge_type=("user", "rates", "problem"),
    k=20
)

print(f"Recall@20: {stats['mean']:.4f} ± {stats['std']:.4f}")
print(f"95% CI: [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]")
```

### Full Model Evaluation

```python
from utils.training_utils import evaluate_model

eval_results = evaluate_model(
    model=model,
    message_data=message_data,
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    edge_type=("user", "rates", "problem"),
    device="cuda"
)
```

---

## 🔬 Hyperparameter Search

Run hyperparameter search on HPC with SLURM:

```bash
# Custom model
sbatch hyper_tuning/job.sh

# CustomAttention model
sbatch hyper_tuning/job_attention.sh

# GFormer model
sbatch hyper_tuning/job_gformer.sh

# PinSAGE model
sbatch hyper_tuning/job_pinsage.sh
```

Each script runs 50 random trials with early stopping and saves:
- `hyperparam_results_*.json` – All trial results
- `hyperparam_best_*.json` – Best configuration
- `hyperparam_best_model_*.pt` – Best model weights
- `best_model_eval_*.json` – Final evaluation on train/val/test

---

## 📈 Results

Best **Test Recall@20** after hyperparameter search:

| Model | Test Recall@20 | Best Config |
|-------|----------------|-------------|
| **GFormer** | **0.189** | latdim=32, head=8, gcn_layer=3, gt_layer=1 |
| Custom (SAGEConv) | 0.174 | hidden=128, layers=2, output_lin=True |
| PinSAGE | 0.173 | hidden=128, out=64, layers=3 |
| CustomAttention | 0.162 | hidden=64, heads=4, layers=3 |

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/custom.ipynb` | Custom SAGEConv model experiments |
| `notebooks/custom_attention.ipynb` | Attention-based model experiments |
| `notebooks/gformer.ipynb` | GFormer training and evaluation |
| `notebooks/lightGCN.ipynb` | LightGCN baseline experiments |
| `notebooks/pinsage.ipynb` | Pinsage testing, exploration and debugging |
| `notebooks/data_graph_statistics.ipynb` | Data and graph statistics exploration |

---

## 🪜 Recommended Workflow

1. **Setup**: Create conda environment from `environment.yml`
2. **Data**: Place pre-scraped JSONs in `data/` (or run scraper)
3. **Graph**: Build heterogeneous graph with `create_hetero_graph()`
4. **Split**: Create temporal splits with `train_val_test_split()`
5. **Train**: Choose a model and train with `train()` or model-specific function
6. **Evaluate**: Use `recall_at_k()` and `evaluate_model()`
7. **Tune**: Run hyperparameter search scripts on HPC for best results

