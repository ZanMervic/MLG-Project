# Climbing the Graph: A GNN-Powered MoonBoard Recommendation System

# Introduction (Vid)

# Dataset (Žan)

(Uvod lahko še spremenim, glede na to kaj bo v introductionu)\
The MoonBoard is a globally standardised training board for climbers. Each board has 18 rows and 11 columns of climbing holds, and a climber solves a problem by moving from the start holds (circled in green) to a goal hold (circled in red). Climbers upload new problems and record their ascents via the official mobile app, which provides a global database of problems. Our goal was to build a recommender system that learns from this data, so the first step was to collect and organise it.

### Scraping the MoonBoard database

We built a Selenium‑based scraper that crawls the MoonBoard website and logbooks of registered users. Focusing on the popular MoonBoard Masters 2019 hold setup, the scraper iterates through pages of users and, for each user, opens their climbing logbook. For every ascent in the logbook it records:

- User attributes such as the climber’s highest grade, height, weight and the number of problems they have sent.
- Problem metadata including the grade, average star rating, total number of ascents (`num_sends`), setter and any recorded foot rules (whether both feet can be used or only one).
- Interaction details – the grade the user assigned to the problem, their rating, number of attempts and an optional comment, along with the date of the ascent.

### Processing problem images

A big part of what makes a user like or dislike a climbing problem are the holds used and their arrangement, so we wanted to capture this information as well.
To reconstruct the holds used in each problem we captured images of every problem using a script runs the MoonBoard app on a device (for example, a phone or emulator) and captures a screenshot of each problem. We then used simple computer‑vision rules to detect the green, blue and red circles that mark the start, middle and end holds. Once located, the positions are converted into hold identifiers and saved in JSON files. These hold lists will later be used when constructing the heterogeneous graph.

### Dataset size and attributes

(TODO: vpisat prave številke)

At the time of writing we had collected [number_of_users] users, [number_of_problems] problems and [number_of_interactions] logged ascents. Each JSON entry contains all information needed to build our graphs. Below is a simplified example showing a user and a problem entry (formatted as JSON):

```json
// example user entry
{
    "bio": "MoonBoard enthusiast",
    "ranking": 42,
    "highest_grade": "7C",
    "height": 175,
    "weight": 65,
    "problems_sent": 1272,
    "problems": {
        "PIZZA DECHU": {
            "grade": "7A",
            "rating": 2.0,
            "date": "2025-09-30",
            "attempts": 1,
            "comment": "tricky but fun"
        },
        ...
    }
}

// example problem entry
{
    "grade": "7A",
    "rating": 4.0,
    "num_sends": 341,
    "setter": "Climber123",
    "holds": "Any marked holds"
    "holds": {
        "start": ["F4","J3"],
        "end": ["G18","J18"],
        "middle": ["C13","C7","D11",...]
    }
}
```

In summary, our dataset contains user metadata, problem metadata and a log of every ascent. We also store processed hold positions for each problem. These files serve as the starting point for constructing the various graphs used in the recommendation models.

# Graph (Žan)

Once we had the raw data, the next step was to convert it into graphs that our models could learn from. Recommender systems typically represent users and items as nodes in a bipartite graph, with edges representing interactions. Our domain, however, also has a natural third entity: holds. To accommodate different models we built both bipartite and heterogeneous graphs.

### Node and edge types

We used PyTorch Geometric’s `HeteroData` container to assemble a single heterogeneous graph. The graph contains up to three node types:

| Node type | Features | Description |
|---|---|---|
| user | `[highest_grade_idx, height, weight, problems_sent]` | Encodes climber ability and demographics. |
| problem | `[grade, rating, num_sends, foot_rules]` | Captures problem difficulty, popularity and foot rules. |
| hold (optional) | One‑hot identity vector | Uniquely identifies each board hold; used only in the heterogeneous version. |

Edges connect these nodes in two ways. Each ascent produces a `(user, rates, problem)` edge with attributes `[grade, rating, attempts]` and a timestamp; a reverse edge `(problem, rev_rates, user)` is added to make message passing symmetric. When using hold nodes we also add `(problem, contains, hold)` edges with one‑hot flags indicating whether the hold is a start, middle or end hold, and the corresponding reverse edges `(hold, rev_contains, problem)`.

### Graph variations

Our models have different requirements. We therefore constructed two flavours of graphs:

- **Bipartite graph:** This contains only user and problem nodes with edges representing each logged ascent. The node feature matrices for users and problems are retained, but there are no hold nodes. This simpler structure is compatible with PinSAGE, which performs neighbor sampling on bipartite graphs, and with GFormer, which uses a normalised adjacency matrix rather than a PyG heterograph. For GFormer we convert the bipartite graph into a dense adjacency matrix and drop all node and edge features.

- **Heterogeneous graph:** This version adds hold nodes connected to their respective problems. We keep all node features and edge attributes. This richer representation is used by our Custom and CustomAttention models, which operate on PyG HeteroData objects and can learn from multiple node and edge types.

Some pre‑built models could not take our PyG heterographs directly. For example, GFormer does not accept a heterogeneous graph or node features; it instead requires an adjacency matrix. As a result we extracted the `(user, rates, problem)` edge list from the bipartite graph and built a normalised adjacency matrix for GFormer. By contrast, PinSAGE expects a bipartite graph and also cannot use features (Google pravi, da lahko uporablja featurje, spomin mi pravi da ne, preveri!). Only our custom models support the full heterogeneous graph with holds and rich features.

### Temporal splitting and graph statistics

Recommender systems are evaluated on their ability to predict future interactions. To mimic this we perform a temporal per‑user split: for each user we sort their interactions by time and allocate the earliest 70 % to a message‑passing set, then 10 % each to train, validation and test sets. The message‑passing edges are used to build node representations, while the train/val/test edges are withheld for loss optimisation and evaluation. This split ensures that the model never observes an edge at prediction time that occurred after an edge in its training set.

When reporting the graph properties we provide the following statistics for the final dataset (TODO: vpisat prave številke):

- Number of user nodes: `[num_user_nodes]`
- Number of problem nodes: `[num_problem_nodes]`
- Number of hold nodes: `[num_hold_nodes]` (in the heterogeneous graph)
- Number of user–problem edges: `[num_user_problem_edges]`
- Number of problem–hold edges: `[num_problem_hold_edges]`
- Graph diameter and density: `[graph_diameter]`, `[graph_density]`

The figure below illustrates the bipartite version of our graph: users connect to problems, with edges labelled by interactions (TODO, bolje opiši, ko bo slikca).

*(TODO: Slika/skica grafa)*

The figure below illustrates the heterogenous version of our graph: users again connect to problems, however here, problems are also connected to holds (TODO, bolje opiši, ko bo slikca).

*(TODO: Slika/skica grafa)*

---

# Approach (Tadeju)

# Models

## Pinsage (Vid)

## GFormer (Žan)

(Napisal ChatGPT, ker ne vem točno kako deluje GFormer, za preverit in popravit)

Graph neural networks such as GraphSAGE and GAT aggregate information locally; to capture long‑range dependencies we experimented with Graph Transformers. The Heterogeneous Graph Transformer (HGT) proposed by Hu et al. introduces node‑ and edge‑type‑dependent parameters to learn attention over heterogeneous graphs. It also includes relative temporal encoding, allowing the model to capture time dynamics. Inspired by HGT, the GFormer model in our project combines a few GCN layers with a multi‑head self‑attention layer to learn latent representations for users and problems.

### How GFormer works in our pipeline

GFormer treats the user–problem interaction graph as a fully connected bipartite graph. It first applies Graph Convolutional Network (GCN) layers to propagate information along the normalized adjacency matrix of the bipartite graph, producing intermediate embeddings. Then a Transformer layer aggregates these embeddings across all nodes using attention heads. Unlike our custom heterogeneous GNNs, GFormer ignores node features and instead relies solely on the pattern of interactions to learn high‑quality embeddings. This makes it particularly suited to sparse recommendation settings where interactions convey more information than demographic features.

In practice, integrating GFormer into our pipeline required several adaptations:

- Graph representation – because GFormer only supports two node types, we used the bipartite graph without hold nodes. We constructed a normalized adjacency matrix from the `(user, rates, problem)` edge list and passed it to the model wrapper.
- Batching and memory – Transformer layers attend over all nodes, which can be memory intensive. We tuned the latent dimension, number of attention heads and number of GCN and Transformer layers to balance performance and hardware constraints.
- Hybrid training – during training we still use the generic BPR loss and hard negative sampling described below. GFormer’s embeddings plug into our existing training loop without modification.

While GFormer does not use node features, its self‑attention mechanism allows information to flow across the entire bipartite graph. In our experiments we tuned its hyper‑parameters (latent dimension, number of attention heads, GCN layers and Transformer layers) to find a configuration that works well on our data. The model runs efficiently on GPUs thanks to the sparse adjacency representation, but training still requires careful batching to avoid memory issues.

## Custom (Tadeju)

## Custom Attention (Tadeju)

# Evaluation (Žan)

Evaluating a recommender system involves both hyper‑parameter tuning and robust metrics. Our evaluation pipeline operates as follows:

1. Temporal splitting. We perform the 70/10/10 split described above (predvidevam da boš Tadej to opisal v Approach?), ensuring that training only uses information available before the interaction being predicted.
2. Hyper‑parameter search. For each model we run a random search over learning rate, embedding dimension, number of layers, dropout, and model‑specific parameters such as the number of attention heads. Each trial trains the model on the training set using the BPR (Bayesian Personalized Ranking) loss with hard negative sampling; we stop early if the validation recall does not improve for a fixed number of epochs. Hard negatives are generated using Personalized PageRank random walks (A je to res? Ali samo simuliramo PPR? Sem pozabiu...) to sample difficult non‑interacted problems.
3. Training and early stopping. Within each trial we train the model for up to 200 epochs. We gradually increase the number of hard negatives per positive edge and apply early stopping if the validation recall plateaus. We use the Adam optimiser with weight decay and monitor recall@20 on the validation set.
4. Full evaluation. After hyper‑parameter search we re‑train the best model on the combined training and validation set, then evaluate it on the held‑out test set. We report Recall@20 – the proportion of true held‑out problems that appear in the top‑20 recommendations – along with its 95 % confidence interval. Other metrics could be added if desired, but recall@20 is the metric used throughout our project.

# Results (Vid)

# Conclusion

# Possible future work

# References

Gformer: https://arxiv.org/abs/2306.02330