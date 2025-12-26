# Climbing the Graph: A GNN-Powered MoonBoard Recommendation System

# Introduction (Vid)

We set out to tackle a gap in the climbing community where no reliable recommendation system exists for standardized training boards such as the MoonBoard, KilterBoard, or Tension Board. These platforms have revolutionized how climbers train by providing globally standardized setups where thousands of problems are shared and logged through mobile apps. The MoonBoard, for instance, features a standardized 18×11 grid of climbing holds, with climbers creating problems by selecting start holds (marked in green), intermediate holds (marked in blue), and a finish hold (marked in red). Similar platforms like the KilterBoard and Tension Board follow comparable principles, creating rich ecosystems of user-generated climbing problems.

![MoonBoard with climbing problems](.\\Report_images\\moonboard_problems.png)

Predicting which problem a climber will tackle next is surprisingly complex. It involves understanding user preferences, problem difficulty, hold configurations, and climbing style all intertwined in ways that traditional recommendation methods struggle to capture. The challenge is compounded by the sparse nature of climbing data: most users have only attempted a small fraction of available problems, and the relationships between users, problems, and holds form a rich, heterogeneous structure that demands sophisticated modeling.

![Presentation of a graph](.\\Report_images\\MLG_graph_2.png)

In this project, we develop a MoonBoard Recommendation System that addresses this challenge using Graph Machine Learning. Our approach models users, problems, and holds as nodes in a heterogeneous graph, with interactions as edges, enabling us to uncover hidden patterns in climbing behavior that traditional collaborative filtering misses. We experiment with multiple Graph Neural Network architectures including PinSAGE, GFormer, and custom heterogeneous GNNs using techniques such as graph-based message passing and Personalized PageRank negative sampling to handle sparse and dynamic data effectively.

Our goal is to move beyond simple popularity-based recommendations that merely suggest the most-sent problems. Instead, we aim to deliver truly personalized suggestions that match each climber's unique preferences, skill level, and climbing style. By learning from the complex relationships between users, problems, and hold configurations, we hope to build a system that helps climbers discover problems they're likely to enjoy and succeed on, rather than just the ones everyone else is doing.

The complete implementation, including data scraping tools, graph construction utilities, model implementations, and training pipelines, is available on GitHub: [https://github.com/ZanMervic/MLG-Project](https://github.com/ZanMervic/MLG-Project).

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

PinSAGE, originally developed for Pinterest's recommendation system, adapts GraphSAGE to bipartite graphs for large-scale recommendation tasks. Unlike standard GraphSAGE which operates on homogeneous graphs, PinSAGE is designed to handle the bipartite structure of user–item interactions efficiently. The model uses neighbor sampling and message passing to learn node embeddings that capture both local graph structure and node features, making it well-suited for recommendation systems where we want to leverage both interaction patterns and node attributes.

### How PinSAGE works in our pipeline

PinSAGE operates on our bipartite user–problem graph using SAGEConv layers within a heterogeneous convolution framework. The model first projects user and problem node features into a shared hidden dimension, then performs multi-layer message passing where each layer aggregates information from neighboring nodes. Unlike GFormer, PinSAGE explicitly uses node features encoding user attributes like highest grade and demographics, and problem attributes like difficulty and popularity alongside the graph structure to learn embeddings. After message passing, the model applies a final linear transformation to produce user and problem embeddings in a common latent space, enabling recommendation via dot product similarity.

In practice, integrating PinSAGE into our pipeline required several adaptations:

- Graph structure – PinSAGE operates on the bipartite graph containing only user and problem nodes, without hold nodes. We use bidirectional edges `(user, rates, problem)` and `(problem, rev_rates, user)` to enable symmetric message passing in both directions.
- Feature handling – The model uses linear projections to align user and problem features to a common hidden dimension before message passing, allowing us to leverage the rich node attributes we collected during data scraping.
- Training approach – We train PinSAGE using BPR loss with Personalized PageRank-based hard negative sampling, progressively increasing the number of hard negatives during training to improve the model's ability to distinguish between similar problems.

PinSAGE's strength lies in its ability to combine graph structure with node features effectively. The neighbor aggregation mechanism allows users and problems to influence each other's embeddings based on their connections, while the node features provide additional signal about user preferences and problem characteristics. In our experiments we tuned hyperparameters including hidden dimension, output dimension, and number of message passing layers to optimize performance on our MoonBoard dataset.

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

After extensive hyperparameter tuning across 50 random trials for each model, we evaluated the best configurations on the held-out test set. Table 1 summarizes the test Recall@20 performance for all four models, along with 95% confidence intervals computed across all test users.

| Model | Recall@20 | 95% CI | Std Dev |
|-------|-----------|--------|---------|
| **GFormer** | **0.189** | [0.183, 0.194] | 0.344 |
| Custom (SAGEConv) | 0.174 | [0.169, 0.179] | 0.320 |
| PinSAGE | 0.173 | [0.168, 0.178] | 0.321 |
| CustomAttention | 0.162 | [0.157, 0.167] | 0.320 |

*Table 1: Test set Recall@20 performance. All models were evaluated on 16,565 test users.*

The hyperparameter search revealed that models generally benefited from moderate hidden dimensions (64–128), 2–3 message passing layers, and careful tuning of hard negative sampling parameters. Learning rates varied between 0.0005 and 0.002, with weight decay consistently set to 1e-5 or 1e-4. The optimal number of hard negatives per positive edge ranged from 1 to 2, with PPR-based sampling using ranks between 10–200.

## Model Performance Analysis

GFormer achieved the highest test Recall@20 of 0.189, outperforming the other models by a small but consistent margin. We attribute this success to GFormer's ability to capture long-range dependencies through its self-attention mechanism, which allows information to flow across the entire bipartite graph rather than being limited to local neighborhoods. Unlike the other models that rely on multi-hop message passing, GFormer's Transformer layer can directly attend to all nodes, potentially discovering more complex patterns in user–problem interactions.

The Custom (SAGEConv) and PinSAGE models performed similarly (0.174 and 0.173 respectively), which is expected given their architectural similarities both use SAGEConv layers for neighbor aggregation. The slight edge of Custom over PinSAGE may stem from its ability to leverage the full heterogeneous graph structure with hold nodes, providing additional signal about problem characteristics through hold configurations.

CustomAttention performed the lowest (0.162), which was somewhat surprising given that attention mechanisms often improve model expressiveness. However, the multi-head attention in this model operates locally within neighborhoods rather than globally, and the additional complexity may have led to overfitting or made optimization more challenging with our limited training data.

## Limitations

Our evaluation reveals several important limitations that should be considered when interpreting these results:

**Data bias**: Our dataset is collected from publicly available MoonBoard logbooks, which may not represent the full diversity of the climbing community. Users who actively log their ascents may differ systematically from casual climbers in terms of skill level, motivation, or climbing style preferences. This selection bias could limit the generalizability of our recommendations to the broader climbing population.

**Metric limitations**: Recall@20 measures whether a user's actual test problems appear in the top-20 recommendations, but it doesn't guarantee that the system is truly personalizing rather than just recommending popular problems. A model could achieve reasonable recall by consistently suggesting the most-sent problems, which would work well for users who haven't tried those problems yet but fails to provide genuine personalization. Our current evaluation doesn't distinguish between these scenarios, meaning a high Recall@20 doesn't necessarily indicate that the system understands individual user preferences beyond popularity patterns.

# Conclusion

We set out to address a significant gap in the climbing community: the lack of reliable recommendation systems for standardized training boards like the MoonBoard, KilterBoard, and Tension Board. The challenge lies in predicting which problems climbers will enjoy and succeed on, which requires understanding complex relationships between user preferences, problem difficulty, hold configurations, and climbing styles all intertwined in ways that traditional recommendation methods struggle to capture.

Our MoonBoard Recommendation System demonstrates that Graph Machine Learning offers a promising approach to this problem. By modeling users, problems, and holds as nodes in a heterogeneous graph, we can leverage Graph Neural Networks to uncover hidden patterns in climbing behavior. Through extensive experimentation with multiple architectures PinSAGE, GFormer, and custom heterogeneous GNNs, we achieved test Recall@20 scores of up to 0.189, showing that personalized recommendations are feasible even with sparse climbing data.

While our results are encouraging, they also highlight important limitations: data bias from public logbooks, missing nuanced preference information, and metric limitations that don't fully distinguish true personalization from popularity-based recommendations. Nevertheless, this work establishes a foundation for building recommendation systems that can help climbers discover problems tailored to their individual preferences and abilities, moving beyond simple popularity rankings.

# Possible future work

Several directions could extend and improve this work:

- **Implementation to other platforms**: The graph-based approach we developed could be adapted to other standardized training boards like KilterBoard and Tension Board, or even extended to outdoor climbing route recommendation systems. The heterogeneous graph structure is flexible enough to accommodate different board layouts and problem formats.

- **Usage of other technologies to tackle this problem**: Future work could explore additional techniques such as incorporating computer vision to automatically extract problem characteristics from images, using natural language processing to analyze user comments and problem descriptions, or leveraging reinforcement learning to adapt recommendations based on user feedback in real-time.

- **Real-time user adaptation**: The current system provides static recommendations based on historical data. A promising direction would be to develop an adaptive system that updates recommendations as users log new ascents, learns from implicit feedback (time spent on problems, repeated attempts), and adjusts to changing user preferences and skill progression over time.

# References

Gformer: https://arxiv.org/abs/2306.02330
GitHub: [https://github.com/ZanMervic/MLG-Project](https://github.com/ZanMervic/MLG-Project)

