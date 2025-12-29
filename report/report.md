# Climbing the Graph: A GNN-Powered MoonBoard Recommendation System

We set out to tackle a gap in the climbing community where no reliable recommendation system exists for standardized training boards such as the MoonBoard, KilterBoard, or Tension Board. These platforms have revolutionized how climbers train by providing globally standardized setups where thousands of problems are shared and logged through mobile apps. The MoonBoard, for instance, features a standardized grid of climbing holds (with different versions offering various layouts), with climbers creating problems by selecting start holds (marked in green), intermediate holds (marked in blue), and a finish hold (marked in red). Similar platforms like the KilterBoard and Tension Board follow comparable principles, creating rich ecosystems of user-generated climbing problems. However, in practice, climbers tend to gravitate toward the most popular problems, even though these may not be the best fit for their individual climbing style or preferences. Discovering less popular problems that match one's specific needs is challenging, requiring extensive manual browsing through thousands of options. This is precisely the problem we aim to solve: moving beyond popularity-based selection toward truly personalized recommendations that help climbers discover problems tailored to their unique preferences and abilities.

<!-- To je samo ena "verzija/layout" moonboarda, obstajajo tudi druge ne 18x11 -->

<!-- Tu bi blo kul omenit, da trenutno (iz izkušenj) vsi plezajo le najbolj popularne smeri, čeprav te mogoče niso najboljše/primerje za stil plezanja, ki ti je všeč in da je iskanje majn popularnih smeri smotano, to je dejanski problem, ki ga hočemo rešit -->
![Presentation of a graph](.\images\cover_image.png)
*An example of the MoonBoard and a problem in the app*

<!-- TODO: dodaj sliko enega ki pleza/moonboard stene -->

Building a recommendation system that goes beyond simple popularity rankings presents significant technical challenges. It requires understanding how user preferences, problem difficulty, hold configurations, and climbing style interact in ways that traditional recommendation methods struggle to capture. The challenge is further compounded by the sparse nature of climbing data: most users have only attempted a small fraction of available problems, making it difficult to identify patterns in individual preferences. Moreover, the relationships between users, problems, and holds form a rich, heterogeneous structure that demands sophisticated modeling to uncover the hidden connections that can help match climbers with problems they will genuinely enjoy.

![Presentation of a graph](.\images\MLG_graph_2.png)
*A visualization of the graph structure, showing the relationships between users, problems, and holds*

In this article, we develop a MoonBoard Recommendation System that addresses this challenge using Graph Machine Learning. Our approach models users, problems, and holds as nodes in a heterogeneous and bipartite graph, with interactions as edges, enabling us to uncover hidden patterns in climbing behavior that traditional collaborative filtering misses. We experiment with multiple Graph Neural Network architectures including PinSAGE, GFormer, and custom heterogeneous GNNs using techniques such as graph-based message passing and random walk based negative sampling to handle sparse and dynamic data effectively.

<!-- heterogeneous + bipartite

- mogoče in this article namest in this project
- ne delamo personalized pageranka ubistvu
-->

Our goal is to move beyond simple popularity-based recommendations that merely suggest the most-sent problems. Instead, we aim to deliver truly personalized suggestions that match each climber's unique preferences, skill level, and climbing style. By learning from the complex relationships between users, problems, and hold configurations, we hope to build a system that helps climbers discover problems they're likely to enjoy and succeed on, rather than just the ones everyone else is doing.

The complete implementation, including data scraping tools, graph construction utilities, model implementations, and training pipelines, is available on GitHub: [https://github.com/ZanMervic/MLG-Project](https://github.com/ZanMervic/MLG-Project).

<!-- Naš github je trenutno kar messy, puhno neke spageti kode, stare kode itd. Malo sem ga poskusil urediti, pa readme.md posodobiti ampak ni še to to -->

# Dataset (Žan)

There is no clean, publicly available dataset for MoonBoard activities and problems. To train and evaluate our models, we therefore built our own dataset by scraping publicly visible MoonBoard data (users, problems, and logged ascents) and then enriching it with additional problem structure by extracting hold positions from problem screenshots. The resulting JSON files form the basis for the graphs used throughout this project.

### Scraping the MoonBoard database

We built a Selenium‑based scraper that crawls the MoonBoard website and logbooks of registered users. Focusing on the popular MoonBoard Masters 2019 hold setup, the scraper iterates through pages of users and, for each user, opens their climbing logbook. For every ascent in the logbook it records:

- User attributes such as the climber’s highest grade, height, weight and the number of problems they have sent.
- Problem metadata including the grade, average star rating, total number of ascents (`num_sends`), setter and any recorded foot rules.
- Interaction details - the grade the user assigned to the problem, their rating, number of attempts and an optional comment, along with the date of the ascent.

### Processing problem images

A big part of what makes a user like or dislike a climbing problem are the holds used and their arrangement, so we wanted to capture this information as well.
To reconstruct the holds used in each problem we captured images of every problem using a script which runs the MoonBoard app on a device (for example, a phone or emulator) and captures a screenshot of each problem. We then used simple computer‑vision rules to detect the green, blue and red circles that mark the start, middle and end holds. Once located, the positions are converted into hold identifiers and saved in JSON files. These hold lists will later be used when constructing the heterogeneous graph.

![MoonBoard with climbing problems](.\images\moonboard_problems.png)

*Screenshots of MoonBoard problems used for hold extraction. Colored circles indicate detected start (green), middle (blue), and end (red) holds, which are converted into discrete hold identifiers for graph construction.*

<!-- TODO: caption -->

### Dataset size and attributes

At the time of writing we had collected 25865 users, 38663 problems and 1668865 logged ascents. Each JSON entry contains all information needed to build our graphs. Below is a simplified example showing a user and a problem entry (formatted as JSON):

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

*Sample JSON structure illustrating the schema for user profiles (including ascent history) and problem metadata (including hold configurations).*

In summary, our dataset contains user metadata, problem metadata and a log of every ascent. We also store processed hold positions for each problem. These files serve as the starting point for constructing the various graphs used in the recommendation models.

# Graph (Žan)

Once we had the raw data, the next step was to convert it into graphs that our models could learn from. Recommender systems typically represent users and items as nodes in a bipartite graph, with edges representing interactions. Our domain, however, also has a natural third entity: holds. To accommodate different models we built both bipartite and heterogeneous graphs.

### Node and edge types

We used PyTorch Geometric’s `HeteroData` container to assemble a single heterogeneous graph. The graph contains up to three node types:

| Node type       | Features                                             | Description                                                                  |
| --------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| user            | `[highest_grade_idx, height, weight, problems_sent]` | Encodes climber ability and demographics.                                    |
| problem         | `[grade, rating, num_sends, foot_rules]`             | Captures problem difficulty, popularity and foot rules.                      |
| hold (optional) | One‑hot identity vector                              | Uniquely identifies each board hold; used only in the heterogeneous version. |

*Overview of node types in the heterogeneous graph and their associated feature vectors, describing how climbers, problems, and holds are represented.*

<!-- Edges connect these nodes in two ways. Each ascent produces a `(user, rates, problem)` edge with attributes `[grade, rating, attempts]` and a timestamp; a reverse edge `(problem, rev_rates, user)` is added to make message passing symmetric. When using hold nodes we also add `(problem, contains, hold)` edges with one‑hot flags indicating whether the hold is a start, middle or end hold, and the corresponding reverse edges `(hold, rev_contains, problem)`. -->

Edges encode climber-problem interactions and (optionally) which holds each problem uses. To make message passing symmetric, we add reverse edges for every relation.

| Edge type                                                                          | Attributes                               | Description                                                                   |
| ---------------------------------------------------------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------- |
| `(user, rates, problem)` `(problem, rev_rates, user)`                              | `[grade, rating, attempts]`, `timestamp` | A logged ascent / interaction from a user to a problem.                       |
| `(problem, contains, hold)` `(hold, rev_contains, problem)` _(heterogeneous only)_ | one‑hot `[is_start, is_middle, is_end]`  | Connects a problem to holds it uses and encodes the hold role in the problem. |

*Overview of edge types and attributes connecting users, problems, and holds, including reverse relations for symmetric message passing.*

### Graph variations

Our models have different requirements. We therefore constructed two types of graphs:

- **Bipartite graph:** This contains only user and problem nodes with edges representing each logged ascent. The node feature matrices for users and problems are retained, but there are no hold nodes. This simpler structure is compatible with PinSAGE and with GFormer.

- **Heterogeneous graph:** This version adds hold nodes connected to their respective problems. We keep all node features and edge attributes. This richer representation is used by our Custom and CustomAttention models, which operate on PyG HeteroData objects and can learn from multiple node and edge types.

### Graph statistics

To sanity-check our construction (and to give a feel for scale and sparsity), we computed a set of basic graph diagnostics. These numbers are useful for spotting obvious issues (e.g., disconnected subgraphs, unexpectedly dense connectivity), and they also help interpret model behavior especially for message-passing GNNs shortest-path distances matter. Unless noted otherwise, the statistics below refer to the **heterogeneous** graph; when the **bipartite** graph differs, we report both.

#### Size and connectivity

| Statistic                      | Heterogeneous graph | Bipartite graph |
| ------------------------------ | ------------------: | --------------: |
| # user nodes                   |              25,826 |          25,826 |
| # problem nodes                |              34,572 |          34,572 |
| # hold nodes                   |                 198 |               / |
| # user-problem edges           |           1,627,580 |       1,627,580 |
| # problem-hold edges           |             304,852 |               / |
| connected components           |                   1 |               6 |
| diameter (largest component)   |                   6 |               7 |
| density                        |         1.05257e-03 |    8.923483e-04 |
| average clustering coefficient |                   0 |               0 |

*Basic structural statistics of the heterogeneous and bipartite graphs, including node and edge counts, connectivity, diameter, density, and clustering coefficient.*
> Note: A clustering coefficient of 0 is expected for bipartite-like graphs, since triangles generally cannot form when edges only connect across partitions.

#### Degree statistics by node type

| Node type                   | min degree | max degree | mean degree | median degree | std. dev. |
| --------------------------- | ---------: | ---------: | ----------: | ------------: | --------: |
| user                        |          1 |      3,000 |       63.02 |            20 |    118.73 |
| problem                     |          3 |     14,861 |       55.90 |            12 |    459.03 |
| hold _(heterogeneous only)_ |         42 |      8,663 |    1,539.00 |      1,017.50 |  1,470.20 |

*Degree distribution statistics for each node type in the heterogeneous graph, showing connectivity patterns and variance across users, problems, and holds.*

The figure below illustrates the bipartite version of our graph: users connect to problems, with edges labelled by interactions (TODO, bolje opiši, ko bo slikca).

![Bipartite graph](.\images\bipartite.png)

*Illustration of the bipartite graph representation, where user nodes connect to problem nodes via interaction edges representing logged ascents, with node features attached to users and problems.*

The figure below illustrates the heterogenous version of our graph: users again connect to problems, however here, problems are also connected to holds (TODO, bolje opiši, ko bo slikca).

![Heterogeneous graph](.\images\heterogeneous.png)

*Illustration of the heterogeneous graph representation, extending the bipartite graph by adding hold nodes and problem–hold edges to explicitly model which holds are used in each problem.*

---

# Approach

<!-- Tukaj bi kakšen "subsection" bil koristen, da se lažje znajdeš -->

We frame the recommendation task as a link prediction problem on the graphs described above. Recommending new problems then corresponds to predicting which missing user-problem edges are most likely to appear in the future. Given a user together with their features and previously climbed problems the model assigns a score to each potential user-problem edge, estimating the likelihood of a future interaction. These scores are then used to rank all unclimbed problems for the user, forming the basis for our evaluation.

<!-- samo gpt uporablja tolko - -->
<!-- Tej "—" so kar problematični, ker se res hitro vidi da je to chat napisal, raje zamenjat s "-" -->

## Metrics

To evaluate the quality of these rankings, we use Recall@k. Recall@k measures how well the model retrieves relevant problems among its top-ranked predictions. For each user, the relevant set consists of the problems they actually climbed in the evaluation split. Given the model’s ranked list of all candidate problems, Recall@k is defined as the proportion of these relevant problems that appear within the top k positions, as illustrated in the figure below. Throughout this work, we report Recall@20 as our primary evaluation metric.

![Recall@k example image](.\images\recallatk.png)

*An illustration showing how Recall@k is calculated.*

## Splitting the graph

But how do we obtain these rankings in the first place? To do so, we must first train our models, which requires carefully splitting the interaction graph into separate subgraphs for training and evaluation.

Since recommender systems are evaluated on their ability to predict future interactions, we apply a temporal per-user split. For each user, we sort their interactions chronologically and allocate the earliest 70% to a message-passing set, followed by 10% each to the training, validation, and test sets. The message-passing edges are used to build node representations, while the train, validation, and test edges are withheld for loss optimisation and evaluation.

This per-user temporal split has several important advantages. First, it prevents user-level temporal leakage because the model is never trained on interactions that occur after the ones it is asked to predict. While this approach does not enforce a globally time-consistent graph, it still closely reflects how recommender systems are deployed in practice. Second, splitting per user rather than globally ensures that every user contributes data to all splits, which helps avoid cold-start effects in the validation and test sets. Finally, this setup mirrors the real-world recommendation scenario where a system must make predictions based only on a user’s past activity and adapt as new interactions arrive over time.

The figure below shows an example split of a graph for link prediction.

![Link prediction split](.\images\splits.png)

*An example of a split for link prediction. Source: Stanford CS224W*

## Loss function

With these splits in place, we can begin training our models. Since Recall@k is not differentiable, it cannot be used directly as a training objective. Instead, we optimise the models using Bayesian Personalized Ranking (BPR) loss, which is specifically designed for pairwise ranking and aligns well with the goal of maximizing Recall@k. BPR is a pairwise ranking approach that trains the model to score observed interactions higher than unobserved ones. For each user, the model is presented with a positive example, such as a problem the user has climbed, and a negative example, such as a problem the user has not climbed. The training objective then encourages the model to assign a higher score to the positive interaction than to the negative one. This approach directly optimizes the relative ordering of items, making it well suited for recommendation tasks evaluated with metrics like Recall@k.

## Negative sampling

Not all negative examples are equally informative for training. We distinguish between easy negatives (sampled uniformly from all unclimbed problems), which are problems the user has clearly not attempted, and hard negatives, which are problems the user has not climbed but are similar to ones they have. Relying only on easy negatives can make training too simple and less effective. For example, if a user has climbed mostly beginner problems, treating an advanced problem they would never attempt as a negative does not teach the model much about ranking relevant items. Hard negatives, in contrast, challenge the model to distinguish between problems the user might realistically climb and those they are unlikely to, leading to better ranking performance.

To obtain hard negatives, we simulate random walks on the training graphs starting from each user and count how many times each problem is visited. Problems that are visited more frequently are more closely connected to the user, either directly or through similar users, making them more likely to be relevant. We then rank the unclimbed problems by visit count and sample hard negatives randomly from a specified range of ranks. By focusing on these intermediate-ranked problems, the model is challenged to learn fine-grained distinctions between problems the user might actually climb and those they are unlikely to.

```python
def simulate_random_walks(
    edges: torch.tensor, num_users: int, walks_per_user=10, walk_length=100
):
    """Simulate random walks on a graph given by edges starting at the first num_users nodes."""
    users = torch.tensor(range(num_users))
    batch_size = 500000 // (walks_per_user * walk_length)
    ppr_edge_index, ppr_values = [], []
    # batch random walks by users
    for batch in users.split(batch_size):
        start_users = batch.repeat_interleave(walks_per_user)
        # simulate random walks using random_walk from torch_cluster
        rw = random_walk(edges[0], edges[1], start=start_users, walk_length=walk_length)
        # Flatten to get (source, target) pairs
        user_ids = start_users.repeat_interleave(rw.size(1))
        visited = rw.flatten()

        # get counts
        uniq, counts = torch.unique(
            torch.stack([user_ids, visited], dim=0), dim=1, return_counts=True
        )
        ppr_edge_index.append(uniq)
        ppr_values.append(counts)
    return torch.cat(ppr_edge_index, dim=1), torch.cat(ppr_values)
```

<!-- ta celi Approch section bi lahko malo skrajšali... obvezno kaka slika, graf,...  -->
<!-- Ne vem, če bi skrajšal, ker je kar pomembno, ampak definitivno razdelit na subsectione. -->

# Models

With the training procedure, negative sampling strategy, and evaluation setup in place, our framework is ready for model training and assessment. The next step is to describe the models we used and how they operate within this framework.

## Pinsage

PinSAGE, originally developed for Pinterest's recommendation system, adapts GraphSAGE to bipartite graphs for large-scale recommendation tasks. Unlike standard GraphSAGE which operates on homogeneous graphs, PinSAGE is designed to handle the bipartite structure of user-item interactions efficiently.

PinSAGE operates on our bipartite user-problem graph using SAGEConv layers. The model projects user and problem node features into a shared hidden dimension, then performs multi-layer message passing where each layer aggregates information from neighboring nodes. Unlike GFormer, PinSAGE explicitly uses node features encoding user attributes like highest grade and demographics, and problem attributes like difficulty and popularity alongside the graph structure to learn embeddings. After message passing, the model applies a final linear transformation to produce user and problem embeddings in a common latent space, enabling recommendation via dot product similarity.

We train PinSAGE using BPR loss with random walk-based hard negative sampling, progressively increasing the number of hard negatives during training.

<!-- Referenciraj članek -->
<!-- Spremeni v en/dva odstavka, brez tistega subsectina -->

## GFormer (Žan)

Graph neural networks such as GraphSAGE and GAT aggregate information locally. To capture long‑range dependencies we experimented with Graph Transformers.

We chose GFormer, a graph transformer for recommendation, due to its open source code, and recommendation specific Transformer architecture.

At a high level, GFormer constructs node representations directly from the user-problem interaction matrix, without explicit neighborhood based message passing. A Transformer style self-attention mechanism is then applied, allowing nodes to attend globally to others in the graph and capture long-range dependencies.

In our setup, we applied GFormer to the user-problem bipartite graph, omitting hold nodes as well as all node and edge features. The graph was represented as a normalised adjacency matrix, which matches the input format expected by the model.

<!-- Referenciraj članek -->

## Heterogeneous Models

Up to this point, our models have not made use of the additional structure available in the data, namely the fact that problems can be connected through the holds they share. While we already constructed a heterogeneous graph that captures these relationships, this structure cannot be handled by traditional recommender system models, which typically assume a simple user-item bipartite graph. As a result, leveraging this information requires us to design custom models that can operate directly on graphs with multiple node and edge types.

We define these heterogeneous GNN models (Schlichtkrull et al., 2018) by using edge-type-specific message passing functions, meaning that each edge type is associated with its own set of learnable weights and its own message passing architecture. This allows the model to apply different transformations to user-problem and problem-hold interactions, reflecting the different semantics of these connections. During message passing, nodes aggregate information from their neighbours using the transformation corresponding to the edge type, and the resulting messages are then combined to form a single representation.

In our experiments, we evaluate two such architectures, one based on GraphSAGE layers (we call this model "Hetero") and one based on GAT layers (we call this model "Hetero Attention"). As with our other models, we tune key hyperparameters such as the number of layers, the latent dimension, and, for attention-based models, the number of attention heads.

```python
class Custom(nn.Module):
    def __init__(self, hetero_data, hidden_channels=64, output_lin=False, num_layers=2, dropout=0.1):
        super().__init__()

        node_types, edge_types = hetero_data.metadata()
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout

        # 1) Linear "input" layer per node type: feature_dim -> hidden_channels
        # This layers will project raw node features to a common hidden dimension
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            # Get feature dim for this node type
            in_channels = hetero_data[node_type].x.size(-1) 
            # Create linear layer and store in dict
            self.lin_dict[node_type] = nn.Linear(in_channels, hidden_channels)

        # 2) Several HeteroConv layers for message passing
        # HeteroConv allows us to define different conv layers per edge type
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                # edge_type is a tuple: (src_type, rel_name, dst_type)
                conv_dict[edge_type] = SAGEConv(
                    (-1, -1),  # infer input dims from x_dict at runtime
                    hidden_channels,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # 3) Final linear layers per node type (optional)
        if output_lin:
            self.lin_dict_out = nn.ModuleDict()
            for node_type in node_types:
                self.lin_dict_out[node_type] = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {"user": user_x, "problem": problem_x, "hold": hold_x}
        # edge_index_dict: {("user","rates","problem"): edge_index, ...}

        # 1) Project raw features to hidden dim
        h_dict = {}
        for node_type, x in x_dict.items():
            h = self.lin_dict[node_type](x)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_dict[node_type] = h

        # 2) Message passing
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            # Apply non-linearity + dropout after each layer
            for node_type in h_dict:
                h = F.relu(h_dict[node_type])
                h = F.dropout(h, p=self.dropout, training=self.training)
                h_dict[node_type] = h

        # 3) Final linear layer per node type (if defined)
        if hasattr(self, 'lin_dict_out'):
            for node_type in h_dict:
                h = self.lin_dict_out[node_type](h_dict[node_type])
                h_dict[node_type] = h

        # 4) Return final node embeddings per type
        return h_dict
```

  <!-- Mogoče je bolje da je kratko in bi bilo kul, da se use sectione modelov skrajša -->
  <!-- Tu bi bilo mogoče kul, še malo o sami implementaciji/strukturi in ideji modelov povedat, ker so edini, ki smo jih mi implementirali -->
  <!-- Specifične stvari za heterogene modele e.g. kako maš za vsak edge type svoj "pipeline" -->

# Evaluation (Žan)

Evaluating a recommender system involves both hyper‑parameter tuning and robust metrics. Our evaluation pipeline operates as follows:

1. Temporal splitting. We perform the 70/10/10/10 split described above, ensuring that training only uses information available before the interaction being predicted.
2. Hyper‑parameter search. For each model we run a random search over learning rate, embedding dimension, number of layers, dropout, and model‑specific parameters such as the number of attention heads. Each trial trains the model on the training set using the BPR loss.
3. Training and early stopping. Within each trial we train the model for up to 200 epochs. We gradually increase the number of hard negatives per positive edge and apply early stopping if the validation recall plateaus. We use the Adam optimiser with weight decay and monitor recall@20 on the validation set.
4. Full evaluation. After hyper‑parameter search we re‑train the best model on the combined training and validation set, then evaluate it on the held‑out test set. We report Recall@20 - the proportion of true held‑out problems that appear in the top‑20 recommendations - along with its 95 % confidence interval. Other metrics could be added if desired, but recall@20 is the metric used throughout our project.

# Results

After extensive hyperparameter tuning across 50 random trials for each model, we evaluated the best configurations on the held-out test set. Table 1 summarizes the test Recall@20 performance for all four models, along with 95% confidence intervals computed across all test users.

| Model             | Recall@20 | 95% CI         | Std Dev |
| ----------------- | --------- | -------------- | ------- |
| Popularity baseline | 0.344     | [0.341, 0.347] | -       |
| **GFormer**       | **0.189** | [0.183, 0.194] | 0.344   |
| Hetero (SAGEConv) | 0.174     | [0.169, 0.179] | 0.320   |
| PinSAGE           | 0.173     | [0.168, 0.178] | 0.321   |
| Hetero Attention  | 0.162     | [0.157, 0.167] | 0.320   |

_Table 1: Test set Recall@20 performance. All models were evaluated on 16,565 test users._

<!--
- men se zdi Heterogeneou lepše ime kot pa Custom
 Res je -->

The hyperparameter search revealed that models generally benefited from moderate hidden dimensions (64-128), 2-3 message passing layers, and careful tuning of hard negative sampling parameters. Learning rates varied between 0.0005 and 0.002, with weight decay consistently set to 1e-5 or 1e-4. The optimal number of hard negatives per positive edge ranged from 1 to 2, with PPR-based sampling using ranks between 10-200.

As a baseline, we also evaluated a simple popularity-based approach that recommends the most-sent problems. This baseline achieved a Recall@20 of 0.344 ± 0.002881, significantly outperforming all our GNN models. While this result highlights the dominance of popular problems in the dataset, it also underscores the challenge of developing truly personalized recommendations that go beyond simple popularity rankings.

## Model Performance Analysis

GFormer achieved the highest test Recall@20 of 0.189 among our GNN models, outperforming the other models by a small but consistent margin, though still falling short of the popularity baseline's 0.344. We attribute this success to GFormer's ability to capture long-range dependencies through its self-attention mechanism, which allows information to flow across the entire bipartite graph rather than being limited to local neighborhoods. Unlike the other models that rely on multi-hop message passing, GFormer's Transformer layer can directly attend to all nodes, potentially discovering more complex patterns in user-problem interactions.

The Custom (SAGEConv) and PinSAGE models performed similarly (0.174 and 0.173 respectively), which is expected given their architectural similarities both use SAGEConv layers for neighbor aggregation. The slight edge of Custom over PinSAGE may stem from its ability to leverage the full heterogeneous graph structure with hold nodes, providing additional signal about problem characteristics through hold configurations.

<!-- To predvidevam da je chat napisal, tkoda bi blo kul preverit, če je res heh 
- nikjer nismo omenl našga baselinea
  <!-- Res je -->
  <!-- Pa "-" je spet treba zamenjat s "-" -->
  <!--
- men se zdi Heterogeneou lepše ime kot pa Custom
 Res je -->


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

Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., and Welling, M. (2018).
*Modeling Relational Data with Graph Convolutional Networks*.

Gformer: https://arxiv.org/abs/2306.02330
GitHub: [https://github.com/ZanMervic/MLG-Project](https://github.com/ZanMervic/MLG-Project)

Cover Image: https://thefrontclimbingclub.com/wp-content/uploads/2017/12/front-ogden-moonboard.jpg

moonboard: https://moonclimbing.com/moonboard
Pinsage: https://arxiv.org/abs/1806.01973v1
heterogene modele: https://arxiv.org/pdf/1703.06103
GraphSage: https://arxiv.org/abs/1706.02216v4


<!-- TODO reference za:
moonboard: https://moonclimbing.com/moonboard
Pinsage: https://arxiv.org/abs/1806.01973v1
heterogene modele: https://arxiv.org/pdf/1703.06103
GraphSage: https://arxiv.org/abs/1706.02216v4
-->
