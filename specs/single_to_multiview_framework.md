# A Generalized Framework & Taxonomy for Converting Single-View Clustering Algorithms into Multi-View Models

- **Document Type**: Technical Specification / Conceptual Framework
- **Related Papers**:
  - `papers/campello2013_hdbscan.pdf` (HDBSCAN Foundation)
  - `papers/yang_wang2018_mvc_survey.pdf` (MVC Survey & Taxonomy)
  - `papers/daume2011_cotraining.pdf` (Co-training Multi-View Spectral Clustering)
  - `papers/kumar2011_coreg_mvc.pdf` (Co-regularized Multi-View Spectral Clustering)
- **Target Specification**: `specs/single_to_multiview_framework.md`

---

## 1. Foundational Theory: Why and How Multi-View Clustering Works

Multi-view clustering (MVC) partitions a dataset of $n$ instances $X = \{x_1, \dots, x_n\}$ where each instance $x_i$ is described by $V \ge 2$ distinct feature representations (views):
$$x_i = \left( x_i^{(1)}, x_i^{(2)}, \dots, x_i^{(V)} \right), \quad x_i^{(v)} \in \mathbb{R}^{d_v}$$

### 1.1 The Two Governing Principles
1. **The Complementary Principle**:
   - Distinct views capture heterogeneous properties of the same underlying physical entity (e.g., an image described by color histograms, edge orientations, and text tags).
   - Individual views are rarely complete on their own; fusing multiple views provides comprehensive structural discriminability that no single view possesses in isolation.
2. **The Consensus Principle**:
   - Because all views observe the same entities, their respective clustering hypotheses must agree on the underlying cluster assignments.
   - **Theoretical PAC Bound (Dasgupta et al., 2001)**:
     $$P\left(f^{(1)} \neq f^{(2)}\right) \ge \max\left\{ P_{\text{err}}\left(f^{(1)}\right), \, P_{\text{err}}\left(f^{(2)}\right) \right\}$$
     The probability of disagreement between two independent view-hypotheses serves as an upper bound on the error rate of either hypothesis alone. Consequently, **minimizing cross-view disagreement (or maximizing cross-view consensus) directly reduces clustering error**.

### 1.2 The Fundamental Tension in Multi-View Design
Any multi-view clustering framework must balance two competing forces:
- **Intra-View Fidelity**: Preserving the unique geometric, manifold, or density structure specific to view $v$.
- **Inter-View Consensus**: Driving the disparate views toward a unified clustering partition or consensus representation.

---

## 2. The Four Opportune Moments of Fusion: A Taxonomy of Conversion Approaches

Every single-view clustering algorithm follows a conceptual pipeline:
$$\text{Data Input } (X) \longrightarrow \text{Proximity / Metric / Kernel } (D, K, G) \longrightarrow \text{Model Optimization / Embeddings } (U, H, T) \longrightarrow \text{Partition / Decision } (Y)$$

Converting a single-view algorithm to a multi-view model can occur at any of the **four opportune moments of fusion**:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                    TAXONOMY OF MULTI-VIEW FUSION PARADIGMS                                            |
+-----------------------------------------------------------------------------------------------------------------------+

 1. EARLY FUSION (Input / Feature Level)
    [View 1 (d1)] ──┐
    [View 2 (d2)] ──┼──> [Unified Feature / Latent Representation (d*)] ──> [Single-View Clustering Algorithm] ──> Labels
    [View V (dV)] ──┘

 2. INTERMEDIATE FUSION (Proximity / Graph / Metric Level)
    [View 1] ──> [Metric / Affinity G1] ──┐
    [View 2] ──> [Metric / Affinity G2] ──┼──> [Consensus Graph / Distance G*] ──> [Single-View Algorithm] ──> Labels
    [View V] ──> [Metric / Affinity GV] ──┘

 3. MODEL & OBJECTIVE CO-REGULARIZATION (Latent Embeddings / Hypothesis Level)
    [View 1 Model] <──── Co-Training / Disagreement Penalty / Consensus Centroid ────> [View 2 Model]
          │                                                                                   │
          └───────────────────────────> Joint Optimization Objective <────────────────────────┘
                                                    │
                                                    ▼
                                            Consensus Solution

 4. LATE FUSION (Decision / Partition Level)
    [View 1] ──> [Algorithm 1] ──> [Partition 1] ──┐
    [View 2] ──> [Algorithm 2] ──> [Partition 2] ──┼──> [Co-Association / Hypergraph / Consensus Tree] ──> Final Labels
    [View V] ──> [Algorithm V] ──> [Partition V] ──┘
+-----------------------------------------------------------------------------------------------------------------------+
```

---

### 2.1 Level 1: Early Fusion (Feature / Input Space)

#### Mechanism
Transform the raw feature matrices $\{X^{(1)}, \dots, X^{(V)}\}$ into a single unified matrix $X^* \in \mathbb{R}^{n \times d^*}$ before running any standard single-view clustering algorithm.

#### Key Approaches
1. **Feature Concatenation**:
   $$X^* = \left[ X^{(1)} \,\|\, X^{(2)} \,\|\, \dots \,\|\, X^{(V)} \right] \in \mathbb{R}^{n \times \sum d_v}$$
2. **Linear / Non-linear Subspace Learning (CCA, Kernel-CCA, Autoencoders)**:
   - Maximize correlation between projected views: $\max_{W_1, W_2} \text{corr}(X^{(1)} W_1, X^{(2)} W_2)$.
   - Use multi-view deep autoencoders with a shared bottleneck layer.
3. **Joint Non-negative Matrix Factorization (NMF)**:
   - Factorize each view toward a shared coefficient matrix $H^*$:
     $$\min_{W^{(v)} \ge 0, H^* \ge 0} \sum_{v=1}^V \|X^{(v)} - W^{(v)} H^*\|_F^2$$

#### Strengths & Limitations
- ✅ **Pros**: Completely agnostic to the downstream clustering algorithm; allows reuse of standard single-view tools without modification.
- ❌ **Cons**: Feature concatenation suffers from the curse of dimensionality; dominates by high-dimensional views; destroys view-specific distance metrics and non-linear manifold topologies.

---

### 2.2 Level 2: Intermediate Fusion (Proximity, Graph & Metric Space)

#### Mechanism
Calculate pairwise proximity matrices, affinity graphs, kernels, or density estimators **independently within each view** using view-appropriate metrics, and then combine them into a single **consensus proximity structure** $G^*$ or $K^*$ prior to clustering.

#### Key Approaches
1. **Auto-Weighted Multi-Kernel Learning (MKL)**:
   - Learn view weights $\{\alpha_v\}$ to form a convex combination of kernels:
     $$K^* = \sum_{v=1}^V \alpha_v^r K^{(v)}, \quad \sum_{v=1}^V \alpha_v = 1, \; \alpha_v \ge 0, \; r > 1$$
2. **Graph / Affinity Fusion & Cross-Diffusion**:
   - Construct affinity matrices $W^{(v)}$ per view.
   - Use **Similarity Network Fusion (SNF)** or **Cross-Diffusion** to iteratively diffuse similarities across view networks:
     $$P^{(v)}_{t+1} = S^{(v)} \times \left( \frac{1}{V-1} \sum_{w \neq v} P^{(w)}_t \right) \times \left(S^{(v)}\right)^T$$
3. **Adaptive / Parameter-Free Graph Fusion (Nie et al., 2016)**:
   - Automatically assign graph weights based on distance to the consensus Laplacian:
     $$\min_{S^*, \alpha_v} \sum_{v=1}^V \alpha_v \|S^* - S^{(v)}\|_F^2 \quad \text{or} \quad \min_{F^T F = I, \alpha_v} \sum_{v=1}^V \alpha_v^r \text{tr}(F^T L^{(v)} F)$$
4. **Multi-View Distance & Density Fusion (Crucial for Density-Based Clustering)**:
   - Aggregate view-specific distance metrics $d^{(v)}(x_i, x_j)$ via generalized $p$-norms or power means:
     $$d^*(x_i, x_j) = \left( \sum_{v=1}^V w_v \left(d^{(v)}(x_i, x_j)\right)^p \right)^{1/p}$$
   - Aggregate view-specific core distances $d_{\text{core}}^{(v)}(x_i)$ into a unified multi-view density indicator.

#### Strengths & Limitations
- ✅ **Pros**: Respects the natural metric space of each view; accommodates heterogeneous feature types (e.g., text cosine distance + image Euclidean distance + graph shortest path); directly feeds into graph/spectral/density pipelines.
- ❌ **Cons**: Requires careful normalization of distance/similarity scales across disparate views.

---

### 2.3 Level 3: Model & Objective Co-Regularization (Latent Representations & Hypotheses)

#### Mechanism
Maintain view-specific models simultaneously and formulate a **joint optimization objective** containing view-specific loss terms and an explicit **cross-view regularizer / coupling penalty** that enforces consensus across view embeddings.

#### Key Approaches
1. **Co-Training / Subspace Bootstrapping (Kumar & Daumé III, 2011)**:
   - In unsupervised settings, the discriminative eigenvectors $U^{(v)}$ of view $v$ serve as soft cluster indicators.
   - Project the similarity matrix $K^{(w)}$ of view $w$ onto the subspace of view $v$:
     $$S^{(w)} = \text{sym}\left( U^{(v)} (U^{(v)})^T K^{(w)} \right)$$
   - Iteratively recompute Laplacians and eigenvectors until consensus stabilizes.
   - *Effect*: Smooths intra-cluster edges and suppresses cross-cluster edges ("gluing" points agreed upon by other views).

2. **Pairwise Co-Regularization (Kumar, Rai, & Daumé III, 2011)**:
   - Penalize disagreement between view embedding Gram matrices:
     $$\max_{\{U^{(v)}\}} \sum_{v=1}^V \text{tr}\left( (U^{(v)})^T \mathcal{L}^{(v)} U^{(v)} \right) + \lambda \sum_{v \neq w} \text{tr}\left( U^{(v)} (U^{(v)})^T U^{(w)} (U^{(w)})^T \right) \quad \text{s.t. } (U^{(v)})^T U^{(v)} = I$$
   - Solved via alternating maximization where each step is single-view spectral clustering on an **adaptively modified Laplacian**:
     $$\tilde{\mathcal{L}}^{(v)} = \mathcal{L}^{(v)} + \lambda \sum_{w \neq v} U^{(w)} (U^{(w)})^T$$

3. **Centroid-Based Co-Regularization**:
   - Introduce a consensus embedding centroid $U^* \in \mathbb{R}^{n \times k}$ ($U^{*T} U^* = I$):
     $$\max_{\{U^{(v)}\}, U^*} \sum_{v=1}^V \text{tr}\left( (U^{(v)})^T \mathcal{L}^{(v)} U^{(v)} \right) + \sum_{v=1}^V \lambda_v \text{tr}\left( U^{(v)} (U^{(v)})^T U^* U^{*T} \right)$$
   - Optimizes view-specific embeddings $U^{(v)}$ and consensus $U^*$ in alternation; final clustering is extracted directly from $U^*$.

#### Strengths & Limitations
- ✅ **Pros**: Strongest mathematical framework; flexible weighting ($\lambda_v$) for noisy or uninformative views; avoids premature information loss by iteratively discovering structure.
- ❌ **Cons**: Requires iterative optimization solvers; higher computational overhead than direct graph fusion.

---

### 2.4 Level 4: Late Fusion (Ensemble / Decision Space)

#### Mechanism
Run single-view clustering algorithms completely independently on each view to generate $V$ separate partitions or hierarchies $\{P^{(1)}, \dots, P^{(V)}\}$, and then combine them into a final consensus clustering.

#### Key Approaches
1. **Co-Association Matrix Clustering**:
   - Construct an $n \times n$ matrix $C$ where $C_{ij} = \frac{1}{V} \sum_{v=1}^V \mathbb{I}\left( P^{(v)}(x_i) == P^{(v)}(x_j) \right)$.
   - Run Single-Linkage, Average-Linkage, or Spectral Clustering on $C$.
2. **Hypergraph Partitioning**:
   - Represent each cluster in each view as a hyperedge in a hypergraph.
   - Apply algorithms like CSPA (Cluster-based Similarity Partitioning), HGPA (HyperGraph Partitioning), or MCLA (Meta-CLUstering Algorithm).
3. **Hierarchical / Dendrogram Consensus**:
   - Align and merge multiple hierarchical trees using tree-edit distances, partition intersection lattices, or joint subtree stability optimization.

#### Strengths & Limitations
- ✅ **Pros**: Completely decentralized and parallelizable; naturally robust to corruptions or missing views; allows mixing entirely different clustering algorithms across views.
- ❌ **Cons**: Loses fine-grained metric and continuous density information; views cannot assist each other during representation learning.

---

## 3. Algorithm-Family-Specific Conversion Recipes

```
+----------------------------------------------------------------------------------------------------------------------+
| Single-View Algorithm Family | Recommended Multi-View Conversion Strategies                                          |
+------------------------------+---------------------------------------------------------------------------------------+
| Centroid-Based               | • Multi-View K-Means (weighted objective over view centroids)                         |
| (e.g., K-Means, K-Medoids)   | • Multi-Kernel K-Means (MKKM) with auto-learned weights                               |
|                              | • Co-clustering / Joint NMF with shared indicator matrix H*                           |
+------------------------------+---------------------------------------------------------------------------------------+
| Spectral & Graph-Based       | • Co-Training Spectral Clustering (iterative subspace projection)                     |
| (e.g., Normalized Cuts)      | • Pairwise / Centroid Co-regularization (adaptive Laplacians)                          |
|                              | • Affinity Network Fusion (SNF) / Auto-weighted Graph Fusion                           |
+------------------------------+---------------------------------------------------------------------------------------+
| Density-Based & Hierarchical | • Multi-View Core Distance & Reachability Fusion (Level 2)                            |
| (e.g., DBSCAN, HDBSCAN)      | • Consensus Minimum Spanning Tree (MST) across multi-view reachability graphs         |
|                              | • Cross-View Mutual Reachability Modulation (Co-training style)                        |
|                              | • Joint Excess-of-Mass (EOM) Stability Optimization over multiple condensed trees    |
+------------------------------+---------------------------------------------------------------------------------------+
```

---

## 4. Architectural Roadmap for Multi-View HDBSCAN (MV-HDBSCAN)

HDBSCAN consists of 5 core sequential stages:
$$\text{Core Distances } d_{\text{core}} \longrightarrow \text{Mutual Reachability Graph } G \longrightarrow \text{Extended MST } \longrightarrow \text{Condensed Tree } \longrightarrow \text{EOM Flat Extraction}$$

Below are the 3 viable architectural pathways for designing **MV-HDBSCAN**:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    MV-HDBSCAN ARCHITECTURAL PATHWAYS                              |
+───────────────────────────────────────────────────────────────────────────────────────────────────+

 PATHWAY A: INTERMEDIATE DENSITY & REACHABILITY FUSION (Recommended Primary Architecture)
 ┌──────────┐
 │ View 1   │──> d_core^(1) ──┐
 └──────────┘                 │
 ┌──────────┐                 ├──> Fused d_core* & Consensus Reachability d_mreach* ──> MST ──> Tree ──> EOM
 │ View 2   │──> d_core^(2) ──┘
 └──────────┘

 PATHWAY B: CO-TRAINING DENSITY / MST ITERATION
 ┌──────────┐
 │ View 1   │──> d_mreach^(1) ──> MST^(1) ──┐ (Contraction / Density guidance)
 └──────────┘                               │
 ┌──────────┐                               ▼
 │ View 2   │──> d_mreach^(2) ──> MST^(2) ──┘ (Iterative mutual reachability modulation)

 PATHWAY C: MULTI-TREE JOINT STABILITY OPTIMIZATION (Late Hierarchical Fusion)
 ┌──────────┐
 │ View 1   │──> Tree^(1) with S^(1)(C) ──┐
 └──────────┘                             ├──> Joint Excess-of-Mass DP: max ∑ S^(v)(C) - γ Disagreement
 ┌──────────┐                             │
 │ View 2   │──> Tree^(2) with S^(2)(C) ──┘
 └──────────┘
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

### Detailed Pathway Breakdown:

1. **Pathway A: Multi-View Mutual Reachability Fusion (Intermediate Level)**:
   - Compute view-specific core distances $d_{\text{core}}^{(v)}(x_i)$ using view-specific metrics (e.g., Cosine for text, Euclidean for spatial/image features).
   - Normalize core distances and fuse them into a unified density scale:
     $$d_{\text{core}}^*(x_i) = \left( \sum_{v=1}^V w_v \left(\tilde{d}_{\text{core}}^{(v)}(x_i)\right)^p \right)^{1/p}$$
   - Construct the consensus mutual reachability distance:
     $$d_{\text{mreach}}^*(x_i, x_j) = \max\left\{ d_{\text{core}}^*(x_i), \, d_{\text{core}}^*(x_j), \, \bigoplus_{v=1}^V w_v \tilde{d}^{(v)}(x_i, x_j) \right\}$$
   - Build a single consensus Minimum Spanning Tree and run standard HDBSCAN condensation and EOM extraction.
   - *Advantages*: Highly efficient ($O(V \cdot d \cdot n^2)$), mathematically consistent with level set theory, preserves density hierarchy properties.

2. **Pathway B: Co-Trained Mutual Reachability (Model Coupling Level)**:
   - Build view-specific mutual reachability graphs $G^{(v)}$.
   - Use connected components / dense paths in view $A$ to attenuate edge weights in view $B$ (e.g., $d_{\text{mreach}}^{(B)\prime}(x_i, x_j) = d_{\text{mreach}}^{(B)}(x_i, x_j) \cdot (1 - \alpha \cdot \text{Sim}^{(A)}(x_i, x_j))$).
   - Recompute MSTs and iterate until reachability matrices stabilize.

3. **Pathway C: Joint Hierarchical Stability Optimization (Tree / Decision Level)**:
   - Construct condensed trees $T^{(1)}, \dots, T^{(V)}$ independently for all views.
   - Formulate a multi-view Excess-of-Mass optimization problem:
     $$\max_{\{\delta_i^{(v)}\}} \sum_{v=1}^V \sum_{i} \delta_i^{(v)} S\left(C_i^{(v)}\right) - \gamma \sum_{v \neq w} \text{Disagreement}\left(P^{(v)}, P^{(w)}\right)$$
   - Solved via tree dynamic programming or coordinate ascent over tree cuts.

---

## 5. Summary Decision Framework

| Scenario / Data Property | Best Fusion Level | Recommended Technique for MV-HDBSCAN |
| :--- | :--- | :--- |
| **Heterogeneous view metrics & scales** | **Intermediate (Level 2)** | View-specific normalized core distances + consensus $d_{\text{mreach}}^*$ |
| **Noisy / partially informative views** | **Intermediate + Auto-weights** | Adaptive view-weight learning ($\alpha_v$) based on variance or graph entropy |
| **Distinct cluster structures per view** | **Co-training / Iterative (Level 3)** | Iterative reachability modulation with cross-view confidence filtering |
| **Ensemble of decoupled pipelines** | **Late / Tree Fusion (Level 4)** | Multi-tree Excess-of-Mass joint stability extraction |
