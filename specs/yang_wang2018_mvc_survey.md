# Paper Summary: Multi-view Clustering: A Survey (Yang & Wang, 2018)

- **Title**: *Multi-view Clustering: A Survey*
- **Authors**: Yan Yang, Hao Wang
- **Source**: *Big Data Mining and Analytics*, Vol. 1, No. 2, June 2018, pp. 83–107
- **Source File**: `papers/yang_wang2018_mvc_survey.pdf`

---

## 1. Core Principles of Multi-View Clustering (MVC)

Multi-view clustering operates on data objects described by multiple distinct feature sets (views): $X = \{X^{(1)}, X^{(2)}, \dots, X^{(m)}\}$. MVC is governed by two fundamental principles:

1. **Complementary Principle**:
   - Each individual view provides specific, distinct characteristics of the data (e.g., an image described by LBP textures, SIFT descriptors, and HOG edge histograms).
   - No single view contains complete information; integrating multiple views provides a more comprehensive, discriminative description.
2. **Consensus Principle**:
   - Multiple views observe the same underlying data instances and should agree on the latent clustering structure.
   - **Theoretical Foundation (PAC Bound - Dasgupta et al., 2001)**:
     $$P(f^{(1)} \neq f^{(2)}) \ge \max\left\{P_{err}(f^{(1)}), P_{err}(f^{(2)})\right\}$$
     The probability of disagreement between two independent view-hypotheses bounds the error rate of either hypothesis. Thus, **minimizing inter-view disagreement / maximizing consensus directly minimizes the clustering error rate**.

---

## 2. Taxonomy of MVC Approaches (5 Main Categories)

```
                       Multi-View Clustering (MVC)
                                   │
   ┌──────────────┬────────────────┼──────────────┬────────────────┐
   ▼              ▼                ▼              ▼                ▼
Co-Training  Multi-Kernel     Multi-View      Multi-View       Multi-Task
  Style       Learning          Graph          Subspace        Multi-View
 (Iterative  (Kernel Fusion)  (Affinity/      (Latent Space/   (Cross-Task
  Bootstrap)                  Spectral)       NMF / Tensor)     Transfer)
                                   │              │
                    ┌──────────────┼────────┐     ├──────────────┐
                    ▼              ▼        ▼     ▼              ▼
                 Graph-         Network- Spectral- Subspace    NMF-
                 based           based    based   Learning     based
```

### 2.1 Co-Training Style Algorithms
- **Mechanism**: Bootstraps clustering across views by iteratively exchanging knowledge/pseudo-labels or using the spectral embedding of one view to constrain the similarity graph of another (e.g., Kumar & Daumé III, 2011).
- **Core Assumption**: Views are conditionally independent given the true cluster labels (or satisfy weaker expansion/difference assumptions).
- **Techniques**: Multi-view EM (co-EM), co-regularized kernel k-means, co-clustering across views.

### 2.2 Multi-Kernel Learning (MKL)
- **Mechanism**: Constructs distinct kernel matrices $K^{(v)}$ for each view (e.g., Gaussian, polynomial, linear) and learns an optimal linear or non-linear combination:
  $$K^* = \sum_{v=1}^m \alpha_v K^{(v)} \quad \text{or} \quad K^*(x_i, x_j) = \sum_v \alpha_v \exp\left(-\frac{\|x_i^{(v)} - x_j^{(v)}\|^2}{2\sigma_v^2}\right)$$
- **Optimization**: Jointly optimizes kernel weights $\{\alpha_v\}$ and cluster indicators (e.g., multiple kernel k-means with matrix-induced regularization or centered kernel alignment).

### 2.3 Multi-View Graph & Spectral Clustering
- **Graph-Based**: Constructs affinity/similarity graphs $S^{(v)}$ for each view, then learns a unified consensus graph $S^*$ (often auto-weighted or parameter-free via adaptive neighbors) to feed into graph cuts or spectral clustering.
- **Network-Based**: Handles many-to-many entity mappings and heterogeneous distributions across domains (Network of Networks).
- **Spectral-Based**: Combines graph Laplacians $L^{(v)} = I - D_v^{-1/2} W^{(v)} D_v^{-1/2}$ into a joint spectral embedding:
  $$\min_{F^T F = I, \, \alpha_v} \sum_{v=1}^m (\alpha_v)^r \text{tr}(F^T L^{(v)} F) \quad \text{s.t.} \quad \sum_v \alpha_v = 1, \; \alpha_v \ge 0$$
  or uses bipartite graphs / shared Markov chain transition matrices.

### 2.4 Multi-View Subspace Clustering
- **Subspace Learning**: Projects view-specific feature spaces into a shared low-dimensional latent subspace $Y$ via canonical correlation analysis (CCA), diversity-induced penalties (Hilbert-Schmidt Independence Criterion - HSIC), or tensor low-rank decomposition.
- **NMF-Based (Non-negative Matrix Factorization)**: Factorizes view data $X^{(v)} \approx W^{(v)} H^*$ toward a shared consensus coefficient matrix $H^*$, often regularized by manifold Laplacians:
  $$\min_{W^{(v)}, H^* \ge 0} \sum_{v=1}^m \left( \|X^{(v)} - W^{(v)} H^*\|_F^2 + \alpha \text{tr}(H^* L^{(v)} (H^*)^T) \right)$$

### 2.5 Multi-Task Multi-View Clustering ($M^2VC$)
- **Mechanism**: Couples multiple related clustering tasks across multiple views, transferring knowledge across tasks while regularizing intra-task view agreement.

---

## 3. Paradigms for Converting Single-View Models to Multi-View

The survey identifies four key stages (**opportune moments of fusion**) where a single-view clustering pipeline can be transformed into a multi-view model:

```
+-------------------------------------------------------------------------------+
|                        PARADIGMS OF MULTI-VIEW CONVERSION                     |
+-------------------------------------------------------------------------------+

 1. EARLY FUSION (Input / Data Level)
    [View 1] ──┐
    [View 2] ──┼──> [Unified Feature / Latent Representation] ──> [Single-View Model]
    [View m] ──┘

 2. INTERMEDIATE FUSION (Proximity / Graph / Kernel Level)
    [View 1] ──> [Affinity / Reachability G1] ──┐
    [View 2] ──> [Affinity / Reachability G2] ──┼──> [Consensus Graph G*] ──> [Clustering]
    [View m] ──> [Affinity / Reachability Gm] ──┘

 3. HYBRID / CO-REGULARIZATION FUSION (Objective / Optimization Level)
    [View 1 Model] <──── Co-Regularization / Constraint Exchange ────> [View 2 Model]
          │                                                                   │
          └─────────────────────> Joint Optimization <────────────────────────┘

 4. LATE FUSION (Partition / Ensemble Level)
    [View 1] ──> [Model 1] ──> [Partition 1] ──┐
    [View 2] ──> [Model 2] ──> [Partition 2] ──┼──> [Consensus / Co-Association] ──> Final
    [View m] ──> [Model m] ──> [Partition m] ──┘
```

### Breakdown of Conversion Mechanisms:

1. **Early Fusion (Data/Feature Level)**:
   - Concatenate features or learn a joint low-rank latent representation (e.g., PCA/CCA/Autoencoder) before running the single-view algorithm.
   - *Limitation*: Obscures view-specific geometry and metric scales.

2. **Intermediate Fusion (Proximity/Graph Level)**:
   - Compute pairwise relationships (distances, affinities, kernels, core distances) within each view independently, respecting each view's natural metric.
   - Fuse these intermediate representations using adaptive weights, min/max operators, or consensus graphs before constructing spanning trees or spectral partitions.

3. **Co-Regularized / Alternating Fusion (Model Level)**:
   - Run the single-view model on each view while penalizing disagreements between their internal representations (e.g., spectral embeddings, cluster indicator matrices, or hierarchy splits).

4. **Late Fusion (Ensemble / Decision Level)**:
   - Execute the single-view algorithm independently on each view to generate individual cluster trees or flat partitions.
   - Combine results using co-association matrices, hypergraph partitioning, or voting schemes.

---

## 4. Synthesis: Implications for Designing Multi-View HDBSCAN (MV-HDBSCAN)

HDBSCAN consists of 5 core algorithmic stages:
$$\text{Core Distances } d_{core} \longrightarrow \text{Mutual Reachability Graph } G \longrightarrow \text{Extended MST } \longrightarrow \text{Condensed Tree } \longrightarrow \text{EOM Flat Extraction}$$

Applying the survey's conversion paradigms directly to HDBSCAN yields concrete design avenues:

| Conversion Paradigm | Application to HDBSCAN | Technical Mechanism |
| :--- | :--- | :--- |
| **Intermediate Proximity Fusion** | **Multi-View Mutual Reachability** | 1. Compute view-specific core distances $d_{core}^{(v)}(x)$ using view-appropriate metrics.<br>2. Fuse core distances: $d_{core}^*(x) = \bigoplus_v w_v d_{core}^{(v)}(x)$ (e.g., generalized mean).<br>3. Fuse mutual reachability: $d_{mreach}^*(x_p, x_q) = \max\{d_{core}^*(x_p), d_{core}^*(x_q), \bigoplus_v w_v d^{(v)}(x_p, x_q)\}$. |
| **Consensus Graph / MST Fusion** | **Multi-Graph MST** | Build view-specific mutual reachability graphs $G^{(v)}$, construct a unified fused graph $G^* = \sum_v w_v G^{(v)}$, and compute a single consensus MST. |
| **Co-Training / Iterative Guidance** | **Cross-View MST / Density Transfer** | Use high-density cluster cores identified in view $A$ to modulate neighborhood density estimates or reachability distances in view $B$. |
| **Late / Hierarchical Tree Fusion** | **Condensed Tree Consensus** | Construct condensed trees $T^{(1)}, \dots, T^{(m)}$ independently, then solve a joint Excess-of-Mass optimization that balances individual view stability with cross-view partition agreement. |
