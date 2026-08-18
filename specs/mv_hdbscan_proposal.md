# Multi-View HDBSCAN (MV-HDBSCAN): Comprehensive Technical Proposal & Architectural Design

- **Document Type**: Algorithmic Design Specification & Architecture Proposal
- **Author**: Antigravity Assistant
- **Target Specification**: `specs/mv_hdbscan_proposal.md`
- **Related References**:
  - `papers/campello2013_hdbscan.pdf` – *Density-Based Clustering Based on Hierarchical Density Estimates*
  - `papers/yang_wang2018_mvc_survey.pdf` – *Multi-View Clustering: A Survey*
  - `papers/daume2011_cotraining.pdf` – *Co-Training for Multi-View Spectral Clustering*
  - `papers/kumar2011_coreg_mvc.pdf` – *Co-Regularized Multi-View Spectral Clustering*
  - `specs/single_to_multiview_framework.md` – *Single-to-Multi-View Conversion Taxonomy*

---

## 1. Executive Summary & Problem Formulation

The goal of **MV-HDBSCAN** is to extend HDBSCAN to multi-view data $\{X^{(1)}, \dots, X^{(V)}\}$, where each view represents the same $n$ data objects in different feature spaces with potentially distinct distance metrics $d^{(v)}(\cdot, \cdot)$.

Single-view HDBSCAN operates on a 5-stage pipeline:
$$\text{Core Distances } d_{\text{core}} \longrightarrow \text{Mutual Reachability Graph } G_{\text{mreach}} \longrightarrow \text{Extended MST } \longrightarrow \text{Condensed Tree } \longrightarrow \text{EOM Stability Flat Extraction}$$

A comprehensive multi-view extension must address **where** and **how** multi-view synergy is introduced across these stages while preserving HDBSCAN's hallmark strengths:
1. **Arbitrary cluster shapes and varying densities**.
2. **Robust noise identification** (non-core / spurious points).
3. **Interpretability** via the condensed hierarchy.
4. **Parameter parsimony** (minimal tuning beyond $m_{\text{pts}}$ and $m_{\text{clSize}}$).

This proposal details **four major architectural paradigms** (covering 9 concrete algorithmic approaches), establishes a mathematical formulation for each, evaluates their theoretical and computational trade-offs, and defines the modular implementation architecture for `src/`.

---

## 2. Taxonomy of MV-HDBSCAN Approaches

```
                                  MV-HDBSCAN PARADIGMS
                                           │
  ┌───────────────────────┬────────────────┴────────────────┬────────────────────────┐
  ▼                       ▼                                 ▼                        ▼
PARADIGM 1:             PARADIGM 2:                       PARADIGM 3:              PARADIGM 4:
EARLY FUSION            INTERMEDIATE FUSION               CO-TRAINING &            LATE / HIERARCHICAL
(Feature / Subspace)    (Proximity / Reachability)        CO-REGULARIZATION        TREE FUSION
  • Feature Concat        • Fused Core Distances            • Co-Trained Reachability • Multi-Tree Joint EOM
  • CCA / Multi-View      • Direct Graph Fusion             • Co-Regularized MST     • Co-Association Ensemble
    Subspace Embedding    • Auto-Weighted Reachability      • Spectral-Density       • Consensus Dendrogram
                          • Cross-Diffusion Reachability      Coupling                 Alignment
```

---

## 3. Detailed Algorithmic Paradigms

---

### Paradigm 1: Early Fusion (Input / Feature Level MV-HDBSCAN)

```
[View 1 (d1)] ──┐
[View 2 (d2)] ──┼──> [Dimensionality Alignment / Shared Subspace Z] ──> [Standard HDBSCAN] ──> Labels
[View V (dV)] ──┘
```

#### Approach 1.1: Standardized Feature Concatenation
- **Formulation**: Standardize each view matrix $X^{(v)}$ (e.g., z-score) and concatenate into $X^* = [X^{(1)}, \dots, X^{(V)}] \in \mathbb{R}^{n \times \sum d_v}$.
- **Execution**: Apply standard single-view HDBSCAN on $X^*$ using Euclidean distance.
- **Evaluation**: Serves as a baseline; severely degraded by heterogeneous metrics and high-dimensional noise.

#### Approach 1.2: Multi-View Subspace Learning (CCA / Autoencoder / Joint NMF)
- **Formulation**: Project all views into a shared low-dimensional latent representation $Z \in \mathbb{R}^{n \times d^*}$ via Canonical Correlation Analysis (CCA) or Joint NMF ($X^{(v)} \approx W^{(v)} H^*$, $Z = H^{*T}$).
- **Execution**: Run single-view HDBSCAN on the latent representation $Z$.
- **Evaluation**: Handles high dimensionality, but projects away fine-grained view-specific local density peaks.

---

### Paradigm 2: Intermediate Fusion (Proximity, Density & Reachability Level)
*⭐ Recommended Primary Engine: Preserves native view metrics while fusing density estimates.*

```
┌──────────┐
│ View 1   │──> d_core^(1), d^(1) ──┐
└──────────┘                        │
┌──────────┐                        ├──> Fused Core Distances d_core* ──> MST* ──> Condensed Tree ──> EOM
│ View 2   │──> d_core^(2), d^(2) ──┤    Consensus Reachability d_mreach*
└──────────┘                        │
┌──────────┐                        │
│ View V   │──> d_core^(V), d^(V) ──┘
└──────────┘
```

#### Approach 2.1: Multi-View Core Distance & Distance Fusion (Unified Reachability)
1. **View-Specific Metric Computation**:
   Compute pairwise distance matrices $D^{(v)}$ using view-appropriate metrics (e.g., Cosine for text, Euclidean for spatial data, Mahalanobis for correlated features).
2. **Distance Calibration & Normalization**:
   Normalize each distance matrix to a comparable scale:
   $$\tilde{D}^{(v)} = \frac{D^{(v)}}{\text{median}(D^{(v)})} \quad \text{or} \quad \tilde{D}^{(v)} = \frac{D^{(v)}}{\|D^{(v)}\|_F}$$
3. **View-Specific Core Distances**:
   Compute $d_{\text{core}}^{(v)}(x_i)$ as the distance from $x_i$ to its $m_{\text{pts}}^{(v)}$-th nearest neighbor in view $v$, then normalize: $\tilde{d}_{\text{core}}^{(v)}(x_i) = d_{\text{core}}^{(v)}(x_i) / \text{median}(D^{(v)})$.
4. **Core Distance Aggregation**:
   Combine core distances across views using a generalized power mean ($p$-mean):
   $$d_{\text{core}}^*(x_i) = \left( \sum_{v=1}^V w_v \left(\tilde{d}_{\text{core}}^{(v)}(x_i)\right)^p \right)^{1/p}$$
   - **Harmonic Mean ($p = -1$)**: Strongly rewards density in *any* view (optimistic multi-view core).
   - **Geometric Mean ($p \to 0$)**: Balanced multi-view density.
   - **Arithmetic Mean ($p = 1$)**: Standard weighted average.
   - **Maximum ($p \to \infty$)**: Strict multi-view density (point must be dense in *all* views).
   - **Minimum ($p \to -\infty$)**: Ultra-optimistic density (dense if dense in at least one view).
5. **Pairwise Distance Aggregation**:
   $$d^*(x_i, x_j) = \left( \sum_{v=1}^V w_v \left(\tilde{d}^{(v)}(x_i, x_j)\right)^p \right)^{1/p}$$
6. **Consensus Mutual Reachability Distance**:
   $$d_{\text{mreach}}^*(x_i, x_j) = \max\left\{ d_{\text{core}}^*(x_i), \, d_{\text{core}}^*(x_j), \, d^*(x_i, x_j) \right\}$$
7. **Consensus Extended MST & Cluster Extraction**:
   Build the Minimum Spanning Tree $T^*$ on $G_{\text{mreach}}^*$, augment with self-edges $d_{\text{core}}^*(x_i)$, condense the tree with $m_{\text{clSize}}$, and extract flat clusters via Excess of Mass (EOM).

#### Approach 2.2: Direct Multi-View Reachability Graph Fusion
1. Compute view-specific mutual reachability matrices independently:
   $$D_{\text{mreach}}^{(v)}(x_i, x_j) = \max\left\{ d_{\text{core}}^{(v)}(x_i), \, d_{\text{core}}^{(v)}(x_j), \, d^{(v)}(x_i, x_j) \right\}$$
2. Normalize and fuse reachability graphs directly:
   $$D_{\text{mreach}}^*(x_i, x_j) = \sum_{v=1}^V w_v \tilde{D}_{\text{mreach}}^{(v)}(x_i, x_j) \quad \text{or} \quad \max_{v} \tilde{D}_{\text{mreach}}^{(v)}(x_i, x_j)$$
3. Build MST directly from $D_{\text{mreach}}^*$.

#### Approach 2.3: Auto-Weighted Adaptive Reachability Fusion
Learn optimal view weights $\{w_v\}$ automatically by minimizing graph dispersion or cross-view entropy:
$$\min_{\{w_v\}, D_{\text{mreach}}^*} \sum_{v=1}^V w_v^r \left\| D_{\text{mreach}}^* - \tilde{D}_{\text{mreach}}^{(v)} \right\|_F^2 \quad \text{s.t.} \quad \sum_v w_v = 1, \; w_v \ge 0, \; r > 1$$
This automatically down-weights noisy, high-variance views without manual parameter tuning.

#### Approach 2.4: Cross-Diffusion Reachability Network Fusion
Convert view-specific reachability matrices to transition probability kernels $P^{(v)}_{ij} \propto \exp(- (d_{\text{mreach}}^{(v)})^2 / 2\sigma_v^2)$, apply Similarity Network Fusion (SNF) cross-diffusion across views for $T$ steps, convert diffused consensus probabilities back to reachability distances $d^* = \sqrt{-\log P^*}$, and compute the MST.

---

### Paradigm 3: Co-Training & Co-Regularized MV-HDBSCAN

```
┌──────────┐                                                ┌──────────┐
│ View 1   │──> d_mreach^(1) ──> MST^(1) ──┐                │ View 2   │──> d_mreach^(2) ──> MST^(2) ──┐
└──────────┘                               │ (Bootstrapping)└──────────┘                               │
     ▲                                     │                     ▲                                     │
     └──────── Re-weight Reachability <────┘                     └──────── Re-weight Reachability <────┘
```

#### Approach 3.1: Co-Trained Iterative Reachability Contraction
Adapted from the co-training spectral principle (*Kumar & Daumé III, 2011*):
1. **Initialize**: Compute initial reachability graphs $D_{\text{mreach}}^{(1)}, \dots, D_{\text{mreach}}^{(V)}$ and their respective MSTs and preliminary clusterings $P^{(1)}, \dots, P^{(V)}$.
2. **Iterative Step ($t = 1, \dots, T_{\max}$)**:
   - For view $v$, calculate the co-clustering affinity matrix from all other views:
     $$A_{-v}(x_i, x_j) = \frac{1}{V-1} \sum_{w \neq v} \mathbb{I}\left( P^{(w)}(x_i) == P^{(w)}(x_j) \right) \cdot \text{membership\_prob}^{(w)}(x_i, x_j)$$
   - Contract (lower) the reachability distance in view $v$ for pairs with high cross-view agreement:
     $$D_{\text{mreach}}^{(v)(t)}(x_i, x_j) = D_{\text{mreach}}^{(v)(t-1)}(x_i, x_j) \cdot \left( 1 - \alpha \cdot A_{-v}(x_i, x_j) \right)$$
   - Recompute MSTs and condensed trees for each view.
3. **Convergence**: Stop when cluster assignments stabilize (typically $2 \le T \le 5$).

#### Approach 3.2: Spectral-Density Coupled HDBSCAN
1. Apply Centroid Co-regularized Spectral Clustering (*Kumar et al., 2011*) to obtain consensus spectral eigenvectors $U^* \in \mathbb{R}^{n \times k}$.
2. Compute spectral affinity distance: $d_{\text{spec}}(x_i, x_j) = \|u_i^* - u_j^*\|_2$.
3. Modulate density reachability with spectral guidance:
   $$d_{\text{mreach}}^{\text{coupled}}(x_i, x_j) = d_{\text{mreach}}^*(x_i, x_j) \cdot \left(1 + \beta \cdot d_{\text{spec}}(x_i, x_j)\right)$$
4. Build MST and extract clusters via HDBSCAN.

---

### Paradigm 4: Late Fusion & Hierarchical Tree Fusion

```
┌──────────┐
│ View 1   │──> Tree^(1) with Stabilities S^(1)(C) ──┐
└──────────┘                                         │
┌──────────┐                                         ├──> Joint Excess-of-Mass Optimization ──> Consensus Flat Partition
│ View 2   │──> Tree^(2) with Stabilities S^(2)(C) ──┤    max ∑ S^(v)(C) - γ Disagreement
└──────────┘                                         │
┌──────────┐                                         │
│ View V   │──> Tree^(V) with Stabilities S^(V)(C) ──┘
└──────────┘
```

#### Approach 4.1: Multi-Tree Joint Excess of Mass (Joint EOM Optimization)
1. Construct independent condensed trees $T^{(1)}, \dots, T^{(V)}$ for all views.
2. For each view $v$, calculate cluster stabilities:
   $$S^{(v)}(C_i^{(v)}) = \sum_{x_j \in C_i^{(v)}} \left( \frac{1}{\varepsilon_{\min}(x_j, C_i^{(v)})} - \frac{1}{\varepsilon_{\max}(C_i^{(v)})} \right)$$
3. Formulate the joint multi-view partition selection problem:
   $$\max_{\{\delta^{(1)}, \dots, \delta^{(V)}\}} \sum_{v=1}^V \left( \sum_{i} \delta_i^{(v)} S^{(v)}\left(C_i^{(v)}\right) \right) - \gamma \sum_{v \neq w} \text{HammingDisagreement}\left( P^{(v)}\left(\delta^{(v)}\right), P^{(w)}\left(\delta^{(w)}\right) \right)$$
   $$\text{subject to } \sum_{j \in I_h^{(v)}} \delta_j^{(v)} = 1 \quad \forall h \in L^{(v)}, \; \delta_i^{(v)} \in \{0, 1\}$$
4. **Solver**: Solved via alternating dynamic programming across tree cuts until mutual stability-consensus equilibrium is reached.

#### Approach 4.2: Multi-View Co-Association Consensus HDBSCAN (Ensemble)
1. Run single-view HDBSCAN independently on all $V$ views, extracting cluster assignments $y^{(v)}$ and point membership probabilities $p^{(v)}$.
2. Build an $n \times n$ weighted co-association matrix $C$:
   $$C_{ij} = \frac{1}{V} \sum_{v=1}^V \mathbb{I}\left( y_i^{(v)} == y_j^{(v)} \land y_i^{(v)} \neq -1 \right) \cdot \sqrt{p_i^{(v)} p_j^{(v)}}$$
3. Convert $C$ to a consensus distance matrix $D_{\text{consensus}} = 1 - C$.
4. Run HDBSCAN / Single-Linkage on $D_{\text{consensus}}$ to extract the final multi-view consensus partition.

---

## 4. Comprehensive Comparison Matrix of Approaches

| Approach | Theoretical Foundation | Metric Heterogeneity | Noise Robustness | Computational Complexity | Hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1.1 Feature Concat** | Heuristic | ❌ Poor | ⚠️ Low | $O(d \cdot n^2)$ | None |
| **1.2 Subspace/NMF** | Shared latent space | ⚠️ Moderate | ⚠️ Moderate | $O(V \cdot d \cdot n^2 + n^3)$ | Subspace dim $d^*$ |
| **2.1 Fused Reachability (Primary)** | **Level set & metric theory** | ✅ **Excellent** | ✅ **High** | **$O(V \cdot d \cdot n^2)$** | Fusion power $p$, view weights $w_v$ |
| **2.2 Direct Graph Fusion** | Proximity merging | ✅ Good | ✅ High | $O(V \cdot n^2)$ | View weights $w_v$ |
| **2.3 Auto-Weighted Reachability** | Entropy / dispersion min | ✅ Excellent | ✅ **Very High** | $O(V \cdot n^2 + \text{iter})$ | Exponent $r$ |
| **2.4 Cross-Diffusion SNF** | Markov random walks | ✅ Good | ✅ High | $O(V \cdot n^2 \cdot T)$ | Kernel $\sigma$, steps $T$ |
| **3.1 Co-Trained Reachability** | Iterative bootstrapping | ✅ Excellent | ✅ High | $O(T_{\max} \cdot V \cdot n^2)$ | Contraction rate $\alpha$, iter $T$ |
| **3.2 Spectral-Density Coupled** | Spectral-Density synergy | ✅ Good | ✅ High | $O(V \cdot n^2 + k n^2)$ | Trade-off $\beta$ |
| **4.1 Multi-Tree Joint EOM** | Joint cluster stability | ✅ Excellent | ✅ High | $O(V \cdot n^2 + V \cdot \kappa)$ | Agreement trade-off $\gamma$ |
| **4.2 Co-Association Ensemble** | Consensus clustering | ✅ Excellent | ✅ High | $O(V \cdot n^2)$ | None |

---

## 5. Proposed Software Architecture for `src/`

We adopt a modular, scikit-learn compatible design structured as follows:

```
src/
├── __init__.py
├── mv_hdbscan/
│   ├── __init__.py
│   ├── base.py                   # Base estimators and validation utilities
│   ├── core_distance.py          # View-specific & fused core distance estimators
│   ├── reachability.py           # Multi-view reachability matrices & fusion operators
│   ├── mst.py                    # Minimum Spanning Tree algorithms (Prim / Fast Borůvka)
│   ├── condensed_tree.py         # Hierarchy condensation and cluster tree representation
│   ├── eom.py                    # Excess of Mass single-tree & multi-tree DP solvers
│   ├── cotraining.py             # Iterative co-trained reachability contraction
│   ├── ensemble.py               # Multi-view co-association ensemble engine
│   └── estimator.py              # Main MVHDBSCAN estimator (scikit-learn compatible)
└── utils/
    ├── __init__.py
    ├── metrics.py                # Distance functions & normalization utilities
    └── synthetic_data.py         # Multi-view synthetic dataset generators
```

### Key Class Definition: `MVHDBSCAN`

```python
class MVHDBSCAN(BaseEstimator, ClusterMixin):
    """Multi-View Hierarchical Density-Based Spatial Clustering of Applications with Noise.
    
    Parameters
    ----------
    min_cluster_size : int, default=5
        The minimum number of samples in a group for that group to be considered a cluster.
    min_samples : int or list of int, default=None
        The number of samples in a neighborhood for a point to be considered a core point.
        Can be specified per-view as a list. Defaults to min_cluster_size.
    fusion_strategy : {'core_fusion', 'graph_fusion', 'auto_weighted', 'co_training', 'joint_eom', 'ensemble'}, default='core_fusion'
        The multi-view fusion strategy to employ.
    power_p : float, default=-1.0
        The generalized mean power for core distance fusion (-1 = Harmonic, 0 = Geometric, 1 = Arithmetic).
    view_weights : array-like of shape (n_views,), default=None
        Relative weight for each view. If None, equal weights are assigned.
    metric : str or list of str, default='euclidean'
        Distance metric to use for each view.
    """
```

---

## 6. Development & Validation Roadmap

1. **Milestone 1: Core Mathematical Primitives**:
   - Implement `core_distance.py`, `metrics.py`, and `reachability.py`.
   - Implement Prim's algorithm for MST construction and `condensed_tree.py`.
   - Implement Excess of Mass (EOM) dynamic programming in `eom.py`.
2. **Milestone 2: Multi-View Fusion Strategies**:
   - Implement intermediate fusion (harmonic/geometric/arithmetic core distance aggregation).
   - Implement auto-weighted adaptive graph fusion.
   - Implement co-trained reachability contraction and multi-tree joint EOM.
3. **Milestone 3: Comprehensive Test Suite & Benchmark Validation**:
   - Unit tests covering edge cases, metric invariances, and noise handling.
   - Benchmarking against single-view HDBSCAN, feature concatenation, and multi-view spectral baselines on synthetic and real datasets (e.g., UCI Digits, Reuters Multilingual, BBC).
