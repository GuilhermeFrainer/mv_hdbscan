# Paper Summary: HDBSCAN (Campello et al., 2013)

- **Title**: *Density-Based Clustering Based on Hierarchical Density Estimates*
- **Authors**: Ricardo J.G.B. Campello, Davoud Moulavi, Joerg Sander
- **Source**: *PAKDD 2013*, LNAI 7819, pp. 160–172
- **Source File**: `papers/campello2013_hdbscan.pdf`

---

## 1. Executive Overview

HDBSCAN (Hierarchical DBSCAN) extends DBSCAN into a hierarchical clustering method that estimates the density-contour tree of the underlying data distribution across all possible density thresholds. It solves core limitations of DBSCAN and OPTICS by:
1. Eliminating the single global distance threshold parameter ($\varepsilon$) while retaining the intuitive smoothing parameter ($m_{pts}$ / `min_samples`).
2. Providing a simplified, interpretable **condensed cluster tree** using a minimum cluster size parameter ($m_{clSize}$ / `min_cluster_size`).
3. Formulating an exact, globally optimal objective based on **Excess of Mass (Cluster Stability)** to automatically extract prominent flat clusters of varying densities from the hierarchy in $O(\kappa)$ time via dynamic programming.

---

## 2. DBSCAN* Formulation

Standard DBSCAN classifies non-core points falling within the $\varepsilon$-neighborhood of a core point as "border objects". Border objects can lead to non-deterministic cluster assignments and do not strictly belong to the statistical density level set $L_\lambda = \{x \mid f(x) \ge \lambda\}$.

HDBSCAN redefines DBSCAN as **DBSCAN\***:
- **Core Object**: Object $x_p \in X$ is a core object w.r.t. $\varepsilon$ and $m_{pts}$ iff $|N_\varepsilon(x_p)| \ge m_{pts}$, where $N_\varepsilon(x_p) = \{x \in X \mid d(x, x_p) \le \varepsilon\}$.
- **$\varepsilon$-Reachable**: Core objects $x_p, x_q$ are $\varepsilon$-reachable iff $x_p \in N_\varepsilon(x_q)$ and $x_q \in N_\varepsilon(x_p)$.
- **Density-Connected**: $x_p, x_q$ are density-connected if there exists a chain of mutually $\varepsilon$-reachable core objects.
- **Cluster**: Maximal subset of mutually density-connected core objects. All non-core objects are strictly labeled as **noise**.

---

## 3. Distance Metric Transformation & Mutual Reachability Graph

To compute DBSCAN\* partitions across all $\varepsilon \in [0, \infty)$ simultaneously:

### 3.1 Core Distance
For an object $x_p \in X$ and smoothing parameter $m_{pts}$:
$$d_{core}(x_p) = \text{distance from } x_p \text{ to its } m_{pts}\text{-th nearest neighbor (including } x_p \text{ itself)}$$

$x_p$ is an $\varepsilon$-core object for any $\varepsilon \ge d_{core}(x_p)$.

### 3.2 Mutual Reachability Distance
For any pair $x_p, x_q \in X$:
$$d_{mreach}(x_p, x_q) = \max\{d_{core}(x_p), d_{core}(x_q), d(x_p, x_q)\}$$

*Properties:*
- **Symmetric**: $d_{mreach}(x_p, x_q) = d_{mreach}(x_q, x_p)$.
- **Metric properties**: Preserves metric properties if $d(\cdot, \cdot)$ is a metric.
- **Density spreading**: Separates low-density (noise) points from high-density clusters by inflating distances involving sparse points.

### 3.3 Mutual Reachability Graph $G_{m_{pts}}$
A complete weighted graph where vertices are data objects $X$ and edge weights are $d_{mreach}(x_p, x_q)$.

**Proposition 1 (Equivalence)**:
The DBSCAN\* clustering for any $\varepsilon$ is identical to cutting the Single-Linkage dendrogram over the mutual reachability distance space at level $\varepsilon$, treating singletons with $d_{core}(x_p) > \varepsilon$ as noise.

---

## 4. Algorithmic Pipeline

### Step 1: Core Distance Computation
Compute $d_{core}(x_p)$ for all $x_p \in X$ using $m_{pts}$-NN queries ($O(n \log n)$ with spatial indexing or $O(d \cdot n^2)$ brute-force).

### Step 2: Minimum Spanning Tree (MST) Construction
Compute the Minimum Spanning Tree of $G_{m_{pts}}$:
- Construct the MST directly using Prim's algorithm in $O(d \cdot n^2)$ space/time, or Fast Dual-Tree Borůvka for Euclidean spaces.

### Step 3: Extended MST ($MST_{ext}$)
Augment the MST by adding a self-loop on each vertex $x_p$ with weight $d_{core}(x_p)$. This records the exact threshold below which $x_p$ is no longer a core object and drops out to noise.

### Step 4: Single-Linkage Hierarchy / Dendrogram
Iteratively remove edges from $MST_{ext}$ in descending order of weight. At each step:
- Current dendrogram scale: $\varepsilon = \text{weight of edge being removed}$.
- If a connected component still has $\ge 1$ edge, assign a cluster label; otherwise, mark as noise.

---

## 5. Hierarchy Simplification (Condensed Tree)

A full Single-Linkage tree contains $n-1$ splits, most of which are individual points falling out of clusters rather than true structural splits.

### Simplification Rules (with Minimum Cluster Size $m_{clSize} \ge 1$):
When an edge is removed from cluster $C$:
1. **Shrinkage (Noise dropout)**: If only one resulting component has $\ge m_{clSize}$ points and others have $< m_{clSize}$ points:
   - Spurious components ($< m_{clSize}$) are marked as noise dropping out at the current density $\lambda = 1/\varepsilon$.
   - The main component retains the identity/label of cluster $C$.
2. **True Cluster Split**: If two or more resulting components have $\ge m_{clSize}$ points:
   - Genuine child clusters are created, each receiving a new unique cluster label.
3. **Disappearance**: If all resulting components have $< m_{clSize}$ points:
   - The cluster vanishes completely, and all remaining points drop out as noise.

*Setting $m_{clSize} = m_{pts}$ yields a single-parameter algorithm.*

---

## 6. Cluster Stability & Optimal Flat Extraction (EOM)

To extract flat clusters without imposing an arbitrary global density threshold $\varepsilon$, HDBSCAN formalizes cluster selection as an optimization problem based on **Excess of Mass**.

### 6.1 Density Scale Definition
Let $\lambda = \frac{1}{\varepsilon}$. As $\varepsilon \to 0$, $\lambda \to \infty$.
- $\lambda_{min}(C_i)$: Density level at which cluster $C_i$ appears (birth).
- $\lambda_{max}(C_i)$: Density level at which cluster $C_i$ splits into sub-clusters or vanishes (death).
- $\lambda_{max}(x_j, C_i) = \min\{1/d_{core}(x_j), \lambda_{max}(C_i)\}$: Density level at which point $x_j$ drops out of $C_i$.

### 6.2 Relative Excess of Mass (Cluster Stability)
For a discrete dataset $X$:
$$S(C_i) = \sum_{x_j \in C_i} \left(\lambda_{max}(x_j, C_i) - \lambda_{min}(C_i)\right) = \sum_{x_j \in C_i} \left(\frac{1}{\varepsilon_{min}(x_j, C_i)} - \frac{1}{\varepsilon_{max}(C_i)}\right)$$

### 6.3 Optimization Formulation
Maximize the total stability of non-overlapping clusters across the condensed tree:
$$\max_{\delta_2, \dots, \delta_\kappa} J = \sum_{i=2}^\kappa \delta_i S(C_i)$$
$$\text{subject to } \delta_i \in \{0, 1\}, \quad \sum_{j \in I_h} \delta_j = 1 \quad \forall h \in L$$
where:
- $\kappa$: Number of nodes in the simplified condensed tree.
- $L$: Set of indices of leaf clusters.
- $I_h$: Set of indices on the path from leaf $C_h$ to root (excluding root).

### 6.4 Globally Optimal Dynamic Programming Solution (Algorithm 3)
Computed bottom-up from leaves to root in $O(\kappa)$ time:

1. **Leaf initialization**: For every leaf $C_h$, set $\hat{S}(C_h) = S(C_h)$ and $\delta_h = 1$.
2. **Bottom-up propagation**: For internal node $C_i$ with children $\{c_1, \dots, c_k\}$:
   $$\hat{S}(C_i) = \max\left(S(C_i), \sum_{c \in children(C_i)} \hat{S}(c)\right)$$
   - If $S(C_i) \ge \sum_{c} \hat{S}(c)$: Select cluster $C_i$ ($\delta_i = 1$) and deselect all clusters in its subtree ($\delta_{desc} = 0$).
   - If $S(C_i) < \sum_{c} \hat{S}(c)$: Propagate the children's sum $\hat{S}(C_i) = \sum \hat{S}(c)$ and set $\delta_i = 0$.
3. **Top-down extraction**: Traverse top-down from root to find the shallowest selected nodes ($\delta = 1$).

---

## 7. Computational Complexity

| Stage | Time Complexity | Space Complexity |
| :--- | :--- | :--- |
| Core Distances | $O(d \cdot n^2)$ or $O(d \cdot n \log n)$ with kd-tree | $O(n)$ |
| Mutual Reachability MST | $O(d \cdot n^2)$ (Prim's) or $O(d \cdot n \log n)$ (Dual-tree Borůvka) | $O(n)$ or $O(n^2)$ if full distance matrix |
| Hierarchy Condensation | $O(n \log n)$ | $O(n)$ |
| Optimal Flat Extraction | $O(\kappa)$ where $\kappa \ll n$ | $O(\kappa)$ |
| **Total Pipeline** | **$O(d \cdot n^2)$** (general metric) / **$O(d \cdot n \log n)$** (low-dim Euclidean) | **$O(n)$** |

---

## 8. Key Takeaways for Multi-View HDBSCAN (MV-HDBSCAN)

To extend HDBSCAN to multi-view settings with $V$ views $\{X^{(1)}, X^{(2)}, \dots, X^{(V)}\}$:
1. **Core Distance Fusion**: Can view-specific core distances $d_{core}^{(v)}(x_i)$ be fused (e.g., harmonic mean, weighted sum, min/max) to define cross-view density?
2. **Mutual Reachability Graph Integration**: How to aggregate or co-regularize edge weights $d_{mreach}^{(v)}$ across views?
3. **Cross-View MST / Graph Laplacians**: Constructing a unified consensus MST or co-regularized affinity graph across views.
4. **Hierarchical / Tree Stability Consensus**: Optimizing joint cluster stability across multiple condensed trees or co-training cluster assignments.
