# Paper Summary: Co-training for Multi-view Spectral Clustering (Kumar & Daumé III, 2011)

- **Title**: *A Co-training Approach for Multi-view Spectral Clustering*
- **Authors**: Abhishek Kumar, Hal Daumé III
- **Source**: *ICML 2011*, Proceedings of the 28th International Conference on Machine Learning, Bellevue, WA, USA, pp. 393–400
- **Source File**: `papers/daume2011_cotraining.pdf`

---

## 1. Executive Summary

This paper adapts the semi-supervised **co-training** paradigm to **unsupervised spectral clustering**. It establishes an iterative bootstrapping mechanism where the spectral embedding (discriminative eigenvectors) derived from one view is used to project and modify the similarity graph of another view. The algorithm is completely **hyperparameter-free** and naturally yields low-rank Laplacians that reduce computational complexity.

---

## 2. Baseline: Single-View Spectral Clustering

Given a single view with data matrix $X \in \mathbb{R}^{n \times d}$:
1. Construct an $n \times n$ positive semi-definite similarity matrix $K$ (e.g., Gaussian RBF kernel $K_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$).
2. Compute the normalized graph Laplacian:
   $$\mathcal{L} = D^{-1/2} K D^{-1/2}, \quad D_{ii} = \sum_j K_{ij}$$
3. Compute the top-$k$ eigenvectors of $\mathcal{L}$ to form the embedding matrix $U \in \mathbb{R}^{n \times k}$ solving:
   $$\max_{U \in \mathbb{R}^{n \times k}} \text{tr}(U^T \mathcal{L} U) \quad \text{s.t.} \quad U^T U = I$$
4. Row-normalize $U \to V$ and run standard $k$-means on the rows of $V$ to assign cluster memberships.

---

## 3. How the Single-View Model is Converted to Multi-View

### 3.1 The Conceptual Bridge: Eigenvectors as "Soft Labels"
In semi-supervised co-training (Blum & Mitchell, 1998), two classifiers iteratively predict discrete labels on unlabeled data to enrich each other's training sets. In unsupervised clustering:
- There are no initial ground-truth labels.
- However, the top-$k$ eigenvectors $U$ of the graph Laplacian contain the essential **discriminative information** separating clusters (ignoring intra-cluster noise).
- Therefore, $U^{(v)}$ acts as a continuous, geometric "labeling" from view $v$ that can guide and filter the graph topology in view $w$.

### 3.2 Subspace Projection Mechanism
Rather than making hard cluster assignments and heuristically adjusting graph edge weights, the authors introduce an algebraic projection:
1. Each column $k_i$ of similarity matrix $K$ is an $n$-dimensional vector of similarities from point $i$ to all other points.
2. Project $K$ onto the subspace spanned by the top-$k$ discriminative eigenvectors $U$ of the *other* view:
   $$\tilde{S} = U U^T K$$
3. Symmetrize the resulting matrix:
   $$S = \text{sym}(U U^T K) = \frac{1}{2}\left( U U^T K + K U U^T \right)$$

### 3.3 Geometric Interpretation: "Gluing" Clusters
When $K_2$ is projected onto $U_1 U_1^T$:
- Within-cluster edge weights in View 2 are averaged out across all pairs identified as belonging to the same cluster in View 1.
- Across-cluster edge weights are attenuated and smoothed.
- This creates an iterative "gluing" effect where points agreed upon by View 1 are pulled closer in View 2's affinity graph, making cross-view consensus emerge naturally in subsequent iterations.

---

## 4. The Complete Multi-View Algorithm

```
Algorithm: Co-trained Multi-View Spectral Clustering
Input: Similarity matrices K_1, K_2 for two views; number of clusters k
Output: Cluster assignments for all n points

1. Initialize Laplacians: L_v = D_v^{-1/2} K_v D_v^{-1/2} for v ∈ {1, 2}
2. Initialize eigenvectors U_v^(0) = argmax tr(U^T L_v U) s.t. U^T U = I
3. For iteration t = 1 to max_iter (typically converges in < 5 iterations):
     a. S_1 = sym( U_2^(t-1) (U_2^(t-1))^T K_1 )
     b. S_2 = sym( U_1^(t-1) (U_1^(t-1))^T K_2 )
     c. Form new Laplacians L_1 from S_1, and L_2 from S_2
     d. Solve for new top-k eigenvectors U_1^(t) and U_2^(t)
4. Row-normalize U_1^(t) and U_2^(t)
5. Form V = [U_1, U_2] (concatenation) or select the more informative view
6. Run k-means on rows of V
```

### Extension to $m > 2$ Views:
For $m$ views, project each view's similarity matrix onto the union of subspaces of all other views:
$$S_v = \text{sym}\left( \left( \sum_{w \neq v} U_w U_w^T \right) K_v \right)$$

---

## 5. Computational and Practical Advantages

1. **Parameter-Free**: Does not introduce any trade-off weights, regularization penalties, or threshold hyperparameters.
2. **Low-Rank Efficiency**: The projected matrix $U U^T K$ has rank at most $k$. Its symmetrized form $S$ has rank at most $2k$. This allows large-scale randomized SVD algorithms to solve eigen-decompositions in $O(k n^2)$ or $O(k^2 n)$ rather than $O(n^3)$.
3. **Fast Empirical Convergence**: Major performance gains occur within the first 1–4 iterations.

---

## 6. Key Takeaways for MV-HDBSCAN

1. **Subspace Guidance on Proximity/Density**:
   - In HDBSCAN, can the mutual reachability graph $G^{(v)}$ or core distances $d_{core}^{(v)}$ of one view be modulated/projected using cluster memberships or MST branch structure from other views?
2. **Cluster "Gluing" via MST Paths**:
   - Similar to how $U U^T K$ averages intra-cluster similarities, HDBSCAN can contract or lower mutual reachability edge weights between points that are already densely connected in another view's MST.
3. **Iterative Density Refinement**:
   - High-density core objects in view $A$ can be used to lower effective core distances in view $B$, bootstrapping mutual reachability consensus.
