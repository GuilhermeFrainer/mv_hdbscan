# Paper Summary: Co-regularized Multi-view Spectral Clustering (Kumar, Rai, & Daumé III, 2011)

- **Title**: *Co-regularized Multi-view Spectral Clustering*
- **Authors**: Abhishek Kumar, Piyush Rai, Hal Daumé III
- **Source**: *NIPS 2011*, Advances in Neural Information Processing Systems 24, Granada, Spain, pp. 1413–1421
- **Source File**: `papers/kumar2011_coreg_mvc.pdf`

---

## 1. Executive Summary

This paper extends single-view spectral clustering to multi-view data by formalizing a **joint optimization framework with co-regularization penalties**. It introduces two co-regularization paradigms:
1. **Pairwise Co-regularization**: Directly couples all pairs of view-specific spectral embeddings through pairwise similarity alignment.
2. **Centroid-Based Co-regularization**: Regularizes all view-specific embeddings toward a shared **consensus centroid embedding $U^*$**, reducing complexity from $O(m^2)$ to $O(m)$.

Both formulations solve alternating standard spectral clustering problems on **adaptively modified Laplacians**, providing rigorous convergence guarantees.

---

## 2. Baseline: Single-View Spectral Clustering

For view $v$ with similarity matrix $K^{(v)} \in \mathbb{R}^{n \times n}$ and normalized Laplacian $\mathcal{L}^{(v)} = (D^{(v)})^{-1/2} K^{(v)} (D^{(v)})^{-1/2}$:
$$\max_{U^{(v)} \in \mathbb{R}^{n \times k}} \text{tr}\left( (U^{(v)})^T \mathcal{L}^{(v)} U^{(v)} \right) \quad \text{s.t.} \quad (U^{(v)})^T U^{(v)} = I$$
where the $n$ rows of $U^{(v)}$ serve as the continuous relaxed cluster indicators for each sample.

---

## 3. How the Single-View Model is Converted to Multi-View

### 3.1 Measuring Disagreement Across Embeddings
Directly minimizing $\|U^{(v)} - U^{(w)}\|_F^2$ is flawed because eigenvectors in different views can span the same subspace up to arbitrary orthogonal rotations. 

Instead, the authors define disagreement over the **induced similarity matrices** in the embedding space:
$$D(U^{(v)}, U^{(w)}) = \left\| \frac{K_{U^{(v)}}}{\|K_{U^{(v)}}\|_F^2} - \frac{K_{U^{(w)}}}{\|K_{U^{(w)}}\|_F^2} \right\|_F^2$$
Using linear kernels $K_{U^{(v)}} = U^{(v)} (U^{(v)})^T$, we have $\|K_{U^{(v)}}\|_F^2 = \text{tr}((U^{(v)} (U^{(v)})^T)^2) = \text{tr}(I_k) = k$.

Expanding the Frobenius norm:
$$D(U^{(v)}, U^{(w)}) \propto - \text{tr}\left( U^{(v)} (U^{(v)})^T U^{(w)} (U^{(w)})^T \right)$$

Minimizing disagreement is mathematically equivalent to **maximizing $\text{tr}\left( U^{(v)} (U^{(v)})^T U^{(w)} (U^{(w)})^T \right)$**.

---

## 4. Formulation 1: Pairwise Co-Regularization

### 4.1 Joint Objective Function ($m$ Views)
$$\max_{U^{(1)}, \dots, U^{(m)}} \sum_{v=1}^m \text{tr}\left( (U^{(v)})^T \mathcal{L}^{(v)} U^{(v)} \right) + \lambda \sum_{\substack{1 \le v, w \le m \\ v \neq w}} \text{tr}\left( U^{(v)} (U^{(v)})^T U^{(w)} (U^{(w)})^T \right)$$
$$\text{subject to } (U^{(v)})^T U^{(v)} = I, \quad \forall 1 \le v \le m$$

### 4.2 Alternating Optimization Step
Fixing all views $w \neq v$, the subproblem for view $v$ becomes:
$$\max_{U^{(v)}} \text{tr}\left( (U^{(v)})^T \left[ \mathcal{L}^{(v)} + \lambda \sum_{w \neq v} U^{(w)} (U^{(w)})^T \right] U^{(v)} \right) \quad \text{s.t.} \quad (U^{(v)})^T U^{(v)} = I$$

**Crucial Insight**:
This subproblem is simply **single-view spectral clustering** performed on an **adaptively modified Laplacian**:
$$\tilde{\mathcal{L}}^{(v)} = \mathcal{L}^{(v)} + \lambda \sum_{w \neq v} U^{(w)} (U^{(w)})^T$$
The standard Laplacian $\mathcal{L}^{(v)}$ is augmented by the projection operator $\sum_{w \neq v} U^{(w)} (U^{(w)})^T$ that encodes the consensus structure discovered by all other views.

---

## 5. Formulation 2: Centroid-Based Co-Regularization

Instead of $O(m^2)$ pairwise couplings, introduces a consensus embedding $U^* \in \mathbb{R}^{n \times k}$ ($U^{*T} U^* = I$):

### 5.1 Joint Objective Function
$$\max_{U^{(1)}, \dots, U^{(m)}, U^*} \sum_{v=1}^m \text{tr}\left( (U^{(v)})^T \mathcal{L}^{(v)} U^{(v)} \right) + \sum_{v=1}^m \lambda_v \text{tr}\left( U^{(v)} (U^{(v)})^T U^* (U^*)^T \right)$$
$$\text{subject to } (U^{(v)})^T U^{(v)} = I \; (\forall v), \quad U^{*T} U^* = I$$

### 5.2 Alternating Updates:
1. **Update View $v$ given $U^*$**:
   $$\max_{U^{(v)}} \text{tr}\left( (U^{(v)})^T \left[ \mathcal{L}^{(v)} + \lambda_v U^* (U^*)^T \right] U^{(v)} \right) \implies \text{Top-}k \text{ eigenvectors of } \mathcal{L}^{(v)} + \lambda_v U^* (U^*)^T$$
2. **Update Consensus $U^*$ given all $\{U^{(v)}\}$**:
   $$\max_{U^*} \text{tr}\left( U^{*T} \left[ \sum_{v=1}^m \lambda_v U^{(v)} (U^{(v)})^T \right] U^* \right) \implies \text{Top-}k \text{ eigenvectors of } \sum_{v=1}^m \lambda_v U^{(v)} (U^{(v)})^T$$

**Advantage**: The final clustering is obtained by running $k$-means directly on the consensus centroid $U^*$, eliminating ad-hoc view concatenations.

---

## 6. Theoretical & Practical Highlights

1. **Convergence**: The joint objective is bounded from above and monotonically increases at each alternating step, guaranteeing convergence to a local optimum (typically in $< 10$ iterations).
2. **Hyperparameter Stability**: Empirical evaluations on synthetic, Reuters multilingual, UCI digits, and Caltech-101 datasets show strong performance over a wide interval of $\lambda \in [0.01, 0.05]$.
3. **Handling View Quality / Noise**: View-specific weights $\lambda_v$ allow down-weighting noisy or uninformative views.

---

## 7. Key Takeaways for MV-HDBSCAN

The co-regularization framework provides three powerful mathematical patterns that can be adapted for Multi-View HDBSCAN:

| Pattern in Kumar et al. (2011) | Analogue in Multi-View HDBSCAN |
| :--- | :--- |
| **Adaptively Modified Graph Laplacian** $\mathcal{L}^{(v)} + \lambda U^* U^{*T}$ | **Adaptively Modified Mutual Reachability Graph**:<br>$d_{mreach}^{(v)*}(x_p, x_q) = d_{mreach}^{(v)}(x_p, x_q) \cdot \left( 1 - \alpha \, \mathbb{I}_{consensus}(x_p \sim x_q) \right)$ |
| **Consensus Centroid Matrix $U^*$** | **Consensus Mutual Reachability Matrix / Minimum Spanning Tree**:<br>Constructing a consensus MST $T^*$ that optimizes joint edge stability across all views. |
| **Joint Multi-View Objective** | **Joint Excess of Mass (EOM) Optimization**:<br>$\max_{\{\delta_i\}} \sum_v S^{(v)}(C_i) - \gamma \, \text{Disagreement}(P^{(1)}, \dots, P^{(m)})$ |
