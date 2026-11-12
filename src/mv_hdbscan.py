"""Multi-View HDBSCAN (MV-HDBSCAN) Clustering Algorithm.

This module implements a scikit-learn compatible Multi-View HDBSCAN clustering
estimator based on mutual reachability graph fusion across multiple views.
"""

from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.spatial.distance import pdist, squareform
from hdbscan import HDBSCAN
from sklearn.base import BaseEstimator, ClusterMixin


class MultiViewHDBSCAN(BaseEstimator, ClusterMixin):
    """Multi-View Hierarchical Density-Based Spatial Clustering of Applications with Noise.

    Performs clustering across multiple feature representations (views) of the same
    underlying data objects by computing view-wise core distances, constructing view-wise
    mutual reachability graphs, and fusing them into a unified multi-view reachability matrix
    before extracting persistent hierarchical clusters via Excess of Mass (EoM).

    The cluster hierarchy is built on the fused matrix itself: the fused matrix is already
    a mutual reachability, and no further core distance or mutual reachability step is
    applied to it. The hierarchy and the cluster extraction are done by the ``hdbscan``
    library (``hdbscan.HDBSCAN`` on a precomputed matrix), which always builds an exact
    minimum spanning tree on this path.

    Parameters
    ----------
    min_cluster_size : int, default=5
        The minimum number of samples in a group for that group to be considered a cluster.
    min_samples : int or None, default=None
        The number of samples in a neighborhood for a point to be considered a core point.
        Follows the ``hdbscan`` library's convention: the point itself is not counted, so
        the core distance is the distance to the `min_samples`-th other point
        (``sklearn.cluster.HDBSCAN`` counts the point itself and is one neighbor short of
        this). It sets the per-view core distances only (for ``fusion="l2_metric"``, the
        core distances of the fused distance); it is not applied again to the fused matrix.
        If None, defaults to `min_cluster_size`.
    fusion : {"max", "mean", "lp", "l2_metric", "min"}, default="max"
        Strategy used to fuse mutual reachability matrices across views:
        - "max": Strict consensus (element-wise maximum). Two points are reachable only if
          mutually reachable across all views simultaneously.
        - "mean": Weighted average of view-wise mutual reachabilities.
        - "lp": Generalized weighted Lp-norm fusion: (sum_v w_v (M^(v))^p)^(1/p).
        - "l2_metric": Quadratic distance metric fusion D_fused = sqrt(sum_v w_v (D^(v))^2),
          followed by core distance and reachability re-estimation.
        - "min": Optimistic union (element-wise minimum).
    weights : sequence of float or None, default=None
        Per-view weights for fusion. If None, views are weighted equally.
        Weights are automatically normalized to sum to 1.0.
    p : float, default=2.0
        Power exponent used when `fusion="lp"`.
    metric : str, callable, or sequence of str/callable, default="euclidean"
        Metric used to compute pairwise distances in each view. Can be a single metric
        string/callable applied across all views, or a sequence specifying a distinct metric
        per view (e.g. ``["cosine", "euclidean"]``).
    scale_views : bool, default=True
        Whether to normalize each view's pairwise distance matrix by its `scale_percentile`
        so that heterogeneous views with different natural scales have commensurate reachabilities.
    scale_percentile : float, default=95.0
        Percentile of the distance matrix used for scale normalization when `scale_views=True`.
    cluster_selection_epsilon : float, default=0.0
        A distance threshold. Clusters below this value are not split.
    cluster_selection_method : {"eom", "leaf"}, default="eom"
        The method used to select clusters from the condensed tree.
    allow_single_cluster : bool, default=False
        Whether to allow the extraction of a single cluster.
    n_jobs : int or None, default=None
        Number of parallel jobs passed to the library as ``core_dist_n_jobs``. If None, the
        library's default is used.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset. Noisy samples are given the label -1.
    probabilities_ : np.ndarray of shape (n_samples,)
        Strength of cluster membership for each sample point (in [0.0, 1.0]).
    fused_reachability_ : np.ndarray of shape (n_samples, n_samples)
        The fused multi-view mutual reachability matrix.
    core_distances_ : list of np.ndarray
        List of core distance arrays for each view, each of shape (n_samples,).
    weights_ : np.ndarray of shape (n_views,)
        The normalized weights assigned to each view.
    metrics_ : list of str or callable
        The resolved metric used for each view.
    n_samples_in_ : int
        The number of samples in the fitted dataset.
    n_features_in_ : list of int
        Number of features per view in the fitted dataset.
    hdbscan_ : hdbscan.HDBSCAN
        The underlying fitted HDBSCAN instance (``metric="precomputed"``, ``min_samples=1``).
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        fusion: str = "max",
        weights: Optional[Sequence[float]] = None,
        p: float = 2.0,
        metric: Union[str, Callable, Sequence[Union[str, Callable]]] = "euclidean",
        scale_views: bool = True,
        scale_percentile: float = 95.0,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = False,
        n_jobs: Optional[int] = None,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.fusion = fusion
        self.weights = weights
        self.p = p
        self.metric = metric
        self.scale_views = scale_views
        self.scale_percentile = scale_percentile
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.n_jobs = n_jobs

    def _validate_and_format_views(self, X: Union[Sequence[np.ndarray], np.ndarray]) -> List[np.ndarray]:
        """Validate input views structure and shapes."""
        if not isinstance(X, (list, tuple, np.ndarray)):
            raise TypeError(f"X must be a list, tuple, or 3D array of views. Got {type(X)}.")

        if isinstance(X, np.ndarray) and X.ndim == 3:
            # Shape (n_views, n_samples, n_features)
            views = [np.asarray(X[v], dtype=np.float64) for v in range(X.shape[0])]
        else:
            views = [np.asarray(v, dtype=np.float64) for v in X]

        if len(views) == 0:
            raise ValueError("Input X contains no views.")

        n_samples = views[0].shape[0]
        if n_samples < 2:
            raise ValueError(f"Each view must have at least 2 samples, got {n_samples}.")

        for idx, v in enumerate(views):
            if v.ndim != 2:
                raise ValueError(f"View {idx} must be a 2D array of shape (n_samples, n_features), got shape {v.shape}.")
            if v.shape[0] != n_samples:
                raise ValueError(
                    f"View {idx} has {v.shape[0]} samples, which differs from View 0 with {n_samples} samples."
                )

        return views

    def _compute_view_reachability(
        self,
        X_v: np.ndarray,
        effective_min_samples: int,
        metric_v: Union[str, Callable] = "euclidean",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute pairwise distance, core distances, and mutual reachability for a single view."""
        n_samples = X_v.shape[0]
        k = min(max(1, effective_min_samples), n_samples - 1)

        D = squareform(pdist(X_v, metric=metric_v))
        # Ensure numerical non-negativity and zero diagonal (e.g. for cosine precision quirks)
        np.maximum(D, 0.0, out=D)
        np.fill_diagonal(D, 0.0)

        sorted_dists = np.sort(D, axis=1)
        core_dists = sorted_dists[:, k]

        if self.scale_views:
            scale_factor = float(np.percentile(D, self.scale_percentile))
            if scale_factor > 1e-12:
                D_norm = D / scale_factor
                core_norm = core_dists / scale_factor
            else:
                D_norm = D
                core_norm = core_dists
        else:
            D_norm = D
            core_norm = core_dists

        mreach = np.maximum(np.maximum(core_norm[:, None], core_norm[None, :]), D_norm)
        np.fill_diagonal(mreach, 0.0)

        return D_norm, core_norm, mreach

    def _fuse_reachability(
        self,
        view_mreaches: Sequence[np.ndarray],
        view_dists: Sequence[np.ndarray],
        weights: np.ndarray,
        effective_min_samples: int,
    ) -> np.ndarray:
        """Fuse view-wise mutual reachability matrices into a unified reachability matrix."""
        n_views = len(view_mreaches)
        method = self.fusion.strip().lower()

        if method == "max":
            fused = np.zeros_like(view_mreaches[0], dtype=np.float64)
            for v in range(n_views):
                fused = np.maximum(fused, view_mreaches[v])
            np.fill_diagonal(fused, 0.0)
            return fused

        elif method == "mean":
            fused = np.zeros_like(view_mreaches[0], dtype=np.float64)
            for v in range(n_views):
                fused += weights[v] * view_mreaches[v]
            np.fill_diagonal(fused, 0.0)
            return fused

        elif method == "lp":
            if self.p <= 0:
                raise ValueError("Parameter p must be strictly positive for 'lp' fusion.")
            fused_p = np.zeros_like(view_mreaches[0], dtype=np.float64)
            for v in range(n_views):
                fused_p += weights[v] * (view_mreaches[v] ** self.p)
            fused = fused_p ** (1.0 / self.p)
            np.fill_diagonal(fused, 0.0)
            return fused

        elif method == "l2_metric":
            D_fused_sq = np.zeros_like(view_dists[0], dtype=np.float64)
            for v in range(n_views):
                D_fused_sq += weights[v] * (view_dists[v] ** 2)
            D_fused = np.sqrt(np.maximum(D_fused_sq, 0.0))
            n_samples = D_fused.shape[0]
            k = min(max(1, effective_min_samples), n_samples - 1)
            core_fused = np.sort(D_fused, axis=1)[:, k]
            fused = np.maximum(np.maximum(core_fused[:, None], core_fused[None, :]), D_fused)
            np.fill_diagonal(fused, 0.0)
            return fused

        elif method == "min":
            fused = np.full_like(view_mreaches[0], np.inf, dtype=np.float64)
            for v in range(n_views):
                fused = np.minimum(fused, view_mreaches[v])
            np.fill_diagonal(fused, 0.0)
            return fused

        else:
            raise ValueError(
                f"Unknown fusion strategy '{self.fusion}'. "
                "Supported options: ['max', 'mean', 'lp', 'l2_metric', 'min']."
            )

    def fit(self, X: Union[Sequence[np.ndarray], np.ndarray], y: Optional[np.ndarray] = None) -> "MultiViewHDBSCAN":
        """Fit the Multi-View HDBSCAN clustering model.

        Parameters
        ----------
        X : sequence of np.ndarray of shapes (n_samples, n_features_v) or 3D np.ndarray
            Input dataset consisting of V distinct feature representations (views) for
            the same n_samples objects.
        y : Ignored
            Not used, present for API consistency with scikit-learn.

        Returns
        -------
        self : MultiViewHDBSCAN
            Fitted estimator instance.
        """
        views = self._validate_and_format_views(X)
        n_views = len(views)
        self.n_samples_in_ = views[0].shape[0]
        self.n_features_in_ = [v.shape[1] for v in views]

        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size}.")

        effective_min_samples = (
            self.min_samples if self.min_samples is not None else self.min_cluster_size
        )
        if effective_min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {effective_min_samples}.")

        # Handle and normalize view weights
        if self.weights is None:
            raw_weights = np.ones(n_views, dtype=np.float64)
        else:
            if len(self.weights) != n_views:
                raise ValueError(
                    f"Length of weights ({len(self.weights)}) does not match number of views ({n_views})."
                )
            raw_weights = np.asarray(self.weights, dtype=np.float64)
            if np.any(raw_weights < 0):
                raise ValueError("All weights must be non-negative.")
            if np.sum(raw_weights) <= 0:
                raise ValueError("Sum of weights must be strictly positive.")

        self.weights_ = raw_weights / np.sum(raw_weights)

        # Resolve and validate view metrics
        if isinstance(self.metric, (str, bytes)) or callable(self.metric):
            metrics = [self.metric] * n_views
        elif isinstance(self.metric, (list, tuple, np.ndarray)):
            if len(self.metric) != n_views:
                raise ValueError(
                    f"Length of metric sequence ({len(self.metric)}) does not match number of views ({n_views})."
                )
            metrics = list(self.metric)
        else:
            raise TypeError(
                f"metric must be a str, callable, or sequence of str/callable. Got {type(self.metric)}."
            )
        self.metrics_ = metrics

        # Compute view-wise distances and mutual reachability
        view_dists = []
        view_cores = []
        view_mreaches = []

        for v_idx, (X_v, m_v) in enumerate(zip(views, self.metrics_)):
            D_v, core_v, mr_v = self._compute_view_reachability(
                X_v, effective_min_samples, metric_v=m_v
            )
            view_dists.append(D_v)
            view_cores.append(core_v)
            view_mreaches.append(mr_v)

        self.core_distances_ = view_cores

        # Fuse mutual reachability matrices
        self.fused_reachability_ = self._fuse_reachability(
            view_mreaches, view_dists, self.weights_, effective_min_samples
        )

        # Build the hierarchy on the fused matrix itself. Every fusion mode returns a
        # matrix that is already a mutual reachability, but the library treats a
        # precomputed matrix as plain distances: it computes core distances on it and
        # applies max(core_p, core_q, d) once more. min_samples=1 makes that step a no-op:
        # the core distance it computes for a row is then one of the row's smallest
        # entries, which never exceeds any off-diagonal entry of that row (or, by
        # symmetry, of that column), so max(core_p, core_q, d) = d. The user-facing
        # min_samples has already been used above, for the per-view core distances; it
        # must not be passed here, or mutual reachability would be applied twice.
        hdbscan_kwargs = {}
        if self.n_jobs is not None:
            hdbscan_kwargs["core_dist_n_jobs"] = self.n_jobs
        self.hdbscan_ = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
            metric="precomputed",
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            **hdbscan_kwargs,
        )

        # Fit on a copy so that fused_reachability_ is left as computed.
        self.labels_ = self.hdbscan_.fit_predict(self.fused_reachability_.copy())
        self.probabilities_ = getattr(self.hdbscan_, "probabilities_", np.ones(self.n_samples_in_))

        return self

    def fit_predict(
        self, X: Union[Sequence[np.ndarray], np.ndarray], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit the Multi-View HDBSCAN model and return cluster labels.

        Parameters
        ----------
        X : sequence of np.ndarray or 3D np.ndarray
            Input dataset consisting of V distinct feature views.
        y : Ignored
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        labels_ : np.ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X, y)
        return self.labels_
