"""Feature Stacking HDBSCAN Baseline Estimator.

This module implements a baseline clustering estimator that standardizes and
concatenates (stacks) all multi-view feature representations horizontally into a single
feature space before applying standard single-view HDBSCAN.
"""

from typing import List, Optional, Sequence, Union
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import HDBSCAN


class FeatureStackingHDBSCAN(BaseEstimator, ClusterMixin):
    """Feature Stacking HDBSCAN Baseline Estimator for Multi-View Data.

    Standardizes each view's feature space (to mitigate scale imbalances) and horizontally
    concatenates all views into a single unified feature matrix X_stacked = [X^(1), X^(2), ..., X^(V)]
    before fitting standard HDBSCAN.

    Parameters
    ----------
    min_cluster_size : int, default=5
        The minimum number of samples in a group for that group to be considered a cluster.
    min_samples : int or None, default=None
        The number of samples in a neighborhood for a point to be considered a core point.
        If None, defaults to `min_cluster_size`.
    standardize : bool, default=True
        Whether to standardize each view (zero mean, unit variance) prior to concatenation.
    metric : str, default="euclidean"
        The metric to use when calculating distance between instances in the stacked space.
    cluster_selection_epsilon : float, default=0.0
        A distance threshold. Clusters below this value are not split.
    cluster_selection_method : {"eom", "leaf"}, default="eom"
        The method used to select clusters from the condensed tree.
    allow_single_cluster : bool, default=False
        Whether to allow the extraction of a single cluster.
    n_jobs : int or None, default=None
        Number of parallel jobs to run for HDBSCAN tree extraction.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset. Noisy samples are given the label -1.
    probabilities_ : np.ndarray of shape (n_samples,)
        Strength of cluster membership for each sample point (in [0.0, 1.0]).
    stacked_features_ : np.ndarray of shape (n_samples, total_features)
        The concatenated (and optionally standardized) feature matrix.
    n_samples_in_ : int
        The number of samples in the fitted dataset.
    n_features_in_ : list of int
        Number of features per view in the fitted dataset.
    n_stacked_features_ : int
        Total number of features across all stacked views.
    hdbscan_ : sklearn.cluster.HDBSCAN
        The underlying fitted HDBSCAN instance.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        standardize: bool = True,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = False,
        n_jobs: Optional[int] = None,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.standardize = standardize
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.n_jobs = n_jobs

    def _validate_and_format_views(self, X: Union[Sequence[np.ndarray], np.ndarray]) -> List[np.ndarray]:
        """Validate input views structure and shapes."""
        if not isinstance(X, (list, tuple, np.ndarray)):
            raise TypeError(f"X must be a list, tuple, or 3D array of views. Got {type(X)}.")

        if isinstance(X, np.ndarray) and X.ndim == 3:
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

    def fit(self, X: Union[Sequence[np.ndarray], np.ndarray], y: Optional[np.ndarray] = None) -> "FeatureStackingHDBSCAN":
        """Fit the Feature Stacking HDBSCAN baseline model.

        Parameters
        ----------
        X : sequence of np.ndarray of shapes (n_samples, n_features_v) or 3D np.ndarray
            Input dataset consisting of V distinct feature views for the same n_samples objects.
        y : Ignored
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        self : FeatureStackingHDBSCAN
            Fitted estimator instance.
        """
        views = self._validate_and_format_views(X)
        self.n_samples_in_ = views[0].shape[0]
        self.n_features_in_ = [v.shape[1] for v in views]

        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size}.")

        effective_min_samples = (
            self.min_samples if self.min_samples is not None else self.min_cluster_size
        )
        if effective_min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {effective_min_samples}.")

        # Process and stack views
        processed_views = []
        for X_v in views:
            if self.standardize:
                mean = np.mean(X_v, axis=0)
                std = np.std(X_v, axis=0)
                std_safe = np.where(std < 1e-12, 1.0, std)
                processed_views.append((X_v - mean) / std_safe)
            else:
                processed_views.append(X_v)

        self.stacked_features_ = np.hstack(processed_views)
        self.n_stacked_features_ = self.stacked_features_.shape[1]

        # Fit HDBSCAN on stacked feature matrix
        self.hdbscan_ = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=effective_min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            copy=True,
            n_jobs=self.n_jobs,
        )

        self.labels_ = self.hdbscan_.fit_predict(self.stacked_features_)
        self.probabilities_ = getattr(self.hdbscan_, "probabilities_", np.ones(self.n_samples_in_))

        return self

    def fit_predict(
        self, X: Union[Sequence[np.ndarray], np.ndarray], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit the Feature Stacking HDBSCAN baseline and return cluster labels.

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
