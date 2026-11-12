"""Unit tests for MultiViewHDBSCAN and FeatureStackingHDBSCAN."""

import pytest
import numpy as np
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score

from src.mv_hdbscan import MultiViewHDBSCAN
from src.stacking import FeatureStackingHDBSCAN


def generate_synthetic_multiview_data(n_samples: int = 150, random_state: int = 42):
    """Generate 2 complementary views for 3 distinct clusters."""
    rng = np.random.default_rng(random_state)
    n = n_samples // 3

    # View 1: Clusters 0 and 1 touch along x, Cluster 2 is separate
    x0_v1 = rng.normal(loc=[-1.0, 0.0], scale=[0.3, 0.1], size=(n, 2))
    x1_v1 = rng.normal(loc=[0.0, 0.0], scale=[0.3, 0.1], size=(n, 2))
    x2_v1 = rng.normal(loc=[2.5, 1.5], scale=[0.2, 0.2], size=(n, 2))
    X1 = np.vstack([x0_v1, x1_v1, x2_v1])

    # View 2: Clusters 1 and 2 touch along x, Cluster 0 is separate
    x0_v2 = rng.normal(loc=[2.5, 1.5], scale=[0.2, 0.2], size=(n, 2))
    x1_v2 = rng.normal(loc=[-1.0, 0.0], scale=[0.3, 0.1], size=(n, 2))
    x2_v2 = rng.normal(loc=[0.0, 0.0], scale=[0.3, 0.1], size=(n, 2))
    X2 = np.vstack([x0_v2, x1_v2, x2_v2])

    y = np.repeat([0, 1, 2], n)
    return [X1, X2], y


class TestMultiViewHDBSCAN:
    def test_sklearn_estimator_conventions(self):
        """Test scikit-learn parameter cloning, get_params, set_params."""
        model = MultiViewHDBSCAN(min_cluster_size=10, min_samples=5, fusion="mean", p=3.0)
        params = model.get_params()
        assert params["min_cluster_size"] == 10
        assert params["min_samples"] == 5
        assert params["fusion"] == "mean"
        assert params["p"] == 3.0

        cloned = clone(model)
        assert cloned.min_cluster_size == 10
        assert cloned.fusion == "mean"

        model.set_params(fusion="max", min_cluster_size=15)
        assert model.fusion == "max"
        assert model.min_cluster_size == 15

    def test_fit_and_fit_predict(self):
        """Test fit, fit_predict, and populated attributes."""
        views, _ = generate_synthetic_multiview_data(n_samples=150)
        model = MultiViewHDBSCAN(min_cluster_size=10, min_samples=5, fusion="max")

        res = model.fit(views)
        assert res is model
        assert hasattr(model, "labels_")
        assert hasattr(model, "probabilities_")
        assert hasattr(model, "fused_reachability_")
        assert hasattr(model, "core_distances_")
        assert hasattr(model, "weights_")
        assert model.n_samples_in_ == 150
        assert model.n_features_in_ == [2, 2]
        assert len(model.labels_) == 150
        assert len(model.probabilities_) == 150
        assert model.fused_reachability_.shape == (150, 150)
        assert len(model.core_distances_) == 2

        # Check fit_predict matches fit
        labels_predict = MultiViewHDBSCAN(min_cluster_size=10, min_samples=5, fusion="max").fit_predict(views)
        np.testing.assert_array_equal(model.labels_, labels_predict)

    def test_input_shapes_and_formats(self):
        """Test 3D array inputs and 3-view datasets."""
        rng = np.random.default_rng(0)
        # 3D array: (3 views, 60 samples, 4 features)
        X_3d = rng.normal(size=(3, 60, 4))
        model = MultiViewHDBSCAN(min_cluster_size=5).fit(X_3d)
        assert model.n_samples_in_ == 60
        assert model.n_features_in_ == [4, 4, 4]
        assert len(model.labels_) == 60

    @pytest.mark.parametrize("fusion_mode", ["max", "mean", "lp", "l2_metric", "min"])
    def test_all_fusion_strategies(self, fusion_mode):
        """Test that all supported fusion strategies fit without error."""
        views, _ = generate_synthetic_multiview_data(n_samples=90)
        model = MultiViewHDBSCAN(min_cluster_size=5, fusion=fusion_mode, weights=[0.6, 0.4])
        labels = model.fit_predict(views)
        assert len(labels) == 90
        assert np.all(model.probabilities_ >= 0.0)
        assert np.all(model.probabilities_ <= 1.0)

    def test_custom_weights(self):
        """Test custom non-equal view weights."""
        views, _ = generate_synthetic_multiview_data(n_samples=90)
        model = MultiViewHDBSCAN(min_cluster_size=5, fusion="mean", weights=[3.0, 1.0])
        model.fit(views)
        np.testing.assert_allclose(model.weights_, [0.75, 0.25])

    def test_complementary_clustering_accuracy(self):
        """Test that MV-HDBSCAN achieves high ARI on complementary overlapping views."""
        views, y_true = generate_synthetic_multiview_data(n_samples=300, random_state=42)
        model = MultiViewHDBSCAN(min_cluster_size=15, min_samples=5, fusion="max")
        labels = model.fit_predict(views)
        ari = adjusted_rand_score(y_true, labels)
        assert ari > 0.95, f"Expected ARI > 0.95 on complementary benchmark, got {ari:.3f}"

    def test_invalid_inputs(self):
        """Test error handling for invalid input structures."""
        model = MultiViewHDBSCAN()

        # Mismatched sample sizes across views
        with pytest.raises(ValueError, match="differs from View 0"):
            model.fit([np.zeros((10, 2)), np.zeros((8, 2))])

        # Empty view list
        with pytest.raises(ValueError, match="contains no views"):
            model.fit([])

        # 1D view array
        with pytest.raises(ValueError, match="must be a 2D array"):
            model.fit([np.zeros(10), np.zeros(10)])

        # Single sample
        with pytest.raises(ValueError, match="at least 2 samples"):
            model.fit([np.zeros((1, 2)), np.zeros((1, 2))])

        # Wrong number of weights
        with pytest.raises(ValueError, match="Length of weights"):
            model.set_params(weights=[0.5, 0.3, 0.2])
            model.fit([np.zeros((10, 2)), np.zeros((10, 2))])

        # Negative weights
        with pytest.raises(ValueError, match="non-negative"):
            model.set_params(weights=[-0.5, 1.5])
            model.fit([np.zeros((10, 2)), np.zeros((10, 2))])

        # Invalid fusion method
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            model.set_params(weights=None, fusion="invalid_strategy")
            model.fit([np.zeros((10, 2)), np.zeros((10, 2))])

    def test_heterogeneous_metrics_cosine_euclidean(self):
        """Test MultiViewHDBSCAN with heterogeneous per-view metrics (cosine and euclidean)."""
        rng = np.random.default_rng(42)
        n_per_cluster = 40
        n_samples = n_per_cluster * 3
        y = np.repeat([0, 1, 2], n_per_cluster)

        # View 1: 32-dim normalized embeddings on hypersphere
        centers_v1 = np.zeros((3, 32))
        centers_v1[0, 0] = 1.0
        centers_v1[1, 1] = 1.0
        centers_v1[2, 2] = 1.0

        v1_list = []
        for c in range(3):
            noise = rng.normal(0, 0.15, size=(n_per_cluster, 32))
            pts = centers_v1[c] + noise
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            v1_list.append(pts)
        X1 = np.vstack(v1_list)

        # View 2: 2D Euclidean features
        centers_v2 = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 5.0]])
        v2_list = []
        for c in range(3):
            noise = rng.normal(0, 0.3, size=(n_per_cluster, 2))
            v2_list.append(centers_v2[c] + noise)
        X2 = np.vstack(v2_list)

        model = MultiViewHDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric=["cosine", "euclidean"],
            fusion="max",
            scale_views=True,
        )
        labels = model.fit_predict([X1, X2])

        assert model.metrics_ == ["cosine", "euclidean"]
        valid_idx = labels != -1
        assert np.sum(valid_idx) > 0.8 * n_samples
        ari = adjusted_rand_score(y[valid_idx], labels[valid_idx])
        assert ari > 0.85

    def test_metric_validation(self):
        """Test metric type and length validation."""
        views, _ = generate_synthetic_multiview_data(n_samples=60)

        # Mismatched sequence length
        model_mismatch = MultiViewHDBSCAN(metric=["cosine"])
        with pytest.raises(ValueError, match="Length of metric sequence"):
            model_mismatch.fit(views)

        model_mismatch3 = MultiViewHDBSCAN(metric=["cosine", "euclidean", "manhattan"])
        with pytest.raises(ValueError, match="Length of metric sequence"):
            model_mismatch3.fit(views)

        # Invalid type
        model_bad_type = MultiViewHDBSCAN(metric=123)
        with pytest.raises(TypeError, match=r"metric must be a str, callable"):
            model_bad_type.fit(views)

    def test_scale_invariance_heterogeneous_metrics(self):
        """Test that scale_views=True provides scale invariance when fusing disparate metrics."""
        rng = np.random.default_rng(42)
        n = 50
        # View 1: Cosine distances in [0, 1]
        X1 = rng.normal(size=(n, 16))
        X1 /= np.linalg.norm(X1, axis=1, keepdims=True)

        # View 2: Euclidean metadata
        X2_orig = rng.normal(loc=0, scale=1.0, size=(n, 4))
        # Scaled by 10,000x
        X2_scaled = X2_orig * 10000.0

        model_orig = MultiViewHDBSCAN(
            min_cluster_size=5,
            metric=["cosine", "euclidean"],
            scale_views=True,
            scale_percentile=95.0,
            fusion="mean",
        ).fit([X1, X2_orig])

        model_scaled = MultiViewHDBSCAN(
            min_cluster_size=5,
            metric=["cosine", "euclidean"],
            scale_views=True,
            scale_percentile=95.0,
            fusion="mean",
        ).fit([X1, X2_scaled])

        np.testing.assert_allclose(
            model_orig.fused_reachability_,
            model_scaled.fused_reachability_,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(model_orig.labels_, model_scaled.labels_)

    def test_estimator_cloning_with_metric_sequence(self):
        """Test scikit-learn clone with list of metrics."""
        model = MultiViewHDBSCAN(metric=["cosine", "cityblock"], min_cluster_size=12)
        cloned = clone(model)
        assert cloned.metric == ["cosine", "cityblock"]
        assert cloned.min_cluster_size == 12


class TestFeatureStackingHDBSCAN:
    def test_sklearn_estimator_conventions(self):
        """Test scikit-learn conventions for FeatureStackingHDBSCAN."""
        model = FeatureStackingHDBSCAN(min_cluster_size=8, standardize=False)
        params = model.get_params()
        assert params["min_cluster_size"] == 8
        assert params["standardize"] is False

        cloned = clone(model)
        assert cloned.min_cluster_size == 8
        assert cloned.standardize is False

    def test_fit_and_fit_predict(self):
        """Test fit and stacked features properties."""
        views, y_true = generate_synthetic_multiview_data(n_samples=120)
        model = FeatureStackingHDBSCAN(min_cluster_size=10, min_samples=5, standardize=True)
        res = model.fit(views)

        assert res is model
        assert hasattr(model, "labels_")
        assert hasattr(model, "probabilities_")
        assert hasattr(model, "stacked_features_")
        assert model.n_samples_in_ == 120
        assert model.n_features_in_ == [2, 2]
        assert model.n_stacked_features_ == 4
        assert model.stacked_features_.shape == (120, 4)

        labels_predict = FeatureStackingHDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(views)
        np.testing.assert_array_equal(model.labels_, labels_predict)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        model = FeatureStackingHDBSCAN()
        with pytest.raises(ValueError, match="differs from View 0"):
            model.fit([np.zeros((10, 2)), np.zeros((8, 2))])


# ---------------------------------------------------------------------------
# Equivalence with the hdbscan library (0.2.0)
# ---------------------------------------------------------------------------

import hdbscan
from pathlib import Path
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.metrics import adjusted_mutual_info_score

FIXTURE_V010 = Path(__file__).parent / "data" / "fused_reachability_v0_1_0.npz"
FUSION_MODES = ["max", "mean", "lp", "l2_metric", "min"]


def generate_blob_views(n_samples: int = 600, n_blobs: int = 8, random_state: int = 0):
    """Overlapping Gaussian blobs in 5-d plus a one-hot covariate view."""
    rng = np.random.default_rng(random_state)
    centers = rng.normal(scale=3.0, size=(n_blobs, 5))
    blob = rng.integers(0, n_blobs, size=n_samples)
    X1 = centers[blob] + rng.normal(scale=1.0, size=(n_samples, 5))
    X2 = np.eye(3)[rng.integers(0, 3, size=n_samples)]
    return [X1, X2]


def assert_same_partition(labels_a, labels_b):
    """Labels equal up to a permutation of the cluster ids (noise must match too)."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    np.testing.assert_array_equal(labels_a == -1, labels_b == -1)
    assert len(np.unique(labels_a)) == len(np.unique(labels_b))
    assert adjusted_mutual_info_score(labels_a, labels_b) == pytest.approx(1.0, abs=1e-12)


class TestHdbscanLibraryEquivalence:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("approx", [True, False])
    def test_stacking_equals_library_on_stacked_features(self, seed, approx):
        """FS is hdbscan.HDBSCAN(min_cluster_size=m) on stacked_features_, label for label."""
        views = generate_blob_views(random_state=seed)
        model = FeatureStackingHDBSCAN(min_cluster_size=10, approx_min_span_tree=approx).fit(views)
        reference = hdbscan.HDBSCAN(min_cluster_size=10, approx_min_span_tree=approx).fit(
            model.stacked_features_
        )
        assert len(set(model.labels_)) > 2
        np.testing.assert_array_equal(model.labels_, reference.labels_)
        np.testing.assert_array_equal(model.probabilities_, reference.probabilities_)

    def test_stacking_default_is_library_default(self):
        """With no approx_min_span_tree argument, FS uses the library's default (True)."""
        views = generate_blob_views(random_state=0)
        model = FeatureStackingHDBSCAN(min_cluster_size=10).fit(views)
        assert model.approx_min_span_tree is True
        assert model.hdbscan_.approx_min_span_tree is hdbscan.HDBSCAN().approx_min_span_tree
        assert model.hdbscan_.algorithm == hdbscan.HDBSCAN().algorithm
        assert model.hdbscan_.min_samples is None

    def test_stacking_min_samples_convention(self):
        """min_samples = k uses the k-th other point (library), not the (k-1)-th (sklearn)."""
        views = generate_blob_views(random_state=0)
        model = FeatureStackingHDBSCAN(
            min_cluster_size=10, min_samples=7, approx_min_span_tree=False
        ).fit(views)
        X = model.stacked_features_
        D = squareform_euclidean(X)
        core = np.sort(D, axis=1)[:, 7]  # column 0 is the point itself
        mreach = np.maximum(np.maximum(core[:, None], core[None, :]), D)
        np.fill_diagonal(mreach, 0.0)
        direct = SklearnHDBSCAN(
            min_cluster_size=10, min_samples=1, metric="precomputed", copy=True
        ).fit(mreach)
        assert_same_partition(model.labels_, direct.labels_)

    @pytest.mark.parametrize("fusion_mode", FUSION_MODES)
    def test_fused_reachability_unchanged_from_v010(self, fusion_mode):
        """fused_reachability_ is bit-for-bit what 0.1.0 produced on a fixed input."""
        ref = np.load(FIXTURE_V010)
        model = MultiViewHDBSCAN(
            min_cluster_size=8,
            min_samples=5,
            fusion=fusion_mode,
            weights=[0.6, 0.4],
            metric=["cosine", "euclidean"],
        ).fit([ref["X1"], ref["X2"]])
        np.testing.assert_array_equal(model.fused_reachability_, ref[f"fused_{fusion_mode}"])

    def test_fused_reachability_matches_equations(self):
        """Max fusion: per-view max(core_p, core_q, d), then the max over views."""
        views = generate_blob_views(n_samples=200, random_state=3)
        k = 6
        model = MultiViewHDBSCAN(min_cluster_size=10, min_samples=k, scale_views=False).fit(views)
        expected = np.zeros((200, 200))
        for X_v in views:
            D = squareform_euclidean(X_v)
            core = np.sort(D, axis=1)[:, k]  # k-th other point
            expected = np.maximum(expected, np.maximum(np.maximum(core[:, None], core[None, :]), D))
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(model.fused_reachability_, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("fusion_mode", FUSION_MODES)
    def test_hierarchy_is_built_on_fused_matrix(self, seed, fusion_mode):
        """MH equals a hierarchy built directly on fused_reachability_ (no second pass).

        Independent reference: sklearn's HDBSCAN on the precomputed matrix with
        min_samples=1, where the core distance is the distance to the point itself (0).
        """
        views = generate_blob_views(random_state=seed)
        model = MultiViewHDBSCAN(min_cluster_size=10, fusion=fusion_mode).fit(views)
        fused_before = model.fused_reachability_.copy()
        direct = SklearnHDBSCAN(
            min_cluster_size=10, min_samples=1, metric="precomputed", copy=True
        ).fit(model.fused_reachability_)
        assert len(set(model.labels_)) > 2
        assert_same_partition(model.labels_, direct.labels_)
        # The fit must not have altered the stored matrix.
        np.testing.assert_array_equal(model.fused_reachability_, fused_before)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_second_reachability_pass_is_a_noop(self, seed):
        """With min_samples=1 the library's own reachability step returns the matrix as is."""
        from hdbscan._hdbscan_reachability import mutual_reachability

        views = generate_blob_views(n_samples=300, random_state=seed)
        model = MultiViewHDBSCAN(min_cluster_size=10).fit(views)
        assert model.hdbscan_.min_samples == 1
        fused = model.fused_reachability_
        again = mutual_reachability(fused.copy(), 1, 1.0)
        off_diagonal = ~np.eye(fused.shape[0], dtype=bool)
        np.testing.assert_array_equal(again[off_diagonal], fused[off_diagonal])

    @pytest.mark.parametrize("seed", [1, 3, 4])
    def test_constant_second_view_equals_single_view_exact_tree(self, seed):
        """A constant second view adds nothing: MH equals HDBSCAN (exact tree) on view 1.

        The comparison must use the exact spanning tree; the library's default
        (approx_min_span_tree=True) can give a very different partition.

        Against ``algorithm="generic"`` (exact tree, Prim's on the dense matrix) the
        partitions are identical. Against the library's default algorithm with
        ``approx_min_span_tree=False`` (Boruvka) the tree is also exact and has the same
        total weight, but mutual reachability has many tied edges, and a point that
        bridges two clusters at exactly its core distance can fall on either side
        depending on the order in which the tree is built. On seeds 1 and 4 this moves
        one point out of 600, hence the tolerance: same noise, same number of clusters,
        and at most 0.5% of the points in a different cluster.
        """
        X1 = generate_blob_views(random_state=seed)[0]
        X2 = np.ones((X1.shape[0], 2))
        model = MultiViewHDBSCAN(min_cluster_size=10).fit([X1, X2])
        assert len(set(model.labels_)) > 2

        generic = hdbscan.HDBSCAN(min_cluster_size=10, algorithm="generic").fit(X1)
        assert_same_partition(model.labels_, generic.labels_)

        default_exact = hdbscan.HDBSCAN(min_cluster_size=10, approx_min_span_tree=False).fit(X1)
        np.testing.assert_array_equal(model.labels_ == -1, default_exact.labels_ == -1)
        assert len(np.unique(model.labels_)) == len(np.unique(default_exact.labels_))
        assert n_points_moved(model.labels_, default_exact.labels_) <= 0.005 * len(X1)


def n_points_moved(labels_a, labels_b):
    """Number of points whose label differs after optimally matching the cluster ids."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.cluster import contingency_matrix

    table = contingency_matrix(labels_a, labels_b)
    rows, cols = linear_sum_assignment(-table)
    return int(table.sum() - table[rows, cols].sum())


def squareform_euclidean(X):
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(X, metric="euclidean"))
