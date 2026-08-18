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
