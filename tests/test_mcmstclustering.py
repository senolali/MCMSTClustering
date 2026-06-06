"""
Test suite for MCMSTClustering.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=mcmstclustering
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from mcmstclustering import MCMSTClustering


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blobs():
    """Two well-separated Gaussian blobs, normalised to [0, 1]."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal([0.0, 0.0], 0.05, (100, 2)),
        rng.normal([1.0, 1.0], 0.05, (100, 2)),
    ])
    return MinMaxScaler().fit_transform(X)


@pytest.fixture
def moons():
    """Two-moon dataset, normalised."""
    X, _ = make_moons(n_samples=300, noise=0.07, random_state=1)
    return MinMaxScaler().fit_transform(X)


@pytest.fixture
def four_blobs():
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.4, random_state=2)
    return MinMaxScaler().fit_transform(X)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestBasicCorrectness:

    def test_two_blobs_finds_two_clusters(self, two_blobs):
        model = MCMSTClustering(N=4, r=0.08, n_micro=2).fit(two_blobs)
        assert model.n_clusters_ == 2

    def test_labels_shape(self, two_blobs):
        labels = MCMSTClustering(N=4, r=0.08, n_micro=2).fit_predict(two_blobs)
        assert labels.shape == (len(two_blobs),)

    def test_labels_dtype_integer(self, two_blobs):
        labels = MCMSTClustering(N=4, r=0.08, n_micro=2).fit_predict(two_blobs)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_noise_label_is_minus_one(self):
        # Very tight r → many noise points
        rng = np.random.default_rng(99)
        X = MinMaxScaler().fit_transform(rng.random((50, 2)))
        labels = MCMSTClustering(N=20, r=0.01, n_micro=10).fit_predict(X)
        assert set(labels).issubset(set([-1]) | set(range(100)))

    def test_moons_finds_two_clusters(self, moons):
        model = MCMSTClustering(N=4, r=0.06, n_micro=3).fit(moons)
        assert model.n_clusters_ == 2

    def test_four_blobs(self, four_blobs):
        model = MCMSTClustering(N=5, r=0.07, n_micro=3).fit(four_blobs)
        assert model.n_clusters_ == 4


# ---------------------------------------------------------------------------
# Attributes after fit
# ---------------------------------------------------------------------------

class TestFittedAttributes:

    def test_n_features_in(self, two_blobs):
        model = MCMSTClustering(N=4, r=0.08, n_micro=2).fit(two_blobs)
        assert model.n_features_in_ == 2

    def test_micro_clusters_is_list(self, two_blobs):
        model = MCMSTClustering(N=4, r=0.08, n_micro=2).fit(two_blobs)
        assert isinstance(model.micro_clusters_, list)

    def test_micro_cluster_centers_shape(self, two_blobs):
        model = MCMSTClustering(N=4, r=0.08, n_micro=2).fit(two_blobs)
        n_mc = len(model.micro_clusters_)
        assert model.micro_cluster_centers_.shape == (n_mc, 2)

    def test_labels_attribute_equals_fit_predict(self, two_blobs):
        model = MCMSTClustering(N=4, r=0.08, n_micro=2).fit(two_blobs)
        labels_fp = MCMSTClustering(N=4, r=0.08, n_micro=2).fit_predict(two_blobs)
        np.testing.assert_array_equal(model.labels_, labels_fp)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestParameterValidation:

    @pytest.mark.parametrize("bad_N", [0, -1, 0.5, "a"])
    def test_invalid_N_raises(self, two_blobs, bad_N):
        with pytest.raises((ValueError, TypeError)):
            MCMSTClustering(N=bad_N, r=0.05, n_micro=2).fit(two_blobs)

    @pytest.mark.parametrize("bad_r", [0, -0.1, "x"])
    def test_invalid_r_raises(self, two_blobs, bad_r):
        with pytest.raises((ValueError, TypeError)):
            MCMSTClustering(N=5, r=bad_r, n_micro=2).fit(two_blobs)

    @pytest.mark.parametrize("bad_n_micro", [0, -3, 1.5])
    def test_invalid_n_micro_raises(self, two_blobs, bad_n_micro):
        with pytest.raises((ValueError, TypeError)):
            MCMSTClustering(N=5, r=0.05, n_micro=bad_n_micro).fit(two_blobs)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_no_micro_clusters_warns(self):
        X = MinMaxScaler().fit_transform(
            np.random.default_rng(7).random((30, 2))
        )
        with pytest.warns(UserWarning, match="No micro-clusters"):
            model = MCMSTClustering(N=100, r=0.001, n_micro=5).fit(X)
        assert model.n_clusters_ == 0
        assert (model.labels_ == -1).all()

    def test_high_dimensional(self):
        rng = np.random.default_rng(10)
        X = np.vstack([
            rng.normal(0, 0.1, (60, 10)),
            rng.normal(3, 0.1, (60, 10)),
        ])
        X = MinMaxScaler().fit_transform(X)
        model = MCMSTClustering(N=4, r=0.1, n_micro=2).fit(X)
        assert model.n_clusters_ >= 1

    def test_single_cluster_possible(self):
        rng = np.random.default_rng(5)
        X = MinMaxScaler().fit_transform(rng.normal(0, 0.05, (80, 2)))
        model = MCMSTClustering(N=4, r=0.2, n_micro=2).fit(X)
        assert model.n_clusters_ >= 1

    def test_reproducibility(self, two_blobs):
        l1 = MCMSTClustering(N=4, r=0.08, n_micro=2).fit_predict(two_blobs)
        l2 = MCMSTClustering(N=4, r=0.08, n_micro=2).fit_predict(two_blobs)
        np.testing.assert_array_equal(l1, l2)

    def test_3d_data(self):
        rng = np.random.default_rng(20)
        X = np.vstack([
            rng.normal([0, 0, 0], 0.05, (80, 3)),
            rng.normal([1, 1, 1], 0.05, (80, 3)),
        ])
        X = MinMaxScaler().fit_transform(X)
        model = MCMSTClustering(N=4, r=0.1, n_micro=2).fit(X)
        assert model.n_clusters_ == 2


# ---------------------------------------------------------------------------
# Scikit-learn estimator checks (subset)
# ---------------------------------------------------------------------------

@parametrize_with_checks([MCMSTClustering()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
