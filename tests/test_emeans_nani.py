"""Opt-in e-means checks for NANI, with the default solver preserved."""
import numpy as np
import pytest
from sklearn.cluster import KMeans

from mdance.cluster.nani import KmeansNANI


def test_default_solver_matches_sklearn():
    rng = np.random.RandomState(8)
    data = rng.normal(size=(100, 4))
    centers = data[[1, 7, 15]]
    nani = KmeansNANI(data, 3, metric='MSD', init_type='strat_all')
    labels, result, iterations = nani.kmeans_clustering(centers)
    reference = KMeans(3, init=centers, n_init=1, random_state=None).fit(data)
    np.testing.assert_array_equal(labels, reference.labels_)
    np.testing.assert_allclose(result, reference.cluster_centers_)
    assert iterations == reference.n_iter_


def test_emeans_needs_explicit_nani_seeds():
    nani = KmeansNANI(np.ones((10, 2)), 2, metric='MSD', init_type='random')
    with pytest.raises(ValueError, match="explicit initial centers"):
        nani.kmeans_clustering('random', algorithm='emeans')


def test_emeans_final_assignment_and_seed():
    pytest.importorskip("faiss.contrib.e_means")
    rng = np.random.RandomState(11)
    data = np.vstack([rng.normal(-2, .3, (200, 3)),
                      rng.normal(2, .3, (200, 3))])
    initial = np.array([data[0], data[-1]])
    nani = KmeansNANI(data, 2, metric='MSD', init_type='strat_all')
    first = nani.kmeans_clustering(initial, algorithm='emeans',
                                   emeans_epochs=3, emeans_seed=4)
    second = nani.kmeans_clustering(initial, algorithm='emeans',
                                    emeans_epochs=3, emeans_seed=4)
    labels, centers, epochs = first
    nearest = np.argmin(((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    np.testing.assert_array_equal(labels, nearest)
    np.testing.assert_array_equal(centers, second[1])
    assert epochs == 3


def test_emeans_rejects_fractional_epochs():
    nani = KmeansNANI(np.ones((10, 2)), 2, metric='MSD', init_type='strat_all')
    with pytest.raises(ValueError, match="positive integer"):
        nani.kmeans_clustering(np.ones((2, 2)), algorithm='emeans', emeans_epochs=2.5)


def test_emeans_rejects_complex_features():
    nani = KmeansNANI(np.ones((10, 2), dtype=complex), 2, metric='MSD', init_type='strat_all')
    with pytest.raises(ValueError, match="finite real"):
        nani.kmeans_clustering(np.ones((2, 2)), algorithm='emeans')
