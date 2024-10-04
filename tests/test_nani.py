import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from mdance import data
from mdance.cluster.nani import KmeansNANI


@pytest.fixture(scope='module')
def traj_numpy():
    """
    Load the trajectory data.
    """
    return np.load(data.sim_traj_numpy)


@pytest.fixture(scope='module')
def blobs():
    """
    Generate blobs.
    """
    data, true_labels = make_blobs(n_samples=100, centers=7, n_features=2, random_state=0)
    return data, true_labels


def test_nani(traj_numpy):
    """
    Test the NANI algorithm.
    """
    n_clusters = 6
    mod = KmeansNANI(traj_numpy, n_clusters, 'MSD', N_atoms=50,
                     init_type='comp_sim', percentage=10)
    initiators = mod.initiate_kmeans()
    initiators = initiators[:n_clusters]
    kmeans = KMeans(n_clusters=n_clusters, init=initiators, n_init=1, 
                    random_state=None)
    kmeans.fit(traj_numpy)
    labels = kmeans.labels_
    sort_labels_by_size = np.argsort(np.bincount(labels))[::-1]
    labels = np.array([np.where(sort_labels_by_size == i)[0][0] for i in labels])
    expected_labels = np.loadtxt(data.labels_6, delimiter=',')
    expected_labels = expected_labels[:, 1]
    assert np.allclose(labels, expected_labels)


def test_nani_2(blobs):
    """
    Test the NANI algorithm.
    """
    data, true_labels = blobs
    n_clusters = 7
    mod = KmeansNANI(data, n_clusters, metric='MSD', N_atoms=1,
                     init_type='comp_sim', percentage=10)
    initiators = mod.initiate_kmeans()
    initiators = initiators[:n_clusters]
    kmeans = KMeans(n_clusters=n_clusters, init=initiators, n_init=1, 
                    random_state=None)
    kmeans.fit(data)
    labels = kmeans.labels_
    expected_labels = [3, 1, 3, 3, 5, 1, 3, 6, 1, 6, 0, 1, 1, 1, 
                       6, 1, 2, 6, 2, 2, 3, 4, 2, 1, 2, 2, 0, 4, 
                       3, 3, 1, 5, 5, 2, 6, 1, 4, 5, 1, 0, 5, 6, 
                       3, 6, 3, 5, 5, 0, 1, 6, 5, 5, 3, 0, 1, 3, 
                       3, 1, 5, 4, 2, 0, 3, 3, 0, 4, 1, 1, 0, 3, 
                       6, 3, 1, 5, 3, 5, 0, 4, 3, 6, 3, 3, 5, 5, 
                       1, 1, 5, 3, 0, 4, 5, 0, 4, 4, 2, 5, 1, 1, 
                       1, 4]
    
    assert np.allclose(labels, expected_labels)