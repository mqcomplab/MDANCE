import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from mdance import data
from mdance.cluster.nani import KmeansNANI


@pytest.fixture(scope='module')
def traj_numpy():
    """
    Load the trajectory data. This is Molecular Dynamics data.
    """
    return np.load(data.sim_traj_numpy)


@pytest.fixture(scope='module')
def blobs():
    """
    Generate blobs. This is a synthetic dataset.
    The data is generated using sklearn's make_blobs function.
    """
    data, true_labels = make_blobs(n_samples=100, centers=7, n_features=2, random_state=0)
    return data, true_labels


def test_nani_comp_sim_md(traj_numpy):
    """
    Test the NANI algorithm coupled with ``comp_sim`` as ``init_type``.
    This takes MD data as input.
    """
    n_clusters = 6
    mod = KmeansNANI(traj_numpy, n_clusters, 'MSD', N_atoms=50,
                     init_type='comp_sim', percentage=10)
    initiators = mod.initiate_kmeans()
    kmeans = KMeans(n_clusters=n_clusters, init=initiators, n_init=1, 
                    random_state=None)
    kmeans.fit(traj_numpy)
    labels = kmeans.labels_
    sort_labels_by_size = np.argsort(np.bincount(labels))[::-1]
    labels = np.array([np.where(sort_labels_by_size == i)[0][0] for i in labels])
    expected_labels = np.loadtxt(data.labels_6, delimiter=',')
    expected_labels = expected_labels[:, 1]
    assert np.allclose(labels, expected_labels)


def test_nani_comp_sim_syn(blobs):
    """
    Test the NANI algorithm coupled with ``comp_sim`` as ``init_type``.
    This takes synthetic data as input.
    """
    data, true_labels = blobs
    n_clusters = 7
    mod = KmeansNANI(data, n_clusters, metric='MSD', N_atoms=1,
                     init_type='comp_sim', percentage=10)
    initiators = mod.initiate_kmeans()
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


def test_nani_strat_syn(blobs):
    """
    Test the NANI algorithm coupled with ``comp_sim`` as ``init_type``.
    This takes synthetic data as input.
    """
    data, true_labels = blobs
    n_clusters = 7
    print(data.shape)
    mod = KmeansNANI(data, n_clusters, metric='MSD', N_atoms=1,
                     init_type='strat_all', percentage=100)
    initiators = mod.initiate_kmeans()
    kmeans = KMeans(n_clusters=n_clusters, init=initiators, n_init=1, 
                    random_state=None)
    kmeans.fit(data)
    labels = kmeans.labels_
    expected_labels = [
        1, 3, 2, 2, 4, 3, 2, 5, 3, 5, 0, 3, 1, 3, 5, 1, 6, 5, 6, 6, 2, 5, 6, 1, 6, 6, 0, 5, 2, 2,
        3, 4, 4, 6, 5, 3, 6, 4, 1, 0, 4, 5, 2, 5, 2, 4, 3, 0, 3, 5, 4, 4, 1, 0, 0, 1, 1, 3, 4, 6,
        6, 0, 2, 1, 0, 5, 1, 3, 0, 2, 5, 2, 1, 4, 2, 3, 0, 5, 2, 5, 1, 2, 4, 4, 3, 1, 4, 1, 0, 6,
        4, 0, 5, 6, 6, 4, 3, 3, 3, 6
    ]
    assert np.allclose(labels, expected_labels)


def test_nani_comp_strat_reduced_md(traj_numpy):
    """
    Test the NANI algorithm coupled with ``strat_reduced`` as ``init_type``.
    This takes MD data as input.
    """
    n_clusters = 6
    mod = KmeansNANI(traj_numpy, n_clusters, 'MSD', N_atoms=50,
                     init_type='strat_reduced', percentage=10)
    initiators = mod.initiate_kmeans()
    kmeans = KMeans(n_clusters=n_clusters, init=initiators, n_init=1, 
                    random_state=None)
    kmeans.fit(traj_numpy)
    labels = kmeans.labels_
    sort_labels_by_size = np.argsort(np.bincount(labels))[::-1]
    labels = np.array([np.where(sort_labels_by_size == i)[0][0] for i in labels])
    expected_labels = np.loadtxt('src/mdance/data/labels_6_strat_reduced.csv', delimiter=',')
    expected_labels = expected_labels[:, 1]
    assert np.allclose(labels, expected_labels)


def test_nani_comp_strat_all_md(traj_numpy):
    """
    Test the NANI algorithm coupled with ``strat_all`` as ``init_type``.
    This takes MD data as input.
    """
    n_clusters = 6
    mod = KmeansNANI(traj_numpy, n_clusters, 'MSD', N_atoms=50,
                     init_type='strat_all', percentage=10)
    initiators = mod.initiate_kmeans()
    kmeans = KMeans(n_clusters=n_clusters, init=initiators, n_init=1, 
                    random_state=None)
    kmeans.fit(traj_numpy)
    labels = kmeans.labels_
    sort_labels_by_size = np.argsort(np.bincount(labels))[::-1]
    labels = np.array([np.where(sort_labels_by_size == i)[0][0] for i in labels])
    expected_labels = np.loadtxt('src/mdance/data/labels_6_strat_all.csv', delimiter=',')
    expected_labels = expected_labels[:, 1]
    assert np.allclose(labels, expected_labels)
    

def test_nani_strat_errors():
    mod = KmeansNANI(np.array([[1, 2], [3, 4], [5, 6]]), n_clusters=3, metric='MSD', N_atoms=50,
                     init_type='strat_all', percentage=80)
    with pytest.raises(ValueError, match="The number of initiators is less than the number of clusters. Try increasing the percentage."):
        mod.initiate_kmeans()