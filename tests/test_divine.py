import numpy as np
import pytest
from sklearn.datasets import make_blobs

from mdance.cluster.divine import Divine   

@pytest.fixture(scope='module')

def data():
    X, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)
    return X

@pytest.mark.parametrize("split", ['MSD', 'radius', 'weighted_MSD'])
@pytest.mark.parametrize("anchors", ['nani', 'outlier_pair', 'splinter_split'])
@pytest.mark.parametrize("refine", [True, False])
def test_divine_combinations(data, split, anchors, refine):

    model = Divine(data=data,
                   split=split,
                   anchors=anchors,
                   init_type='comp_sim',
                   end='k',
                   k=3,
                   N_atoms=1,
                   threshold=0.0,
                   refine=refine,
                   percentage=10)

    clusters, labels, scores, msds = model.run()

    assert isinstance(clusters, list)
    assert all(isinstance(c, np.ndarray) for c in clusters)
    assert len(clusters) == 3
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(data)
    assert np.unique(labels).size == 3
    assert all(len(score) == 3 for score in scores)
    assert all(len(msd) == 4 for msd in msds)


def test_divine_threshold(data):
    model = Divine(data=data,
                   split='weighted_MSD',
                   anchors='nani',
                   init_type='comp_sim',
                   end='k',
                   k=3,
                   N_atoms=1,
                   threshold=0.2,
                   refine=True)

    clusters, labels, _, _ = model.run()
    for cluster in clusters:
        assert len(cluster) >= 30


def test_divine_point(data):
    """
    Test the point-wise stopping condition.
    """
    small_data = data[:5]
    model = Divine(data=small_data,
                   split='MSD',
                   anchors='splinter_split',
                   init_type='strat_all',
                   end='points',
                   N_atoms=1,
                   threshold=0.0,
                   refine=False)

    clusters, labels, scores, msds = model.run()

    assert len(clusters) == len(small_data)
    assert np.all(np.bincount(labels) == 1)
    assert all(len(msd) == 4 for msd in msds)


def test_divine_invalid_split(data):
    """
    Check invalid `split` raises ValueError.
    """
    with pytest.raises(ValueError):
        Divine(data, split='invalid', anchors='nani', end='k', k=3).run()


def test_divine_invalid_anchors(data):
    """
    Check invalid `anchors` raises ValueError.
    """
    with pytest.raises(ValueError):
        Divine(data, split='MSD', anchors='bad_anchor', end='k', k=3).run()


def test_divine_invalid_k(data):
    """
    Check `k` exceeding data size raises ValueError.
    """
    with pytest.raises(ValueError):
        Divine(data[:5], split='MSD', anchors='nani', end='k', k=10).run()


def test_divine_empty_input():
    """
    Check behavior when data is empty.
    """
    data = np.empty((0, 2))
    model = Divine(data=data, split='MSD', anchors='nani', end='k', k=3)
    clusters, labels, scores, msds = model.run()
    assert clusters == []
    assert labels.size == 0
    assert scores == []
    assert msds == []
