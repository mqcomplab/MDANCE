import numpy as np
import pytest
from sklearn.datasets import make_blobs

from mdance.cluster.equal import ExtendedQuality, compute_scores


@pytest.fixture(scope='module')
def blobs():
    """
    Generate blobs.
    """
    data, true_labels = make_blobs(n_samples=1000, centers=7, n_features=2, random_state=0)
    return data, true_labels


def test_equal(blobs):
    """
    Test the eQual algorithm.
    """
    data, true_labels = blobs
    eQ = ExtendedQuality(data=data, threshold=0.8, 
                         metric='MSD', N_atoms=1, 
                         seed_method='comp_sim', n_seeds=3, 
                         check_sim=True, reject_lowd=True, 
                         sim_threshold=2)
    clusters = eQ.run()
    top_pops = eQ.calculate_populations(clusters)
    ch, db = compute_scores(clusters)
    assert np.allclose(ch, 1565.3279046137613, atol=1e-6)
    assert np.allclose(db, 0.9568400160972921, atol=1e-6)
    assert top_pops == [0.8, 28, '0.570000', '0.086000', 
                        '0.081000', '0.066000', '0.060000', 
                        '0.055000', '0.048000', '0.047000', 
                        '0.046000', '0.042000', '0.039000']