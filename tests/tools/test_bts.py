import numpy as np
import pytest
import mdance.tools.bts as bts


@pytest.fixture(scope='module')
def bit_data():
    """Create a 5x6 binary array for testing."""
    arr = np.array(
        [[0, 1, 0, 0, 1, 0],
         [1, 0, 1, 1, 0, 1],
         [1, 0, 0, 0, 1, 1],
         [1, 1, 0, 1, 1, 1],
         [0, 1, 1, 0, 1, 1]]
    )
    return arr


@pytest.fixture(scope='module')
def continuous_data():
    """Create a 5x6 continuous array for testing."""
    arr = np.array(
        [[1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
         [7.8, 8.9, 9.0, 1.2, 2.3, 3.4],
         [4.5, 0.6, 6.7, 7.8, 8.9, 0.4],
         [1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
         [7.8, 8.9, 9.0, 1.2, 2.3, 3.4]]
    )
    return arr


@pytest.fixture(scope='module', params=['bit_data', 'continuous_data'])
def matrix(request):
    """Parametrized fixture for bit and continuous data."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def c_sum(matrix):
    """Calculate the sum of the continuous data."""
    return np.sum(matrix, axis=0)


@pytest.fixture(scope='module')
def sq_sum(matrix):
    """Calculate the sum of squares of the continuous data."""
    return np.sum(matrix**2, axis=0)


@pytest.fixture(scope='module')
def n_objects(matrix):
    """Calculate the number of data points."""
    return len(matrix)
  

def test_extended_comparison_full(matrix, bit_data, continuous_data):
    """Test the extended comparison function."""
    if np.array_equal(matrix, bit_data):
        N_atoms = 1
        expected_msd = 2.56
    elif np.array_equal(matrix, continuous_data):
        N_atoms = 2
        expected_msd = 45.5704
    msd = bts.mean_sq_dev(matrix, N_atoms=N_atoms)
    assert np.allclose(msd, expected_msd, rtol=1e-05, atol=1e-08)
    ec = bts.extended_comparison(matrix, 'full', 'MSD', N_atoms=N_atoms)
    assert np.allclose(ec, expected_msd, rtol=1e-05, atol=1e-08)


def test_extended_comparison_condensed(matrix, bit_data, continuous_data, c_sum, sq_sum, n_objects):
    """Test the extended comparison function."""
    if np.array_equal(matrix, bit_data):
        N_atoms = 1
        expected_msd = 2.56
    elif np.array_equal(matrix, continuous_data):
        N_atoms = 2
        expected_msd = 45.5704
    msd = bts.msd_condensed(c_sum, sq_sum, N=n_objects, N_atoms=N_atoms)
    assert np.allclose(msd, expected_msd, rtol=1e-05, atol=1e-08)
    ec = bts.extended_comparison((c_sum, sq_sum), 'condensed', 'MSD', n_objects, N_atoms)
    assert np.allclose(ec, expected_msd, rtol=1e-05, atol=1e-08)


def test_extended_comparisons_esim(matrix, bit_data, continuous_data, c_sum):
    """Test the extended comparison function."""
    if np.array_equal(matrix, bit_data):
        expected_esim = 0.2
    elif np.array_equal(matrix, continuous_data):
        expected_esim = 0.4
    esim = bts.extended_comparison(c_sum, 'full', metric='RR', N=n_objects, 
                                   c_threshold=None, w_factor='fraction')
    print(esim)
    assert np.allclose(esim, expected_esim, rtol=1e-05, atol=1e-08)