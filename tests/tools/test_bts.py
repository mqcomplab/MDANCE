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
def data_type(request):
    """Parametrized fixture for bit and continuous data."""
    return request.getfixturevalue(request.param)


def test_mean_sq_dev(data_type, bit_data, continuous_data):
    """Test the mean square deviation function."""
    if np.array_equal(data_type, bit_data):
        N_atoms = 1
        expected_msd = 2.56  
    elif np.array_equal(data_type, continuous_data):
        N_atoms = 2
        expected_msd = 45.5704 
    msd = bts.mean_sq_dev(data_type, N_atoms=N_atoms)
    assert np.allclose(msd, expected_msd, rtol=1e-05, atol=1e-08)