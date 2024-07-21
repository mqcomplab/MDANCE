import mdance.tools.bts as bts
import numpy as np
import pytest


@pytest.fixture(scope='module')
def bit_data():
    arr = np.array(
        [[0, 1, 0, 0, 1, 0],
         [1, 0, 1, 1, 0, 1],
         [1, 0, 0, 0, 1, 1],
         [1, 1, 0, 1, 1, 1],
         [0, 1, 1, 0, 1, 1]
        ])
    return arr


@pytest.fixture(scope='module')
def continuous_data():
    arr = np.array(
        [[1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
         [7.8, 8.9, 9.0, 1.2, 2.3, 3.4],
         [4.5, 0.6, 6.7, 7.8, 8.9, 0.4],
         [1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
         [7.8, 8.9, 9.0, 1.2, 2.3, 3.4]
        ])
    return arr


@pytest.fixture(scope='module', params=[bit_data, continuous_data])
def data_type(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def c_sum(data_type):
    return np.sum(data_type, axis=0)


@pytest.fixture(scope='module')
def n_objects(data_type):
    return len(data_type)


def test_mean_sq_dev(data_type):
    if np.array_equal(data_type, bit_data):
        expected_msd = 2.56
        N_atoms = 1
        msd = bts.mean_sq_dev(data_type, N_atoms=N_atoms)
    elif np.array_equal(data_type, continuous_data):
        expected_msd = 2.56
        N_atoms = 2
        msd = bts.mean_sq_dev(data_type, N_atoms=N_atoms)
    print(msd)
    assert np.allclose(msd, expected_msd, rtol=1e-05, atol=1e-08)
    