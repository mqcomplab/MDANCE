import numpy as np
import pytest

from mdance.inputs.preprocess import normalize_file
from mdance import data
import sys
sys.path.insert(0, "./")
from src.mdance.tools import bts


@pytest.fixture(scope='module')
def bit_data():
    """
    Create a 5x6 binary array for testing.
    """
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
    """
    Create a 5x6 continuous array for testing.
    """
    arr = np.array(
        [[1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
         [7.8, 8.9, 9.0, 1.2, 2.3, 3.4],
         [4.5, 0.6, 6.7, 7.8, 8.9, 0.4],
         [3.2, 4.3, 5.4, 6.5, 7.6, 8.7],
         [9.8, 0.9, 1.0, 2.1, 3.2, 4.3]]
    )
    return arr


@pytest.fixture(scope='module')
def sim_data():
    return np.load(data.sim_traj_numpy)


@pytest.fixture(scope='module', params=['bit_data', 'continuous_data', 'sim_data'])
def matrix(request):
    """
    Parametrized fixture for bit and continuous data.
    """
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def c_sum(matrix):
    """
    Calculate the sum of the continuous data.
    """
    return np.sum(matrix, axis=0)


@pytest.fixture(scope='module')
def sq_sum(matrix):
    """
    Calculate the sum of squares of the continuous data.
    """
    return np.sum(matrix**2, axis=0)


@pytest.fixture(scope='module')
def n_objects(matrix):
    """
    Calculate the number of data points.
    """
    return len(matrix)
  

def test_extended_comparison_full(matrix, bit_data, continuous_data, sim_data):
    """
    Test the extended comparison function.
    """
    if np.array_equal(matrix, bit_data):
        N_atoms = 1
        expected_msd = 2.56
    elif np.array_equal(matrix, continuous_data):
        N_atoms = 2
        expected_msd = 47.1272
    elif np.array_equal(matrix, sim_data):
        N_atoms = 50
        expected_msd = 17.646554567969837
    msd = bts.mean_sq_dev(matrix, N_atoms=N_atoms)
    assert np.allclose(msd, expected_msd, rtol=1e-05, atol=1e-08)
    ec = bts.extended_comparison(matrix, 'full', 'MSD', N_atoms=N_atoms)
    assert np.allclose(ec, expected_msd, rtol=1e-05, atol=1e-08)


def test_extended_comparison_condensed(matrix, bit_data, continuous_data, sim_data,
                                       c_sum, sq_sum, n_objects):
    """
    Test the extended comparison function.
    """
    if np.array_equal(matrix, bit_data):
        N_atoms = 1
        expected_msd = 2.56
    elif np.array_equal(matrix, continuous_data):
        N_atoms = 2
        expected_msd = 47.1272
    elif np.array_equal(matrix, sim_data):
        N_atoms = 50
        expected_msd = 17.646554567969837
    msd = bts.msd_condensed(c_sum, sq_sum, N=n_objects, N_atoms=N_atoms)
    assert np.allclose(msd, expected_msd, rtol=1e-05, atol=1e-08)
    ec = bts.extended_comparison((c_sum, sq_sum), 'condensed', 'MSD', n_objects, N_atoms)
    assert np.allclose(ec, expected_msd, rtol=1e-05, atol=1e-08)


def test_ec_1D():
    """
    Test the extended_comparison function with zero values.
    """
    matrix = np.array([1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        bts.extended_comparison(matrix, 'full', 'MSD')


def test_ec_one_object(c_sum, sq_sum):
    """
    Test the extended_comparison function with zero values.
    """
    matrix = np.array([[1, 0, 0, 0, 0, 0]])
    msd = bts.extended_comparison(matrix, 'full', 'MSD')
    assert msd == 0.0
    msd = bts.extended_comparison([c_sum, sq_sum], 'condensed', 'MSD', N=1)
    assert msd == 0.0

def test_extended_comparisons_esim(matrix, bit_data, continuous_data, sim_data,
                                   c_sum, n_objects):
    """
    Test the extended comparison function. 

    .. note:: Continuous data are required to be normalized before using the ESIM metric.
    """

    if np.array_equal(matrix, bit_data):
        expected_esim = 0.8
    elif np.array_equal(matrix, continuous_data):
        matrix, min, max, avg = normalize_file(matrix, norm_type='v3')
        c_sum = np.sum(matrix, axis=0)
        expected_esim = 1.0
    elif np.array_equal(matrix, sim_data):
        matrix, min, max, avg = normalize_file(matrix, norm_type='v3')
        c_sum = np.sum(matrix, axis=0)
        expected_esim = 0.8843967691266875
    esim = bts.extended_comparison([c_sum], 'condensed', metric='RR', N=n_objects, 
                                   c_threshold=None, w_factor='fraction')
    assert np.allclose(esim, expected_esim, rtol=1e-05, atol=1e-08)


def test_calculate_comp_sim(matrix, bit_data, continuous_data, sim_data):
    """
    Test the calculate_comp_sim function.
    """
    if np.array_equal(matrix, bit_data):
        N_atoms = 1
        expected_cc = np.array([2.25, 2.0, 2.625, 2.625, 2.5])
        
    elif np.array_equal(matrix, continuous_data):
        N_atoms = 2
        expected_cc = np.array([51.120625, 35.74125, 42.540625, 49.545625, 41.960625])
    elif np.array_equal(matrix, sim_data):
        N_atoms = 50
        expected_cc = np.load(data.cc_sim)
        expected_cc = expected_cc[:, 1]
    cc = bts.calculate_comp_sim(matrix, 'MSD', N_atoms=N_atoms)
    expected_cc = np.array(expected_cc)
    assert np.allclose(cc, expected_cc, rtol=1e-05, atol=1e-08)
    

def test_calculate_medoid(matrix, bit_data, continuous_data, sim_data):
    """
    Test the calculate_comp_sim function.
    """
    if np.array_equal(matrix, bit_data):
        N_atoms = 1
        expected_idx = 2
    elif np.array_equal(matrix, continuous_data):
        N_atoms = 2
        expected_idx = 0
    elif np.array_equal(matrix, sim_data):
        N_atoms = 50
        expected_idx = 409
    idx = bts.calculate_medoid(matrix, 'MSD', N_atoms)
    assert idx == expected_idx


def test_calculate_outlier(matrix, bit_data, continuous_data, sim_data):
    """
    Test the calculate_outlier function.
    """
    if np.array_equal(matrix, bit_data):
        N_atoms = 1
        expected_idx = 1
    elif np.array_equal(matrix, continuous_data):
        N_atoms = 2
        expected_idx = 1
    elif np.array_equal(matrix, sim_data):
        N_atoms = 50
        expected_idx = 754
    idx = bts.calculate_outlier(matrix, 'MSD', N_atoms)
    assert idx == expected_idx


def test_trim_outliers(matrix, bit_data, continuous_data, sim_data):
    """
    Test the trim_outliers function.
    """
    if np.array_equal(matrix, bit_data):
        N_atoms = 1
        expected_matrix = np.array(
            [[1, 0, 0, 0, 1, 1],
             [1, 1, 0, 1, 1, 1]]
        )
        output = bts.trim_outliers(matrix, 0.6, 'MSD', N_atoms)
    elif np.array_equal(matrix, continuous_data):
        N_atoms = 2
        expected_matrix = np.array(
            [[1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
             [3.2, 4.3, 5.4, 6.5, 7.6, 8.7]]
        )
        output = bts.trim_outliers(matrix, 0.6, 'MSD', N_atoms)
    elif np.array_equal(matrix, sim_data):
        N_atoms = 50
        output = bts.trim_outliers(matrix, 0.99, 'MSD', N_atoms)
        expected_matrix = np.load(data.trimmed_sim)
    assert np.array_equal(output, expected_matrix)


def test_diversity_selection(matrix, bit_data, continuous_data, sim_data):
    """
    Test the diversity_selection function.
    """
    if np.array_equal(matrix, bit_data):
        N_atoms = 1
        expected_idxs = [2, 0]
        idxs = bts.diversity_selection(matrix, 40, 'MSD', N_atoms=N_atoms)
    elif np.array_equal(matrix, continuous_data):
        N_atoms = 2
        expected_idxs = [0, 1]
        idxs = bts.diversity_selection(matrix, 40, 'MSD', N_atoms=N_atoms)
    elif np.array_equal(matrix, sim_data):
        N_atoms = 50
        expected_idxs = [409, 4972, 3136, 754, 1064, 1735, 4037, 2375, 
                         1335, 1257, 4639, 1711, 3393, 3264, 737, 1634, 
                         5792, 1734, 3392, 3304, 1353, 1467, 525, 3238, 
                         450, 2970, 5102, 1253, 3979, 2951]
        idxs = bts.diversity_selection(matrix, 0.5, 'MSD', N_atoms=N_atoms)
    assert idxs == expected_idxs