import mdance.tools.esim_modules as esim
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
def c_sum(bit_data):
    return np.sum(bit_data, axis=0)


@pytest.fixture(scope='module')
def n_objects(bit_data):
    return len(bit_data)


def test_calculate_counters(c_sum, n_objects):
    counter_dict = esim.calculate_counters(c_sum, n_objects)
    expected_dict = {
        'a': 2, 'w_a': 1.2, 
        'd': 0, 'w_d': 0.0, 
        'total_sim': 2, 'total_w_sim': 1.2, 
        'total_dis': 4, 'total_w_dis': 4.0, 
        'p': 6, 'w_p': 5.2}
    assert counter_dict == expected_dict


def test_gen_sim_dict(c_sum, n_objects):
    sim_dict = esim.gen_sim_dict(c_sum, n_objects)
    expected_dict = {
        'BUB': 0.2, 'Fai': 0.2, 
        'Gle': 0.3, 'Ja': 0.36, 
        'JT': 0.2, 'RT': 0.12, 
        'RR': 0.2, 'SM': 0.2, 
        'SS1': 0.12, 'SS2': 0.3
        }
    # Give a tolerance of 1e-10
    for key in sim_dict:
        assert abs(sim_dict[key] - expected_dict[key]) < 1e-10


def test_calc_medoid(bit_data):
    medoid_idx = esim.calc_medoid(bit_data)
    expected_idx = 3
    assert medoid_idx == expected_idx


def test_calc_outlier(bit_data):
    outlier_idx = esim.calc_outlier(bit_data)
    expected_idx = 0
    assert outlier_idx == expected_idx


def test_calc_comp_sim(bit_data):
    comp_sim = esim.calc_comp_sim(bit_data)
    expected_comp_sim = [
        (0, 1/3), 
        (1, 1/3), 
        (2, 1/4), 
        (3, 1/6), 
        (4, 1/4)
        ]
    # Give a tolerance of 1e-10
    for i, j in zip(comp_sim, expected_comp_sim):
        assert i[0] == j[0]
        assert abs(i[1] - j[1]) < 1e-10