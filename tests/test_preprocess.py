import numpy as np
import pytest

import mdance.inputs.preprocess as pre
from mdance import data


@pytest.fixture(scope='module')
def expected_traj_numpy():
    """
    Load the expected trajectory data.
    """
    return np.load(data.sim_traj_numpy)


def test_gen_traj_numpy(expected_traj_numpy):
    """
    Test the generation of a trajectory in numpy format.
    """
    atom_sel = 'resid 3 to 12 and name N CA C O H'
    traj_numpy = pre.gen_traj_numpy(data.top, data.traj, atom_sel)
    assert np.allclose(traj_numpy, expected_traj_numpy)


def test_Normalizer(expected_traj_numpy):
    """
    Test the Normalizer class.
    """
    norm = pre.Normalizer(data=expected_traj_numpy)
    matrix = norm.get_v3_norm()
    min, max, avg = norm.get_min_max()
    assert np.all(matrix >= 0) and np.all(matrix <= 1)
    assert min == pytest.approx(17.97506)
    assert max == pytest.approx(42.1838)