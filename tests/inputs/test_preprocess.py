import numpy as np
import pytest

import mdance.inputs.preprocess as pre


@pytest.fixture(scope='module')
def expected_traj_numpy():
    traj_numpy = np.load('../../data/md/backbone.npy')
    return traj_numpy


def test_gen_traj_numpy(expected_traj_numpy):
    prmtop = '../../data/md/aligned_tau.pdb'
    traj = '../../data/md/aligned_1000_tau.dcd'
    atom_sel = 'resid 3 to 12 and name N CA C O H'
    traj_numpy = pre.gen_traj_numpy(prmtop, traj, atom_sel)
    assert np.allclose(traj_numpy, expected_traj_numpy)


def test_Normalizer(expected_traj_numpy):
    norm = pre.Normalizer(data=expected_traj_numpy)
    matrix = norm.get_v3_norm()
    min, max, avg = norm.get_min_max()
    assert np.all(matrix >= 0) and np.all(matrix <= 1)
    assert min == pytest.approx(17.97506)
    assert max == pytest.approx(42.1838)