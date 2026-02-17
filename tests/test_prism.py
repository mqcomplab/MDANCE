import numpy as np
import pytest

from mdance.cluster.prism import PRISM


@pytest.fixture
def pathways():
    """
    Fixture for PRISM pathways.

    Format: list of (trajectory_id, frames) tuples where:
      - trajectory_id is a string that can be cast to int (PRISM enforces numeric IDs)
      - frames is a 2D numpy array of shape (n_frames, n_features)
    """
    frames_all = [
        ("0", np.array([
            [-74.23699951,  67.61009979],
            [-53.13778687, 113.60979462],
            [-53.13781738, 113.60980225],
            [ 34.90659714,  95.15899658],
            [ 63.34859848,  27.15789986],
            [ 44.89619827,  42.38359833],
            [ 45.58160019,  27.85600090],
            [ 54.48529816,  71.10389709],
            [ 55.10719681,   6.53830004],
            [ 61.41569901,  38.73909760],
        ])),
        ("1", np.array([
            [ -78.74139404,   86.88970184],
            [-124.14131165,  133.28540039],
            [ -62.45608521,  179.37069702],
            [ -94.38858032, -179.26660156],
            [-122.37721252,  122.54660034],
            [ -75.46469116,  166.67678833],
            [ -89.63159180,  152.05979919],
            [ -59.07040405,  162.12379456],
            [ -62.35968018,  153.83779907],
            [ -90.17648315,  152.10429382],
        ])),
        ("2", np.array([
            [-142.79400635,   95.42210388],
            [ -88.90679932,  131.95779419],
            [-153.45500183,  165.94718933],
            [-149.34820557, -174.05441284],
            [-116.89601135,  167.75419617],
            [-116.65299988,  159.13800049],
            [ -61.93249512,  143.74169922],
            [ -90.13357544,  143.02029419],
            [-118.12361145,  179.47659302],
            [-132.56130981,  176.59658813],
        ])),
        # Duplicate some "families" like your SHINE fixture to encourage clustering
        ("3", np.array([
            [-142.79400635,   95.42210388],
            [ -88.90679932,  131.95779419],
            [-153.45500183,  165.94718933],
            [-149.34820557, -174.05441284],
            [-116.89601135,  167.75419617],
            [-116.65299988,  159.13800049],
            [ -61.93249512,  143.74169922],
            [ -90.13357544,  143.02029419],
            [-118.12361145,  179.47659302],
            [-132.56130981,  176.59658813],
        ])),
        ("4", np.array([
            [ -77.67678833,   95.04299927],
            [ -69.02969360,  143.34530640],
            [ -94.15249634, -172.65719604],
            [ -96.93420410,  148.96929932],
            [ -95.92849731,  143.04420471],
            [ -70.60339355,  148.94981384],
            [ -72.47839355,  161.40290833],
            [-101.28518677,  162.35510254],
            [ -97.30999756,  143.41461182],
            [ -80.29470825, -164.96868896],
        ])),
        ("5", np.array([
            [ -77.67678833,   95.04299927],
            [ -69.02969360,  143.34530640],
            [ -94.15249634, -172.65719604],
            [ -96.93420410,  148.96929932],
            [ -95.92849731,  143.04420471],
            [ -70.60339355,  148.94981384],
            [ -72.47839355,  161.40290833],
            [-101.28518677,  162.35510254],
            [ -97.30999756,  143.41461182],
            [ -80.29470825, -164.96868896],
        ])),
    ]
    return frames_all


def test_process_trajs_numeric_ids(pathways):
    """PRISM should coerce numeric-looking IDs to int keys."""
    mod = PRISM(
        pathways,
        metric="MSD",
        t=2,
        criterion="maxclust",
        link="ward",
        option=1,
        k=3,
        k_final=3,
        weight_scheme="weighted_avg",
        N_atoms=1,
        percentage=100,
    )
    pathway_dict = mod.process_trajs()
    assert set(pathway_dict.keys()) == {0, 1, 2, 3, 4, 5}
    assert len(pathway_dict) == 6


def test_process_trajs_rejects_non_numeric_id(pathways):
    """PRISM should raise a clear error for non-numeric trajectory IDs."""
    bad = list(pathways) + [("trajA", np.array([[0.0, 1.0], [2.0, 3.0]]))]
    mod = PRISM(
        bad,
        metric="MSD",
        t=2,
        criterion="maxclust",
        link="ward",
        option=1,
        k=3,
        k_final=3,
        weight_scheme="weighted_avg",
        N_atoms=1,
        percentage=100,
    )
    with pytest.raises(ValueError, match="not numeric"):
        mod.process_trajs()


def test_validate_inputs_rejects_bad_weight_scheme(pathways):
    """PRISM should reject invalid weight schemes during init validation."""
    with pytest.raises(ValueError, match="weight_scheme"):
        PRISM(
            pathways,
            metric="MSD",
            t=2,
            criterion="maxclust",
            link="ward",
            option=1,
            k=3,
            k_final=3,
            weight_scheme="not_a_scheme",
            N_atoms=1,
            percentage=100,
        )


def test_prism_runs(pathways):
    """Smoke test: PRISM should run end-to-end and return linkage + clusters."""
    mod = PRISM(
        pathways,
        metric="MSD",
        t=2,
        criterion="maxclust",
        link="ward",
        option=3,
        k=3,
        k_final=3,
        weight_scheme="weighted_avg",
        N_atoms=1,
        percentage=100,
    )
    link_matrix, clusters = mod.run()

    # Basic structural checks
    assert link_matrix is not None
    assert clusters is not None
    assert len(clusters) == len(pathways)               # one label per trajectory
    assert len(np.unique(clusters)) <= 2                # since t=2 maxclust


def test_plot(pathways):
    """Smoke test: plot() should run after run()."""
    mod = PRISM(
        pathways,
        metric="MSD",
        t=2,
        criterion="maxclust",
        link="ward",
        option=2,
        k=3,
        k_final=3,
        weight_scheme="weighted_avg",
        N_atoms=1,
        percentage=100,
    )
    mod.run()
    assert mod.plot() is not None


def test_inconsistent_feature_dims_raises():
    """PRISM should fail if trajectories have different n_features."""
    bad = [
        ("0", np.ones((10, 2))),
        ("1", np.ones((10, 3))),  # different feature dimension
    ]
    mod = PRISM(
        bad,
        metric="MSD",
        t=2,
        criterion="maxclust",
        link="ward",
        option=1,
        k=3,
        k_final=3,
        weight_scheme="weighted_avg",
        N_atoms=1,
        percentage=100,
    )
    # The error is triggered during medoid construction inside run()
    with pytest.raises(ValueError, match="same number of features"):
        mod.run()
