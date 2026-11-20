import numpy as np
import time
import pytest

from mdance.cluster import mdbirch
from mdance.cluster.mdbirch import mdBirch

@pytest.fixture
def test_data():
    """
    Fixture for test data. Returns a tuple containing synthetic clustering data
    and true labels for validation.
    """
    np.random.seed(42)
    n_samples, n_features = 100, 50
    
    # Create synthetic clusters
    cluster1 = np.random.normal(0, 1, (n_samples//3, n_features))
    cluster2 = np.random.normal(5, 1, (n_samples//3, n_features))
    cluster3 = np.random.normal(-3, 1, (n_samples - 2*(n_samples//3), n_features))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.hstack([
        np.zeros(n_samples//3),
        np.ones(n_samples//3),
        np.full(n_samples - 2*(n_samples//3), 2)
    ])
    
    return data, true_labels.astype(int)


@pytest.fixture
def small_test_data():
    """Fixture for small test data for quick tests."""
    np.random.seed(123)
    return np.random.normal(0, 1, (30, 10))


def test_basic_functionality(test_data):
    """Test basic mdBirch functionality."""
    data, _ = test_data
    
    model = mdBirch(threshold=1.0, branching_factor=50)
    model.fit(data)
    
    clusters = model.get_cluster_mol_ids()
    centroids = model.get_centroids()
    
    assert len(clusters) > 0
    assert len(centroids) == len(clusters)
    assert all(len(cluster) > 0 for cluster in clusters)
    
    # Check all frames are assigned
    assigned_frames = set()
    for cluster in clusters:
        assigned_frames.update(cluster)
    assert len(assigned_frames) == len(data)


def test_merge_criteria(small_test_data):
    """Test radius merge criterion."""
    model = mdBirch(threshold=1.0)
    model.fit(small_test_data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) > 0


@pytest.mark.parametrize("threshold", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_threshold_sensitivity(small_test_data, threshold):
    """Test different threshold values."""
    model = mdBirch(threshold=threshold)
    model.fit(small_test_data)
    
    clusters = model.get_cluster_mol_ids()
    n_clusters = len(clusters)
    
    assert n_clusters > 0
    assert n_clusters <= len(small_test_data)


@pytest.mark.parametrize("bf", [10, 25, 50, 100])
def test_branching_factor(small_test_data, bf):
    """Test different branching factors."""
    model = mdBirch(threshold=1.0, branching_factor=bf)
    model.fit(small_test_data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) > 0


def test_single_data_point():
    """Test clustering with a single data point."""
    data = np.array([[1, 2, 3, 4, 5]])
    model = mdBirch(threshold=1.0)
    model.fit(data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) == 1
    assert len(clusters[0]) == 1


def test_identical_data_points():
    """Test clustering with identical data points."""
    data = np.tile([1, 2, 3, 4, 5], (10, 1))
    model = mdBirch(threshold=1.0)
    model.fit(data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) == 1


def test_very_small_threshold(small_test_data):
    """Test behavior with very small threshold values."""
    model = mdBirch(threshold=0.001)
    model.fit(small_test_data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) > 5


def test_data_validation(test_data):
    """Test clustering quality and data validation."""
    data, true_labels = test_data
    
    model = mdBirch(threshold=1.5)
    model.fit(data)
    
    clusters = model.get_cluster_mol_ids()
    centroids = model.get_centroids()
    
    # Create frame-label mapping
    frame_labels = np.full(len(data), -1)
    for cluster_id, frame_indices in enumerate(clusters):
        for frame_idx in frame_indices:
            frame_labels[frame_idx] = cluster_id
    
    # Validation checks
    assert np.all(frame_labels >= 0)
    assert len(np.unique(frame_labels)) == len(clusters)
    
    # Check centroid dimensions
    for centroid in centroids:
        assert centroid.shape[0] == data.shape[1]


@pytest.mark.parametrize("size", [50, 100, 200])
def test_performance_scaling(size):
    """Test performance with different data sizes."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (size, 25))
    
    start_time = time.time()
    model = mdBirch(threshold=1.0)
    model.fit(data)
    
    clusters = model.get_cluster_mol_ids()
    execution_time = time.time() - start_time
    
    assert execution_time < 10
    assert len(clusters) > 0


def test_consistency():
    """Test result consistency across multiple runs."""
    np.random.seed(123)
    data = np.random.normal(0, 1, (50, 20))
    
    results = []
    for i in range(3):
        model = mdBirch(threshold=1.0, branching_factor=50)
        model.fit(data)
        clusters = model.get_cluster_mol_ids()
        results.append(len(clusters))
    
    assert len(set(results)) == 1


def test_error_handling():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        # Test with empty data
        empty_data = np.array([]).reshape(0, 5)
        model = mdBirch(threshold=1.0)
        model.fit(empty_data)


@pytest.mark.parametrize("bad_X", [
    np.zeros(5),                                   # 1-D
    np.zeros((2, 3, 4)),                           # 3-D
    np.zeros((5, 0)),                              # no features
    np.array([[1.0, np.nan], [0.0, 1.0]]),         # NaN
    np.array([[1.0, np.inf], [0.0, 1.0]]),         # inf
    np.array([["a", "b"], ["c", "d"]]),            # not numeric
    np.ones((3, 2), dtype=complex),                # complex
])
def test_invalid_input_raises(bad_X):
    with pytest.raises(ValueError):
        mdBirch(threshold=1.0).fit(bad_X)


def test_unsupported_merge_criterion():
    with pytest.raises(ValueError):
        mdBirch(threshold=1.0, merge_criterion='diameter')


def test_fit_without_set_merge():
    """fit works in a fresh process without calling set_merge first."""
    import subprocess, sys
    code = ("import numpy as np; from mdance.cluster.mdbirch import mdBirch; "
            "m = mdBirch(threshold=1.0).fit(np.random.default_rng(0).normal(size=(40, 6))); "
            "print(len(m.get_cluster_mol_ids()))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) > 0


def test_models_do_not_share_settings():
    """Fitting a model on data with a different feature count does not
    change the clusters of another model."""
    rng = np.random.default_rng(0)
    X_small = rng.normal(size=(200, 5))
    X_large = rng.normal(size=(200, 500))

    expected = mdBirch(threshold=1.0).fit(X_small).get_cluster_mol_ids()
    model = mdBirch(threshold=1.0)
    mdBirch(threshold=1.0).fit(X_large)
    assert model.fit(X_small).get_cluster_mol_ids() == expected


def test_set_merge_is_deprecated():
    with pytest.warns(FutureWarning):
        mdbirch.set_merge('radius', features=5)
    with pytest.warns(FutureWarning), pytest.raises(ValueError):
        mdbirch.set_merge('diameter')


def test_integer_input_matches_float():
    """Integer input is converted to float, so centroids are not truncated."""
    rng = np.random.default_rng(1)
    X_int = rng.integers(-20, 20, size=(300, 4))
    X_float = X_int.astype(np.float64)

    model_int = mdBirch(threshold=4.0).fit(X_int)
    model_float = mdBirch(threshold=4.0).fit(X_float)
    assert model_int.get_cluster_mol_ids() == model_float.get_cluster_mol_ids()
    assert np.array_equal(np.array(model_int.get_centroids()),
                          np.array(model_float.get_centroids()))
    assert model_int.root_.init_centroids_.dtype == np.float64


def test_sparse_input_matches_dense():
    from scipy import sparse
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 8))
    X[X < 0.5] = 0
    dense = mdBirch(threshold=1.0).fit(X).get_cluster_mol_ids()
    for fmt in (sparse.csr_matrix, sparse.csc_matrix):
        assert mdBirch(threshold=1.0).fit(fmt(X)).get_cluster_mol_ids() == dense


def test_refit_with_different_feature_count_raises():
    rng = np.random.default_rng(3)
    model = mdBirch(threshold=1.0).fit(rng.normal(size=(20, 4)))
    with pytest.raises(ValueError):
        model.fit(rng.normal(size=(20, 5)))


if __name__ == "__main__":
    # For direct script execution
    pytest.main([__file__, "-v"])