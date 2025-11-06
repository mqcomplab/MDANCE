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
    
    mdbirch.set_merge('radius', features=data.shape[1])
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
    mdbirch.set_merge('radius', features=small_test_data.shape[1])
    model = mdBirch(threshold=1.0)
    model.fit(small_test_data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) > 0


@pytest.mark.parametrize("threshold", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_threshold_sensitivity(small_test_data, threshold):
    """Test different threshold values."""
    mdbirch.set_merge('radius', features=small_test_data.shape[1])
    model = mdBirch(threshold=threshold)
    model.fit(small_test_data)
    
    clusters = model.get_cluster_mol_ids()
    n_clusters = len(clusters)
    
    assert n_clusters > 0
    assert n_clusters <= len(small_test_data)


@pytest.mark.parametrize("bf", [10, 25, 50, 100])
def test_branching_factor(small_test_data, bf):
    """Test different branching factors."""
    mdbirch.set_merge('radius', features=small_test_data.shape[1])
    model = mdBirch(threshold=1.0, branching_factor=bf)
    model.fit(small_test_data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) > 0


def test_single_data_point():
    """Test clustering with a single data point."""
    data = np.array([[1, 2, 3, 4, 5]])
    mdbirch.set_merge('radius', features=data.shape[1])
    model = mdBirch(threshold=1.0)
    model.fit(data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) == 1
    assert len(clusters[0]) == 1


def test_identical_data_points():
    """Test clustering with identical data points."""
    data = np.tile([1, 2, 3, 4, 5], (10, 1))
    mdbirch.set_merge('radius', features=data.shape[1])
    model = mdBirch(threshold=1.0)
    model.fit(data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) == 1


def test_very_small_threshold(small_test_data):
    """Test behavior with very small threshold values."""
    mdbirch.set_merge('radius', features=small_test_data.shape[1])
    model = mdBirch(threshold=0.001)
    model.fit(small_test_data)
    
    clusters = model.get_cluster_mol_ids()
    assert len(clusters) > 5


def test_data_validation(test_data):
    """Test clustering quality and data validation."""
    data, true_labels = test_data
    
    mdbirch.set_merge('radius', features=data.shape[1])
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
    mdbirch.set_merge('radius', features=data.shape[1])
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
        mdbirch.set_merge('radius', features=data.shape[1])
        model = mdBirch(threshold=1.0, branching_factor=50)
        model.fit(data)
        clusters = model.get_cluster_mol_ids()
        results.append(len(clusters))
    
    assert len(set(results)) == 1


def test_error_handling():
    """Test error handling for invalid inputs."""
    with pytest.raises(Exception):
        # Test with empty data
        empty_data = np.array([]).reshape(0, 5)
        mdbirch.set_merge('radius', features=5)
        model = mdBirch(threshold=1.0)
        model.fit(empty_data)


if __name__ == "__main__":
    # For direct script execution
    pytest.main([__file__, "-v"])