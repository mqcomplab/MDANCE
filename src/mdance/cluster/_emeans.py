"""Optional e-means solver for the existing NANI centroid initialization."""
import numpy as np


def fit_emeans(data, initiators, n_clusters, nepochs=5, seed=1234,
               batch_law='auto'):
    """Return final-centroid labels, centers, and the epoch count.

    The third return value denotes epochs, unlike Lloyd iteration count.
    Only explicit NANI centers are supported. Inputs are converted to float32.
    """
    X = np.asarray(data)
    initial = np.asarray(initiators)
    if X.ndim != 2 or X.shape[0] < n_clusters or X.shape[1] < 1:
        raise ValueError("data must have shape (n_frames >= n_clusters, n_features)")
    if initial.shape != (n_clusters, X.shape[1]):
        raise ValueError("initiators must have shape (n_clusters, n_features)")
    if not np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.complexfloating) or not np.all(np.isfinite(X)):
        raise ValueError("data must contain finite real values")
    if not np.issubdtype(initial.dtype, np.number) or np.issubdtype(initial.dtype, np.complexfloating) or not np.all(np.isfinite(initial)):
        raise ValueError("initiators must contain finite real values")
    if isinstance(nepochs, (bool, np.bool_)) or not isinstance(nepochs, (int, np.integer)) or nepochs <= 0:
        raise ValueError("nepochs must be a positive integer")
    X = np.ascontiguousarray(X, dtype=np.float32)
    initial = np.ascontiguousarray(initial, dtype=np.float32)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(initial)):
        raise ValueError("inputs cannot be represented as finite float32 values")

    try:
        from faiss.contrib.e_means import NumpyEmeans
    except ImportError as exc:
        raise ImportError(
            "e-means requires Faiss with faiss.contrib.e_means.NumpyEmeans"
        ) from exc
    solver = NumpyEmeans(X.shape[1], n_clusters, nepochs=nepochs,
                         seed=seed, batch_law=batch_law,
                         capture_assignments=False)
    centers, _ = solver.train(X, init_centroids=initial.copy())
    # Training labels precede the final centroid update; recompute them.
    _, labels = solver.assign(X)
    return labels, np.asarray(centers), nepochs
