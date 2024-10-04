import numpy as np


def test_db():
    """
    Test the Davies-Bouldin index and the optimal number of clusters.
    """
    scores_csv = 'scripts/nani/outputs/10comp_sim_summary.csv'
    n_clus, db = np.loadtxt(scores_csv, unpack=True, delimiter=',', usecols=(0, 3))

    # Plot the Davies-Bouldin index and the optimal number of clusters
    all_indices = np.argsort(db)
    min_db_index = all_indices[0]
    min_db = n_clus[min_db_index]
    all_indices = np.delete(all_indices, 0)
    second_min_index = all_indices[0]
    second_min_db = n_clus[second_min_index]
    
    assert min_db == 6
    assert second_min_db == 5

    # Calculate the second derivative (before + after - 2*current)
    arr = db
    x = n_clus[1:-1]
    result = []
    for start_index, n_clusters in zip(range(1, len(arr) - 1), x):
        temp = arr[start_index + 1] + arr[start_index - 1] - (2 * arr[start_index])
        if arr[start_index] <= arr[start_index - 1] and arr[start_index] <= arr[start_index + 1]:
            result.append((n_clusters, temp))
    result = np.array(result)
    if len(result) == 0:
        print('No maxima found')
    elif len(result) >= 1:
        sorted_indices = np.argsort(result[:, 1])[::-1]
        sorted_result = result[sorted_indices]
        min_x = sorted_result[0][0]
        if len(sorted_result) >= 2:
            sec_min_x = sorted_result[1][0]

    assert min_x == 8
    assert sec_min_x == 6