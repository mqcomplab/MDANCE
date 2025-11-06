import numpy as np

from mdance.cluster import mdbirch
from mdance import data

# Parameters and configuration
input_traj_numpy = data.sim_traj_numpy
sieve = 1
threshold = 3.0600

if __name__ == '__main__':
    # Load trajectory data
    traj_numpy = np.load(input_traj_numpy)[::sieve]
    
    # Set merge criterion
    D = traj_numpy.shape[1]
    mdbirch.set_merge('radius', features=D)

    # Initialize and fit mdBirch model
    model = mdbirch.mdBirch(threshold=threshold)
    model.fit(traj_numpy)
    
    clusters = model.get_cluster_mol_ids()
    centroids = model.get_centroids()
    
    # Create frame-to-cluster label assignment
    n_frames = len(traj_numpy)
    frame_labels = np.full(n_frames, -1, dtype=int)
    
    for cluster_id, frame_indices in enumerate(clusters):
        for frame_idx in frame_indices:
            frame_labels[frame_idx] = cluster_id
    
    print(f"Number of clusters: {len(clusters)}")
    
    # Create CSV output with frame,label 
    frame_numbers = np.arange(n_frames)
    csv_data = np.column_stack((frame_numbers, frame_labels))
    
    # Save as CSV with header
    output_csv = f'mdbirch_labels_{threshold}.csv'
    np.savetxt(output_csv, csv_data, delimiter=',', header='frame,label', fmt='%d', comments='')

    
