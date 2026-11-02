import os

import numpy as np

from mdance.cluster import mdbirch
from mdance import data
from mdance.tools.bts import extended_comparison


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
N_atoms = 50                                        # Number of atoms in the system (for the cluster MSD)
sieve = 1                                           # Use every sieve-th frame (1 = all frames)

# mdBIRCH parameters
threshold = 3.06                                    # Subcluster radius; larger values give fewer clusters
output_dir = 'outputs'                              # Folder for output files


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectory data
    traj_numpy = np.load(input_traj_numpy)[::sieve]

    # Set merge criterion
    D = traj_numpy.shape[1]
    mdbirch.set_merge('radius', features=D)

    # Initialize and fit mdBirch model
    model = mdbirch.mdBirch(threshold=threshold)
    model.fit(traj_numpy)

    clusters = model.get_cluster_mol_ids()
    # Renumber the clusters by size, largest first
    clusters = sorted(clusters, key=len, reverse=True)
    centroids = model.get_centroids()

    # Create frame-to-cluster label assignment
    n_frames = len(traj_numpy)
    frame_labels = np.full(n_frames, -1, dtype=int)

    for cluster_id, frame_indices in enumerate(clusters):
        for frame_idx in frame_indices:
            frame_labels[frame_idx] = cluster_id

    print(f"Number of clusters: {len(clusters)}")

    # Create CSV output with frame,label
    frame_numbers = np.arange(n_frames) * sieve
    csv_data = np.column_stack((frame_numbers, frame_labels))
    csv_data = csv_data[frame_labels >= 0]  # frames not assigned to any cluster are left out

    # Save as CSV with header
    output_csv = f'{output_dir}/mdbirch_labels_{threshold}.csv'
    params = f'mdBIRCH, threshold: {threshold}, n_clusters: {len(clusters)}, sieve: {sieve}'
    np.savetxt(output_csv, csv_data, delimiter=',', header=f'{params}\nFrame Index, Cluster Index', fmt='%d')

    # Calculate population and MSD of each cluster
    with open(f'{output_dir}/mdbirch_summary_{threshold}.csv', 'w') as f:
        f.write(f'# {params}, total frames: {n_frames}\n')
        f.write('# Cluster Index, Fraction of total frames, Number of frames, MSD\n')
        for i in np.unique(frame_labels[frame_labels >= 0]):
            members = traj_numpy[frame_labels == i]
            msd = extended_comparison(members, data_type='full', metric='MSD', N_atoms=N_atoms)
            f.write(f'{i},{len(members) / n_frames},{len(members)},{msd}\n')
