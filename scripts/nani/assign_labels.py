import os

import numpy as np

from mdance.cluster.nani import KmeansNANI
from mdance import data
from mdance.tools.bts import extended_comparison, calculate_medoid


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
N_atoms = 50                                        # Number of atoms in the system
sieve = 1                                           # Use every sieve-th frame (1 = all frames)

# NANI parameters
n_clusters = 6                                      # Number of clusters (pick from screen_nani.py)
init_type = 'strat_all'                             # Seeding method, same as in screen_nani.py
metric = 'MSD'                                      # Distance metric for clustering
n_structures = 11                                   # Frames saved per cluster, closest to its medoid
output_dir = 'outputs'                              # Folder for output files


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    traj_numpy = np.load(input_traj_numpy)[::sieve]
    mod = KmeansNANI(data=traj_numpy, n_clusters=n_clusters, N_atoms=N_atoms, init_type=init_type,
                     metric=metric, percentage=10)
    labels, centers, n_iter = mod.execute_kmeans_all()
    sort_labels_by_size = np.argsort(np.bincount(labels))[::-1]
    labels = np.array([np.where(sort_labels_by_size == i)[0][0] for i in labels])
    params = f'NANI, init_type: {init_type}, n_clusters: {n_clusters}, metric: {metric}, sieve: {sieve}'

    # Save best frames indices for each cluster
    best_frames_indices = []
    for label in np.unique(labels):
        cluster = np.where(labels == label)[0]
        if len(cluster) > 1:
            medoid_index = calculate_medoid(traj_numpy[cluster], metric=metric, N_atoms=N_atoms)
            medoid = traj_numpy[cluster][medoid_index]
            msd_to_medoid = [extended_comparison(np.array([frame, medoid]), data_type='full', metric=metric,
                                                 N_atoms=N_atoms) for frame in traj_numpy[cluster]]
            best = cluster[np.argsort(msd_to_medoid)[:n_structures]]
        else:
            best = cluster
        for frame in best:
            best_frames_indices.append((frame * sieve, label))
    np.savetxt(f'{output_dir}/nani_best_frames_{n_clusters}.csv', best_frames_indices, delimiter=',', fmt='%d',
               header=f'{params}, n_structures: {n_structures}\nFrame Index, Cluster Index')

    # Save cluster labels
    with open(f'{output_dir}/nani_labels_{n_clusters}.csv', 'w') as f:
        f.write(f'# {params}\n')
        f.write('# Frame Index, Cluster Index\n')
        for i, row in enumerate(labels):
            f.write(f'{i * sieve},{row}\n')

    # Calculate population and MSD of each cluster
    with open(f'{output_dir}/nani_summary_{n_clusters}.csv', 'w') as f:
        f.write(f'# {params}, total frames: {len(labels)}\n')
        f.write('# Cluster Index, Fraction of total frames, Number of frames, MSD\n')
        for i in np.unique(labels[labels >= 0]):
            members = traj_numpy[labels == i]
            msd = extended_comparison(members, data_type='full', metric=metric, N_atoms=N_atoms)
            f.write(f'{i},{len(members) / len(labels)},{len(members)},{msd}\n')
