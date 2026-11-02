import os
import pickle as pkl

import numpy as np

from mdance import data
from mdance.cluster.equal import ExtendedQuality
from mdance.tools.bts import extended_comparison, calculate_medoid


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
N_atoms = 50                                        # Number of atoms in the system
sieve = 1                                           # Use every sieve-th frame (1 = all frames)

# eQual parameters
metric = 'MSD'                                      # Distance metric for clustering
n_seeds = 3                                         # Seeds tried per iteration (int, or fraction of the data)
check_sim = True                                    # Reject clusters less similar than sim_threshold
reject_lowd = True                                  # Reject clusters smaller than min_samples
sim_threshold = 16                                  # Largest similarity value a cluster may have
min_samples = 10                                    # Minimum cluster size

# Output parameters
threshold = 5.80                                    # Cluster radius threshold (pick from screen_equal.py)
n_structures = 11                                   # Frames saved per cluster, closest to its medoid
open_clusters = None                                # Saved clusters .pkl to reuse (None = recluster)
output_dir = 'outputs'                              # Folder for output files


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    traj_numpy = np.load(input_traj_numpy)[::sieve]
    total_frames = len(traj_numpy)

    if open_clusters:
        clusters = pkl.load(open(open_clusters, 'rb'))
    else:
        eQ = ExtendedQuality(data=traj_numpy, threshold=threshold,
                             metric=metric, N_atoms=N_atoms, n_seeds=n_seeds,
                             check_sim=check_sim, reject_lowd=reject_lowd,
                             sim_threshold=sim_threshold, min_samples=min_samples)
        clusters = eQ.run()
    params = (f'eQual, threshold: {threshold}, n_clusters: {len(clusters)}, metric: {metric}, n_seeds: {n_seeds}, '
              f'sim_threshold: {sim_threshold}, min_samples: {min_samples}, sieve: {sieve}')

    # eQual returns the frames of each cluster; map them back to frame indices
    # (identical frames are matched to separate indices, in order)
    frame_index = {}
    for i, frame in enumerate(traj_numpy):
        frame_index.setdefault(frame.tobytes(), []).append(i)
    labels = np.full(total_frames, -1, dtype=int)
    cluster_indices = {}
    for k, v in clusters.items():
        cluster_indices[k] = np.array([frame_index[np.asarray(frame).tobytes()].pop(0) for frame in v])
        labels[cluster_indices[k]] = k

    # Extract best n structures for each cluster
    best_frames_indices = []
    for k, cluster in cluster_indices.items():
        if len(cluster) > 1:
            medoid_index = calculate_medoid(traj_numpy[cluster], metric=metric, N_atoms=N_atoms)
            medoid = traj_numpy[cluster][medoid_index]
            msd_to_medoid = [extended_comparison(np.array([frame, medoid]), data_type='full', metric=metric,
                                                 N_atoms=N_atoms) for frame in traj_numpy[cluster]]
            best = cluster[np.argsort(msd_to_medoid)[:n_structures]]
        else:
            best = cluster
        for frame in best:
            best_frames_indices.append((frame * sieve, k))
    np.savetxt(f'{output_dir}/equal_best_frames_{threshold}.csv', best_frames_indices, delimiter=',', fmt='%d',
               header=f'{params}, n_structures: {n_structures}\nFrame Index, Cluster Index')

    # Save cluster labels (frames not assigned to any cluster are left out)
    frame_vs_cluster = np.column_stack((np.arange(total_frames) * sieve, labels))
    frame_vs_cluster = frame_vs_cluster[labels >= 0]
    np.savetxt(f'{output_dir}/equal_labels_{threshold}.csv', frame_vs_cluster, delimiter=',', fmt='%d',
               header=f'{params}\nFrame Index, Cluster Index')

    # Calculate population and MSD of each cluster
    with open(f'{output_dir}/equal_summary_{threshold}.csv', 'w') as f:
        f.write(f'# {params}, total frames: {total_frames}\n')
        f.write('# Cluster Index, Fraction of total frames, Number of frames, MSD\n')
        for i in np.unique(labels[labels >= 0]):
            members = traj_numpy[labels == i]
            msd = extended_comparison(members, data_type='full', metric=metric, N_atoms=N_atoms)
            f.write(f'{i},{len(members) / total_frames},{len(members)},{msd}\n')
