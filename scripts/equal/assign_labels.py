import pickle as pkl

import numpy as np

from mdance import data
from mdance.cluster.equal import ExtendedQuality
from mdance.tools.bts import extended_comparison, calculate_medoid


# System info - EDIT THESE
input_traj_numpy = data.sim_traj_numpy
N_atoms = 50
sieve = 1

# eQUAL params - EDIT THESE
metric = 'MSD'                                                      # Default 
n_seeds = 3                                                         # Default
check_sim = True                                                    # Default
reject_lowd = True                                                  # Default
sim_threshold = 16
min_samples = 10                                                    # Default

# extract params- EDIT THESE
threshold = 5.80
n_structures = 11                                                   # Default
sorted_by = 'frame'                                                 # Default
open_clusters = None                                                # Default


if __name__ == '__main__':
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
    
    # Extract best n structures for each cluster
    best_frames = []
    for k, v in clusters.items():
        medoid_index = calculate_medoid(np.array(v), metric=metric, N_atoms=N_atoms)
        medoid = v[medoid_index]
        msd_to_medoid = []
        for i, frame in enumerate(v):
            msd_to_medoid.append((i, extended_comparison(np.array([frame, medoid]), data_type='full', 
                                                         metric=metric, N_atoms=N_atoms)))
        msd_to_medoid = np.array(msd_to_medoid)
        sorted_indices = np.argsort(msd_to_medoid[:, 1])
        best_n_structures = [v[idx] for idx in sorted_indices[:n_structures]]
        best_frames.append(best_n_structures)
    
    # Save best n structures for each cluster to file
    best_frames_indices = []
    for i, frame in enumerate(traj_numpy):
        i = i * sieve
        for k, v in enumerate(best_frames):
            if any((frame == x).all() for x in v):
                best_frames_indices.append((i, k))
                break
    best_frames_indices = np.array(best_frames_indices)
    best_frames_indices = best_frames_indices[best_frames_indices[:, 1].argsort()]
    np.savetxt(f'best_frames_indices_{threshold}.csv', best_frames_indices, 
               header=f'eQual,threshold,{threshold},n_clusters,{len(clusters)}\nframe,cluster', 
               delimiter=',', fmt='%s')
    
    # Save cluster assignments to file
    frame_vs_cluster = []
    for i, frame in enumerate(traj_numpy):
        i = i * sieve
        for k, v in clusters.items():
            if any((frame == x).all() for x in v):
                frame_vs_cluster.append((i, k))
                break
    frame_vs_cluster = np.array(frame_vs_cluster)
    if sorted_by == 'cluster':
        frame_vs_cluster = frame_vs_cluster[frame_vs_cluster[:, 1].argsort()]
    np.savetxt(f'cluster_labels_{threshold}.csv', frame_vs_cluster, 
               header=f'eQual,threshold,{threshold},n_clusters,{len(clusters)}\nframe,cluster', 
               delimiter=',', fmt='%s')