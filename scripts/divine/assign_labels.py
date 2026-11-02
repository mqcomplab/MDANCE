import os
import time
import sys
import numpy as np
from mdance import data
from mdance.cluster.divine import Divine
from mdance.tools.bts import extended_comparison, calculate_medoid


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
N_atoms = 50                                        # Number of atoms in the system
sieve = 1                                           # Use every sieve-th frame (1 = all frames)

# DIVINE parameters
split = 'weighted_MSD'                              # Cluster to split next {'MSD', 'radius', 'weighted_MSD'}
anchors = 'nani'                                    # Seed choice for each split {'nani', 'outlier_pair', 'splinter_split'}
init_type = 'strat_all'                             # NANI seeding method, used when anchors = 'nani'

k = 6                                               # Number of clusters to assign
threshold = 0                                       # Minimum subcluster size, as a fraction of the data
refine = True                                       # Refine each split with k-means

top_n = 11                                          # Frames saved per cluster, closest to its medoid
output_dir = 'outputs'                              # Folder for output files


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    full_traj = np.load(input_traj_numpy)
    selected_indices = np.arange(len(full_traj))[::sieve]
    traj_numpy = full_traj[selected_indices]

    mod = Divine(data=traj_numpy, split=split, anchors=anchors, init_type=init_type,
                end='k', k=k, N_atoms=N_atoms, threshold=threshold, refine=refine)

    clusters, labels, scores, msds = mod.run()

    # Renumber the clusters by size, largest first
    order = sorted(range(len(clusters)), key=lambda c: len(clusters[c]), reverse=True)
    clusters = [clusters[c] for c in order]
    new_label = np.empty(len(order), dtype=int)
    new_label[order] = np.arange(len(order))
    labels = new_label[labels]
    params = f'DIVINE, split: {split}, anchors: {anchors}, init_type: {init_type}, k: {k}, sieve: {sieve}'

    output_data = np.column_stack((selected_indices, labels))
    np.savetxt(f"{output_dir}/divine_labels_{k}.csv", output_data, delimiter=",",
               header=f"{params}\nFrame Index, Cluster Index", fmt="%d")

    best_frames_list = []

    for cid, cluster in enumerate(clusters):

        cluster_data = traj_numpy[cluster]
        medoid_rel_idx = calculate_medoid(cluster_data, metric="MSD", N_atoms=N_atoms)
        medoid_frame = cluster_data[medoid_rel_idx]

        distances = [extended_comparison(np.array([frame, medoid_frame]), data_type='full', metric="MSD", N_atoms=N_atoms) for frame in cluster_data]
        sorted_indices = np.argsort(distances)[:top_n]
        best_frames = cluster[sorted_indices]

        best_frames_indices = np.column_stack((selected_indices[best_frames], [cid] * len(best_frames)))
        best_frames_list.append(best_frames_indices)

    best_frames_indices = np.vstack(best_frames_list)

    np.savetxt(f"{output_dir}/divine_best_frames_{k}.csv", best_frames_indices, delimiter=',',
                fmt='%d', header=f"{params}, top_n: {top_n}\nFrame Index, Cluster Index")

    # Calculate population and MSD of each cluster
    with open(f'{output_dir}/divine_summary_{k}.csv', 'w') as f:
        f.write(f'# {params}, total frames: {len(labels)}\n')
        f.write('# Cluster Index, Fraction of total frames, Number of frames, MSD\n')
        for i in np.unique(labels[labels >= 0]):
            members = traj_numpy[labels == i]
            msd = extended_comparison(members, data_type='full', metric='MSD', N_atoms=N_atoms)
            f.write(f'{i},{len(members) / len(labels)},{len(members)},{msd}\n')
