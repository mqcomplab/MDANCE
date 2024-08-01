import os

import numpy as np

from mdance.cluster.nani import KmeansNANI
from mdance import data
from mdance.tools.bts import extended_comparison, calculate_medoid


# System info - EDIT THESE
input_traj_numpy = data.sim_traj_numpy
N_atoms = 50
sieve = 1

# K-means params - EDIT THESE
n_clusters = 6
init_type = 'comp_sim'                                              # Default
metric = 'MSD'                                                      # Default
n_structures = 11                                                   # Default
output_dir = 'outputs'                                              # Default


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    traj_numpy = np.load(input_traj_numpy)[::sieve]
    mod = KmeansNANI(data=traj_numpy, n_clusters=n_clusters, N_atoms=N_atoms, init_type=init_type, 
                     metric=metric, percentage=10)
    labels, centers, n_iter = mod.execute_kmeans_all()
    sort_labels_by_size = np.argsort(np.bincount(labels))[::-1]
    labels = np.array([np.where(sort_labels_by_size == i)[0][0] for i in labels])
    best_frames = []
    cluster_msd = []

    # Save best frames indices for each cluster
    for i, label in enumerate(np.unique(labels)):
        cluster = np.where(labels == label)[0]
        if len(cluster) > 1:
            medoid_index = calculate_medoid(traj_numpy[cluster], metric=metric, N_atoms=N_atoms)
            medoid = traj_numpy[cluster][medoid_index]
            msd_to_medoid = []
            for j, frame in enumerate(traj_numpy[cluster]):
                msd_to_medoid.append((j, extended_comparison(
                    np.array([frame, medoid]), data_type='full', metric=metric, N_atoms=N_atoms)))
            msd_to_medoid = np.array(msd_to_medoid)
            sorted_indices = np.argsort(msd_to_medoid[:, 1])
            best_n_structures = traj_numpy[cluster][sorted_indices[:n_structures]]
            best_frames.append(best_n_structures)
    
    best_frames_indices = []
    for i, frame in enumerate(traj_numpy):
        i = i * sieve
        for j, cluster in enumerate(best_frames):
            if np.any(np.all(cluster == frame, axis=1)):
                best_frames_indices.append((i, j))
    best_frames_indices = np.array(best_frames_indices)
    best_frames_indices = best_frames_indices[best_frames_indices[:, 1].argsort()]
    np.savetxt(f'{output_dir}/best_frames_indices_{n_clusters}.csv', best_frames_indices, delimiter=',', fmt='%s', 
               header=f'Numer of clusters,{n_clusters}\nFrame Index,Cluster Index')
    
    # Save cluster labels
    with open(f'{output_dir}/labels_{n_clusters}.csv', 'w') as f:
        f.write(f'# init_type: {init_type}, Number of clusters: {n_clusters}\n')
        f.write('# Frame Index, Cluster Index\n')
        for i, row in enumerate(labels):
            f.write(f'{i * sieve},{row}\n')
    
    # Calculate population of each cluster
    with open(f'{output_dir}/summary_{n_clusters}.csv', 'w') as f:
        f.write(f'# Number of clusters, {n_clusters}\n')
        f.write('# Cluster Index, Fraction out of total pixels\n')
        for i, row in enumerate(np.bincount(labels)):
            f.write(f'{i},{row/len(labels)}\n')