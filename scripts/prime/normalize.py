import os

import numpy as np

from mdance import data
from mdance.inputs.preprocess import normalize_file, Normalizer


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
cluster_labels = data.labels_6                      # Labels file from a clustering script
output_dir = 'outputs/normed_clusters'              # Folder for the normalized cluster files
output_base_name = 'normed_clusttraj'               # Prefix for the normalized cluster files
n_clusters = 6                                      # Number of clusters in the labels file


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    traj_numpy = np.load(input_traj_numpy)
    frames, labels = np.loadtxt(cluster_labels, delimiter=',', dtype=int, unpack=True)

    # Clusters are numbered by size, so the first n_clusters are the most populated
    all_clusttraj = []
    for i in range(n_clusters):
        all_clusttraj.append(traj_numpy[frames[labels == i]])
    concat_clusttraj = np.concatenate(all_clusttraj)
    normed_data, min, max, avg = normalize_file(concat_clusttraj, norm_type='v3')
    np.save(f'{output_dir}/normed_data.npy', normed_data)
    for i, traj in enumerate(all_clusttraj):
        norm = Normalizer(data=traj, custom_min=min, custom_max=max)
        normed_frame = norm.get_v3_norm()
        np.save(f'{output_dir}/{output_base_name}.c{i}.npy', normed_frame)
