import numpy as np
import pickle as pkl

from mdance import data
from mdance.tools.bts import calculate_medoid, extended_comparison


# System info - EDIT THESE
input_traj_numpy = data.sim_traj_numpy
N_atoms = 50
sieve = 1

# HELM params - EDIT THESE
n_clusters = 10
pre_cluster_labels = data.labels_60
pickle_file = 'intra-helm.pkl'
metric = 'MSD'                                                      # Default
extract_type = 'top'                                                # Default
n_structures = 11                                                   # Default


if __name__ == '__main__':
    # Load data and cluster labels
    traj_numpy = np.load(input_traj_numpy)[::sieve]
    total_frames = len(traj_numpy)
    with open(pickle_file, 'rb') as f:
        cluster_dict = pkl.load(f)
    all_indices = [i[0] for i in cluster_dict[n_clusters]]; print(f'cluster indices: {all_indices}')
    pre_frames, pre_labels = np.loadtxt(pre_cluster_labels, unpack=True, dtype=int, delimiter=',')
    pre_frames = pre_frames // sieve
    
    # Load cluster info based on pre_labels
    pre_clusters = []
    unique_labels = np.unique(pre_labels)
    for i, label in enumerate(unique_labels):
        pre_clusters.append(traj_numpy[pre_frames[pre_labels == label]])
    
    # Sort clusters by population
    cluster_indices = cluster_dict[n_clusters]
    unsorted_indices = []
    for i, index in enumerate(cluster_indices):
        population = index[2]   # Nik is number of frames in cluster
        unsorted_indices.append((i, population))
    sorted_indices = sorted(unsorted_indices, key=lambda x: x[1], reverse=True)
    
    # Save cluster data and medoids
    all_medoids = []
    individual_frac = []
    frame_indices_all = {}
    best_frame_indices = []
    for i, index in enumerate(sorted_indices):
        frame_indices_all[i] = []
        cluster_dict[i] = cluster_indices[index[0]]
        pre_clus_indices = cluster_dict[i][0]
        individual_frac.append('%.6f' % (cluster_dict[i][2] / total_frames))
        
        # Find the corresponding frame indices in the original data
        for j, label in enumerate(pre_labels):
            for index in pre_clus_indices:
                if label == index:
                    frame_indices_all[i].append(pre_frames[j])
    
    for key in frame_indices_all:
        # Concatenate data that corresponds to the cluster_dict[i] (pre_clus_indices)
        frames = np.stack([traj_numpy[j] for j in frame_indices_all[key]])
        if len(frames) > 2:
            medoid_index = calculate_medoid(frames, metric=metric, N_atoms=N_atoms)
            all_medoids.append((i, medoid_index))
            
            # Find the closest frame to the medoid in the pre_clusters
            medoid = frames[medoid_index]
            msd_to_medoid = []
            for j, frame in enumerate(frames):
                msd_to_medoid.append((j, extended_comparison(np.array([frame, medoid]), data_type='full', 
                                                             metric=metric, N_atoms=N_atoms)))
            msd_to_medoid = np.array(msd_to_medoid)
            sorted = np.argsort(msd_to_medoid[:, 1])[:n_structures]
            best = np.array(frame_indices_all[key])[sorted]
            best_frame_indices.append(best)
        else:
            best_frame_indices.append(None)
    
    
    # Save cluster labels
    with open(f'helm_cluster_labels_{n_clusters}.csv', 'w') as f:
        f.write(f'# Helm,number of clusters,{n_clusters}\n')
        f.write('# frame_index,cluster_index\n')
        for cluster in frame_indices_all.keys():
            for i, frame in enumerate(frame_indices_all[cluster]):       
                f.write(f'{frame * sieve},{cluster}\n')
         
    # Save best frame indices for each cluster
    with open(f'helm_best_frames_indices_{n_clusters}.csv', 'w') as f:
        f.write(f'# Helm,number of clusters,{n_clusters}\n')
        f.write('# frame_index,cluster_index\n')
        for i, cluster in enumerate(best_frame_indices):
            for frame in cluster:
                f.write(f'{frame * sieve},{i}\n')
    
    # Population of top 10 clusters
    top_10_frac = sum(float(i) for i in individual_frac[:10])
    with open(f'helm_summary_{n_clusters}.csv', 'w') as f:
        f.write(f'# Population Summary\n# top 10 frac,{top_10_frac},number of clusters,{n_clusters}\n')
        f.write('# cluster index,frac of total frames\n')
        for i, frac in enumerate(individual_frac):
            f.write(f'{i},{frac}\n')