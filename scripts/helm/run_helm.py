import os
import pickle as pkl

import numpy as np

from mdance import data
from mdance.cluster.helm import HELM, compute_scores
from mdance.outputs.plot import plot_scores
from mdance.tools.bts import extended_comparison


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
cluster_labels = data.labels_60                     # Labels of the initial clusters (e.g. from NANI)
sieve = 1                                           # Use every sieve-th frame (1 = all frames)
N_atoms = 50                                        # Number of atoms in the system

# HELM parameters
metric = 'MSD'                                      # Distance metric for clustering
merge_scheme = 'intra'                              # Merge scheme {'inter', 'intra', 'half'}
link = None                                         # Linkage method {None, 'ward'}
n_clusters = 1                                      # Either n_clusters or eps must be set
eps = None                                          # Either n_clusters or eps must be set
trim_start = False                                  # Remove the least compact initial clusters first
trim_val = None                                     # Either trim_val or trim_k must be set
trim_k = 5                                          # Either trim_val or trim_k must be set
save_pairwise_sim = False                           # Save the pairwise similarity matrix
output_dir = 'outputs'                              # Folder for output files

# Analysis parameters
plot_metrics = True                                 # Plot DBI and CHI, and suggest values
x_min = 5                                           # Only suggest values >= x_min (None = all)


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    traj_numpy = np.load(input_traj_numpy)[::sieve]
    frames, labels = np.loadtxt(cluster_labels, dtype=int, unpack=True, delimiter=',')
    # Keep frames that are in the sieved trajectory and skip unassigned frames (-1)
    keep = (frames % sieve == 0) & (labels >= 0)
    frames, labels = frames[keep] // sieve, labels[keep]
    unique_labels = np.sort(np.unique(labels))

    # Extract cluster data from trajectory
    input_clusters = []
    for i, each_unique in enumerate(unique_labels):
        input_clusters.append(traj_numpy[frames[labels == each_unique]])
    N0 = len(input_clusters)
    print(f'Number of clusters: {len(input_clusters)}')

    # Build cluster dictionary to pass to HELM
    cluster_dict = {N0: []}
    for i, d in enumerate(input_clusters):
        Indicesik = [i]
        Nik = len(input_clusters[i])
        c_sumik = np.sum(input_clusters[i], axis=0)
        sq_sumik = np.sum(input_clusters[i]**2, axis=0)
        cluster_dict[N0].append([Indicesik, (c_sumik, sq_sumik), Nik])

    # Run HELM
    helm = HELM(cluster_dict=cluster_dict, metric=metric, N_atoms=N_atoms,
                merge_scheme=merge_scheme, link=link, n_clusters=n_clusters, eps=eps,
                trim_start=trim_start, trim_val=trim_val, trim_k=trim_k,
                save_pairwise_sim=save_pairwise_sim
                )()

    with open(f'{output_dir}/helm_{link or merge_scheme}.pkl', 'wb') as outfile:
        pkl.dump(helm, outfile)

    # Compute CH, DB and average MSD for each number of clusters
    scores = []
    N0 = len(helm.values())
    for key, value in enumerate(helm.values()):
        input_cluster_indices = []
        for i in value:
            input_cluster_indices += i[0]
        arr = np.concatenate([input_clusters[i] for i in input_cluster_indices])
        ch, db = compute_scores(value, arr)
        msds = [extended_comparison((c_sum, sq_sum), data_type='condensed', metric=metric, N=Nik, N_atoms=N_atoms)
                for _, (c_sum, sq_sum), Nik in value]
        scores.append([key, ch, db, np.mean(msds)])

    # remove None values from scores
    scores = [i for i in scores if i[1] is not None]

    scores_csv = f'{output_dir}/helm_{link or merge_scheme}_scores.csv'
    with open(scores_csv, 'w') as f:
        f.write(f'# HELM, merge_scheme: {merge_scheme}, link: {link}, metric: {metric}, sieve: {sieve}\n')
        f.write('# Number of clusters, Calinski-Harabasz, Davies-Bouldin, Average MSD\n')
        for i in scores:
            f.write(f'{N0 - i[0]},{i[1]},{i[2]},{i[3]}\n')

    # Plot the DBI and CHI of the screening and suggest values
    if plot_metrics:
        plot_scores(scores_csv, x_min=x_min)
