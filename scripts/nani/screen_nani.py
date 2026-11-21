import os

import numpy as np

from mdance.cluster.nani import KmeansNANI, compute_scores
from mdance import data
from mdance.tools.bts import extended_comparison
from mdance.outputs.plot import plot_scores


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
N_atoms = 50                                        # Number of atoms in the system
sieve = 1                                           # Use every sieve-th frame (1 = all frames)

# NANI parameters
output_dir = 'outputs'                              # Folder for output files
init_type = 'strat_all'                             # Seeding method {'strat_all', 'strat_reduced', 'comp_sim', 'div_select', 'k-means++', 'random'}
metric = 'MSD'                                      # Distance metric for clustering
algorithm = 'kmeans'                               # 'kmeans' or 'emeans' (explicit NANI seeds only)
emeans_epochs = 5                                   # Used only for algorithm='emeans'
start_n_clusters = 5                                # At least 2 clusters
end_n_clusters = 30                                 # Maximum number of clusters

# Analysis parameters
plot_metrics = True                                 # Plot DBI and CHI, and suggest values
x_min = None                                        # Only suggest values >= x_min (None = all)


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    traj_numpy = np.load(input_traj_numpy)[::sieve]
    if init_type in ['k-means++', 'random', 'vanilla_kmeans++']:
        percentage = ''

    # `comp_sim` and `div_select` are ran only once to get the initiators
    elif init_type in ['comp_sim', 'div_select', 'strat_all', 'strat_reduced']:
        percentage = 10
        mod = KmeansNANI(data=traj_numpy, n_clusters=end_n_clusters, metric=metric,
                         N_atoms=N_atoms, init_type=init_type, percentage=percentage)
        initiators = mod.initiate_kmeans()

    all_scores = []
    for i in range(start_n_clusters, end_n_clusters+1):
        total = 0

        # Run k-means clustering
        if init_type in ['comp_sim', 'div_select', 'strat_all', 'strat_reduced']:
            mod = KmeansNANI(data=traj_numpy, n_clusters=i, metric=metric, N_atoms=N_atoms,
                             init_type=init_type, percentage=percentage)
            labels, centers, n_iter = mod.kmeans_clustering(initiators[:i], algorithm=algorithm,
                                                           emeans_epochs=emeans_epochs)
        elif init_type in ['k-means++', 'random']:
            mod = KmeansNANI(data=traj_numpy, n_clusters=i, metric=metric,
                             N_atoms=N_atoms, init_type=init_type)
            labels, centers, n_iter = mod.kmeans_clustering(initiators=init_type, algorithm=algorithm,
                                                           emeans_epochs=emeans_epochs)
        elif init_type == 'vanilla_kmeans++':
            mod = KmeansNANI(data=traj_numpy, n_clusters=i, metric=metric,
                             N_atoms=N_atoms, init_type=init_type)
            initiators = mod.initiate_kmeans()
            labels, centers, n_iter = mod.kmeans_clustering(initiators=initiators, algorithm=algorithm,
                                                           emeans_epochs=emeans_epochs)

        # Compute scores
        ch_score, db_score = compute_scores(data=traj_numpy, labels=labels)

        # Calculate MSD for each cluster
        dict = {}
        for j in range(i):
            dict[j] = np.where(labels == j)[0]
            dict[j] = traj_numpy[dict[j]]
        for key in dict:
            msd = extended_comparison(np.array(dict[key]), data_type='full',
                                        metric=metric, N_atoms=N_atoms)
            total += msd
        all_scores.append((i, ch_score, db_score, total/i, n_iter))

    suffix = '' if algorithm == 'kmeans' else f'_{algorithm}'
    scores_csv = f'{output_dir}/nani_{init_type}{suffix}_scores.csv'
    with open(scores_csv, 'w') as f:
        f.write(f'# NANI, init_type: {init_type}, percentage: {percentage}, metric: {metric}, sieve: {sieve}\n')
        count_label = 'Number of iterations' if algorithm == 'kmeans' else 'E-means epochs'
        f.write(f'# Number of clusters, Calinski-Harabasz, Davies-Bouldin, Average MSD, {count_label}\n')
        for row in all_scores:
            f.write(','.join(map(str, row)) + '\n')

    # Plot the DBI and CHI of the screening and suggest values
    if plot_metrics:
        plot_scores(scores_csv, x_min=x_min)
