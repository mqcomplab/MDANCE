import os
import time
import sys
import numpy as np
from mdance import data
from mdance.cluster.divine import Divine
from mdance.outputs.plot import plot_scores


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
N_atoms = 50                                        # Number of atoms in the system
sieve = 1                                           # Use every sieve-th frame (1 = all frames)

# DIVINE parameters
split = 'weighted_MSD'                              # Cluster to split next {'MSD', 'radius', 'weighted_MSD'}
anchors = 'nani'                                    # Seed choice for each split {'nani', 'outlier_pair', 'splinter_split'}
init_type = 'strat_all'                             # NANI seeding method, used when anchors = 'nani'

k = 30                                              # Keep splitting until there are k clusters
threshold = 0                                       # Minimum subcluster size, as a fraction of the data
refine = True                                       # Refine each split with k-means
output_dir = 'outputs'                              # Folder for output files

# Analysis parameters
plot_metrics = True                                 # Plot DBI and CHI, and suggest values
x_min = 5                                           # Only suggest values >= x_min (None = all)


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    full_traj = np.load(input_traj_numpy)
    selected_indices = np.arange(len(full_traj))[::sieve]
    traj_numpy = full_traj[::sieve]

    mod = Divine(data=traj_numpy, split=split, anchors=anchors, init_type=init_type,
                end='k', k=k, N_atoms=N_atoms, threshold=threshold, refine=refine)
    clusters, labels, scores, msds = mod.run()

    output_data = np.column_stack((selected_indices, labels))

    params = f'DIVINE, split: {split}, anchors: {anchors}, init_type: {init_type}, k: {k}, sieve: {sieve}'
    np.savetxt(f"{output_dir}/divine_{anchors}_msds.csv", msds, fmt="%d,%d,%d,%.5f", delimiter=",",
               header=f"{params}\nLevel, Cluster Index, Number of frames, MSD")

    # Level L of the hierarchy has L + 1 clusters
    msds = np.asarray(msds)
    scores_csv = f"{output_dir}/divine_{anchors}_scores.csv"
    with open(scores_csv, 'w') as f:
        f.write(f'# {params}\n')
        f.write('# Number of clusters, Calinski-Harabasz, Davies-Bouldin, Average MSD\n')
        for n, ch, db in scores:
            avg_msd = np.mean(msds[msds[:, 0] == n - 1, 3])
            f.write(f'{int(n)},{ch},{db},{avg_msd}\n')

    # Plot the DBI and CHI of the screening and suggest values
    if plot_metrics:
        plot_scores(scores_csv, x_min=x_min)
