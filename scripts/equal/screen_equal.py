import os
import pickle as pkl

import numpy as np

from mdance import data
from mdance.cluster.equal import ExtendedQuality, compute_scores
from mdance.tools.bts import extended_comparison
from mdance.outputs.plot import plot_n_clusters, plot_scores


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

# Screening parameters
start_threshold = 5                                 # First threshold to screen
end_threshold = 6                                   # Last threshold to screen
step = 0.1                                          # Step between thresholds
save_clusters = False                               # Save the clusters of each threshold as .pkl
output_dir = 'outputs'                              # Folder for output files

# Analysis parameters
plot_metrics = True                                 # Plot DBI and CHI, and suggest values
x_min = None                                        # Only suggest values >= x_min (None = all)


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    # float sieve will divide the total frames by the sieve value and randomly select that many frames
    if isinstance(sieve, float):
        traj_numpy = np.load(input_traj_numpy)
        total_frames = len(traj_numpy)
        total_indices = total_frames / sieve
        random_indices = np.random.choice(total_frames, int(total_indices), replace=False)
        traj_numpy = traj_numpy[random_indices]
        print(f'Using {len(traj_numpy)} frames.')

    # int sieve will select every sieve-th frame
    else:
        traj_numpy = np.load(input_traj_numpy)[::sieve]
        total_frames = len(traj_numpy)

    top_pops = []
    scores = []

    for threshold in np.arange(start_threshold, end_threshold, step):
        threshold = round(float(threshold), 10)     # e.g. 5.199999999999999 -> 5.2
        # Run eQual for each threshold
        eQ = ExtendedQuality(data=traj_numpy, threshold=threshold,
                             metric=metric, N_atoms=N_atoms, n_seeds=n_seeds,
                             check_sim=check_sim, reject_lowd=reject_lowd,
                             sim_threshold=sim_threshold, min_samples=min_samples)
        clusters = eQ.run()
        top_pops.append(eQ.calculate_populations(clusters))
        ch, db = compute_scores(clusters)
        msds = [extended_comparison(np.array(v), data_type='full', metric=metric, N_atoms=N_atoms)
                for v in clusters.values()]
        avg_msd = np.mean(msds) if msds else None
        scores.append([threshold, len(clusters), ch, db, avg_msd])

        # Option to save clusters
        if save_clusters:
            os.makedirs(f'{output_dir}/clusters_dict', exist_ok=True)
            with open(f'{output_dir}/clusters_dict/equal_threshold_{threshold}.pkl', 'wb') as f:
                pkl.dump(clusters, f)
    scores = [i for i in scores if i[2] is not None]

    header = (f'# eQual screening, metric: {metric}, n_seeds: {n_seeds}, check_sim: {check_sim}, '
              f'sim_threshold: {sim_threshold}, reject_lowd: {reject_lowd}, min_samples: {min_samples}, sieve: {sieve}\n')
    p_header = '# Threshold, Number of clusters, Top 10 combined, ' + ', '.join(f'Top {i}' for i in range(1, 11))

    with open(f'{output_dir}/equal_screening.csv', 'w') as infile:
        infile.write(header + p_header + '\n')
        for line in top_pops:
            infile.write(','.join(map(str, line)) + '\n')

    # Write CH, DB and average MSD for each threshold
    with open(f'{output_dir}/equal_scores.csv', 'w') as f:
        f.write(header + '# Threshold, Number of clusters, Calinski-Harabasz, Davies-Bouldin, Average MSD\n')
        for line in scores:
            f.write(','.join(map(str, line)) + '\n')

    # Plot the DBI, CHI and number of clusters of the screening and suggest values
    if plot_metrics:
        plot_scores(f'{output_dir}/equal_scores.csv', x_min=x_min)
        plot_n_clusters(f'{output_dir}/equal_screening.csv')
