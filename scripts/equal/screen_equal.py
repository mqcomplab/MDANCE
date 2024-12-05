import os
import pickle as pkl

import numpy as np

from mdance import data
from mdance.cluster.equal import ExtendedQuality, compute_scores


# System info - EDIT THESE
input_traj_numpy = data.sim_traj_numpy
N_atoms = 50
sieve = 1

# eQUAL params - EDIT THESE
metric = 'MSD'                                                      # Default
n_seeds = 3
check_sim = True                                                    # Default
reject_lowd = True                                                  # Default
sim_threshold = 16
min_samples = 10                                                    # Default

# thresholds params- EDIT THESE
start_threshold = 5
end_threshold = 6
step = 0.1
save_clusters = False                                                # Default False


if __name__ == '__main__':
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
        # Run eQual for each threshold
        eQ = ExtendedQuality(data=traj_numpy, threshold=threshold, 
                             metric=metric, N_atoms=N_atoms, n_seeds=n_seeds, 
                             check_sim=check_sim, reject_lowd=reject_lowd,
                             sim_threshold=sim_threshold, min_samples=min_samples)
        clusters = eQ.run()
        top_pops.append(eQ.calculate_populations(clusters))
        ch, db = compute_scores(clusters)
        scores.append([threshold, ch, db])
        
        # Option to save clusters
        if save_clusters:
            if not os.path.exists('clusters_dict'):
                os.mkdir('clusters_dict')
            with open(f'clusters_dict/equal_threshold_{threshold}.pkl', 'wb') as f:
                pkl.dump(clusters, f)
    scores = [i for i in scores if i[1] is not None]
    
    header = f'#eQual Cluster Screening Summary\n'
    p_header = '#threshold,n_clusters,top_10_combined,top1,top2,top3,top4,top5,top6,top7,top8,top9,top10'
    
    with open(f'equal_screening.csv', 'w') as infile:
        infile.write(header + p_header + '\n')
        for line in top_pops:
            infile.write(','.join(map(str, line)) + '\n')

    # Write CH and DB scores for each threshold
    with open(f'equal_screening_scores.csv', 'w') as f:
        f.write(header + '#threshold,Calinski-Harabasz,Davies-Bouldin\n')
        for line in scores:
            f.write(','.join(map(str, line)) + '\n')