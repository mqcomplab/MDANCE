import pickle as pkl

import numpy as np

from mdance import data
from mdance.cluster.helm import HELM, compute_scores


# System info - EDIT THESE
input_traj_numpy = data.sim_traj_numpy
cluster_labels = data.labels_60
sieve = 1
N_atoms = 50                                    # Number of atoms in the system

# HELM params - EDIT THESE
metric = 'MSD'                                  # Default  
merge_scheme = 'intra'                          # {'inter', 'intra'}
link = 'ward'
n_clusters = 1                                  # Either n_clusters or eps must be set
eps = None                                      # Either n_clusters or eps must be set
trim_start = False                              # Default
trim_val = None                                 # Either trim_val or trim_k must be set
trim_k = 5                                     # Either trim_val or trim_k must be set
save_pairwise_sim = False                       # Default
align_method = None                            # Default


if __name__ == '__main__':
    traj_numpy = np.load(input_traj_numpy)[::sieve]
    frames, labels = np.loadtxt(cluster_labels, dtype=int, unpack=True, delimiter=',')
    # divide sieve by 10 to get the correct frame number
    frames = frames // sieve
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
        if align_method:
            cluster_dict[N0].append([Indicesik, (c_sumik, sq_sumik), Nik, 
                                     input_clusters[i]])
        else:
            cluster_dict[N0].append([Indicesik, (c_sumik, sq_sumik), Nik])
    
    # Run HELM
    helm = HELM(cluster_dict=cluster_dict, metric=metric, N_atoms=N_atoms, 
                merge_scheme=merge_scheme, link=link, n_clusters=n_clusters, 
                trim_start=trim_start, trim_val=trim_val, trim_k=trim_k,
                align_method=align_method, save_pairwise_sim=save_pairwise_sim
                )()

    with open(f'{link}-helm.pkl', 'wb') as outfile:
        pkl.dump(helm, outfile)

    # Compute CH and DB scores
    scores = []
    N0 = len(helm.values())
    for key, value in enumerate(helm.values()):
        input_cluster_indices = []
        for i in value:
            input_cluster_indices += i[0]
        arr = np.concatenate([input_clusters[i] for i in input_cluster_indices])
        ch, db = compute_scores(value, arr)
        scores.append([key, ch, db])
    
    # remove None values from scores
    scores = [i for i in scores if i[1] is not None]

    with open(f'{link}_helm_scores.csv', 'w') as f:
        f.write('#n_clusters,CH,DB\n')
        for i in scores:
            f.write(f'{N0 - i[0]},{i[1]},{i[2]}\n')