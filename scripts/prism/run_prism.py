import glob

import matplotlib.pyplot as plt
import numpy as np
import time

from mdance.cluster.prism import PRISM


# System Parameters
file_pattern = 'ala/data/*.csv'     # Pattern pointing to trajectory files (each file = one pathway)
metric = 'MSD'

# PRISM Parameters
option = 2                          # Medoid construction: 1=global, 2=per-traj, 3=two-stage
k = 20                               # Number of clusters for representative medoid construction
k_final = k                         # Second-stage cluster count (used only if option == 3)
weight_scheme = 'weighted_avg'      # Average Hausdorff normalization scheme      

t = 2                                        
criterion = 'maxclust'
linkage_method = 'ward'


if __name__ == '__main__':
    frames_all = []      

    """
    Modify parsing below based on your file structure and data format.

    Each trajectory must be stored as:
        (unique_id, 2D numpy array)

    Requirements:
    - frame_array must be 2D with shape (n_frames, n_features)
    - All trajectories must have the same number of features (columns)
    """     
    for file in glob.glob(file_pattern, recursive=True):

        traj = file.split('_')[-1].split('.')[0]
        frame = np.genfromtxt(file, delimiter=',')
        frames_all.append((traj, frame))

    """for file in glob.glob(file_pattern, recursive=True):
        traj = file.split('/')[-1].split('.')[0] 
        frame = np.load(file)
        frames_all.append((traj, frame))"""

    # Run PRISM
    mod = PRISM(
        frames_all,
        metric,
        t=t,
        criterion=criterion,
        link=linkage_method,
        option=option,
        k=k,
        k_final=k_final,
        weight_scheme=weight_scheme,
    )

    link, clusters = mod.run()

    suffix = f'prism_opt{option}_k{k}_kfinal{k_final}'

    # Plot the dendrogram
    ax = mod.plot()
    plt.savefig(f'{suffix}.png', bbox_inches='tight', dpi=500, pad_inches=0.1)
    plt.close()

    # Save medoids per trajectory
    np.save(f'{suffix}_medoids.npy', mod.medoids_per_traj, allow_pickle=True)
