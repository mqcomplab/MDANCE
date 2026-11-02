import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from mdance import data
from mdance.cluster.shine import Shine


# System parameters
file_pattern = str(data.ala_pathways / '*.csv')     # One file per pathway; the last number in the name is the pathway index
metric = 'MSD'                                      # Distance metric for clustering
N_atoms = 1                                         # Number of atoms in the system

# SHINE parameters
sampling =  None                                    # Sampling scheme {'diversity', 'quota', None}
frac = 1                                            # Fraction of frames to be sampled
merge_scheme = 'intra'                              # Merge scheme {'intra', 'inter', 'semi_sum', 'max', 'min', 'haus'}
t = 2                                               # Maximum number of clusters
output_dir = 'outputs'                              # Folder for output files


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    frames_all = []

    # Change this to read the data from different file patterns.
    for file in glob.glob(file_pattern):
        traj = file.split('_')[-1].split('.')[0]
        frame = np.load(file) if file.endswith('.npy') else np.genfromtxt(file, delimiter=',')
        frames_all.append((traj, frame))

    # Run SHINE
    mod = Shine(frames_all, metric, N_atoms=N_atoms, t=t, criterion='maxclust',
                link='ward', merge_scheme=merge_scheme,
                sampling=sampling, frac=frac)
    link, clusters = mod.run()

    # Save cluster assignments of the pathways
    params = (f'SHINE, merge_scheme: {merge_scheme}, sampling: {sampling}, frac: {frac}, t: {t}, '
              f'metric: {metric}, N_atoms: {N_atoms}')
    assignments = sorted((int(traj_id), int(cl) - 1) for traj_id, cl in zip(mod.pathways.keys(), clusters))
    np.savetxt(f'{output_dir}/shine_labels_{merge_scheme}_{int(frac*100)}.csv', assignments, delimiter=',', fmt='%d',
               header=f'{params}\nPathway Index, Cluster Index')

    # Plot the dendrogram
    ax = mod.plot()
    plt.savefig(f'{output_dir}/shine_dendrogram_{merge_scheme}_{int(frac*100)}.png', bbox_inches='tight', dpi=500, pad_inches=0.1)
    plt.close()
