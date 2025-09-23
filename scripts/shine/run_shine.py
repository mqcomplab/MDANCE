import glob

import matplotlib.pyplot as plt
import numpy as np

from mdance.cluster.shine import Shine


# System Parameters
file_pattern = 'data/*.csv'                                 # Follows data/trajectory_*.csv pattern where * is the trajectory number
metric = 'MSD'                                              # Metric to be used for clustering
N_atoms = 1                                                 # Number of atoms in the system

# SHINE Parameters
sampling = 'diversity'                                      # Sampling scheme {'diversity', 'quota', None}
frac = 0.5                                                  # Fraction of frames to be sampled
merge_scheme = 'intra'                                      # Merge scheme {'intra', 'semi_sum', 'min'}
t = 2                                                       # This would be max number of clusters requested

if __name__ == '__main__':
    frames_all = []

    # Change this to read the data from different file patterns.
    for file in glob.glob(file_pattern):
        traj = file.split('_')[-1].split('.')[0]
        frame = np.genfromtxt(file, delimiter=',')
        frames_all.append((traj, frame))
    
    # Run SHINE
    mod = Shine(frames_all, 'MSD', N_atoms=1 , t=t, criterion='maxclust', 
                link='ward', merge_scheme=merge_scheme, 
                sampling=sampling, frac=frac)
    link, clusters = mod.run()
    
    # Plot the dendrogram
    ax = mod.plot()
    plt.savefig('div_dendro_{}_{}.png'.format(merge_scheme, int(frac*100)), bbox_inches='tight', dpi=500, pad_inches=0.1)
    plt.close()
