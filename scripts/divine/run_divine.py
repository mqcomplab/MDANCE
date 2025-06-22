import time
import sys
import numpy as np
from mdance import data
from mdance.cluster.divine import Divine            

# System Information
input_traj_numpy = data.sim_traj_numpy
N_atoms = 101                                 
sieve = 1                              

# Parameters
end = 'k'                          
split = "weighted_MSD"                  
anchors = "nani"           
init_type = "strat_all"

k = 30                                        
threshold = 0            

if __name__ == '__main__':
    full_traj = np.load(input_traj_numpy)
    selected_indices = np.arange(len(full_traj))[::sieve] 
    traj_numpy = full_traj[::sieve]               

    mod = Divine(data=traj_numpy, split=split, anchors=anchors, init_type=init_type,
                end='k', k=k, N_atoms=N_atoms, threshold=threshold, refine=True)
    clusters, labels, scores, msds = mod.run()

    output_data = np.column_stack((selected_indices, labels)) 

    np.savetxt(f"divine_{anchors}_scores.csv", scores, delimiter=",", header="n_clusters,CH_score,DB_score", fmt=["%d", "%.7f", "%.7f"], comments="")    
    np.savetxt(f"divine_{anchors}_msds.csv", msds, fmt="%d,%d,%d,%.5f",delimiter=",",header="level,cluster_index,population,msd",comments="")       

