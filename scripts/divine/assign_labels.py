import time
import sys
import numpy as np
from mdance import data
from mdance.cluster.divine import Divine        
from mdance.tools.bts import extended_comparison, calculate_medoid           

# System Information
input_traj_numpy = data.sim_traj_numpy
N_atoms = 101                                 
sieve = 1                           

# Parameters
end = 'k'                          
split = "weighted_MSD"                  
anchors = "nani"           
init_type = "strat_all"

k = 6                                        
threshold = 0             

top_n = 11

if __name__ == '__main__':
    full_traj = np.load(input_traj_numpy)
    selected_indices = np.arange(len(full_traj))[::sieve]
    traj_numpy = full_traj[selected_indices]

    mod = Divine(data=traj_numpy, split=split, anchors=anchors, init_type=init_type,
                end='k', k=k, N_atoms=N_atoms, threshold=threshold, refine=True)
    
    clusters, labels, scores, msds = mod.run()

    output_data = np.column_stack((selected_indices, labels)) 
    np.savetxt(f"divine_{anchors}_labels_{k}.csv", output_data, delimiter=",",
               header="frame_index,label", fmt="%d")

    best_frames_list = []

    for cid, cluster in enumerate(clusters):

        cluster_data = traj_numpy[cluster]
        medoid_rel_idx = calculate_medoid(cluster_data, metric="MSD", N_atoms=N_atoms)
        medoid_frame = cluster_data[medoid_rel_idx]

        distances = [extended_comparison(np.array([frame, medoid_frame]), data_type='full', metric="MSD", N_atoms=N_atoms) for frame in cluster_data]
        sorted_indices = np.argsort(distances)[:top_n]
        best_frames = cluster[sorted_indices]

        best_frames_indices = np.column_stack((selected_indices[best_frames], [cid] * len(best_frames)))
        best_frames_list.append(best_frames_indices)

    best_frames_indices = np.vstack(best_frames_list)
    best_frames_indices = best_frames_indices[np.argsort(best_frames_indices[:, 1])]

    np.savetxt(f"divine_{anchors}_best_frames_indices_{k}.csv", best_frames_indices, delimiter=',',
                fmt='%d', header="Frame Index,Cluster Index", comments='')
    
    