from mdance.scripts import assign_labels

# System info - EDIT THESE
input_traj_numpy = '../../examples/md/backbone.npy'
N_atoms = 50
sieve = 1

# K-means params - EDIT THESE
n_clusters = 6
init_type = 'comp_sim'                                              # Default
metric = 'MSD'                                                      # Default
n_structures = 11                                                   # Default
output_dir = 'outputs'                                              # Default

if __name__ == '__main__':
    assign_labels(input_traj_numpy, N_atoms, sieve, n_clusters, init_type, metric, n_structures, output_dir)