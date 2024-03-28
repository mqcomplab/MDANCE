from mdance.scripts import screen_nani

# System info
input_traj_numpy = '../../examples/md/backbone.npy'
N_atoms = 50
sieve = 1

# NANI parameters
output_dir = 'outputs'                        
init_types = ['comp_sim']                                           # Must be a list
metric = 'MSD'
start_n_clusters = 5                                                # At least 2 clusters
end_n_clusters = 30                                                 # Maximum number of clusters

if __name__ == '__main__':
    screen_nani(input_traj_numpy, N_atoms, sieve, init_types, metric, start_n_clusters, end_n_clusters, output_dir) # type: ignore