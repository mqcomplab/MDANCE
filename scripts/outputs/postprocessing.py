import os

import MDAnalysis as mda
import numpy as np

from mdance import data


# System parameters
input_top = data.top                                # Topology file (any MDAnalysis format)
input_traj = data.traj                              # Trajectory file (any MDAnalysis format)
cluster_assignment = data.best_frames_6             # Frame, cluster CSV (best frames or labels file)

# Output parameters
output_base_name = 'best_frames'                    # Prefix for the output trajectories
output_format = 'dcd'                               # Output trajectory format (any MDAnalysis format)
write_top_pdb = True                                # Also save each cluster's top frame as .pdb (the medoid for best-frames files)
output_dir = 'outputs'                              # Folder for output files


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    u = mda.Universe(input_top, input_traj)
    print(f'Number of atoms in the trajectory: {u.atoms.n_atoms}')

    frames, labels = np.loadtxt(cluster_assignment, delimiter=',', dtype=int, unpack=True)

    # Write one trajectory per cluster; -1 marks unassigned frames
    for cluster in np.unique(labels[labels >= 0]):
        cluster_frames = frames[labels == cluster]
        output_file = f'{output_dir}/{output_base_name}_{cluster}.{output_format}'
        with mda.Writer(output_file, u.atoms.n_atoms) as W:
            for ts in u.trajectory[cluster_frames]:
                W.write(u.atoms)
        print(f'Cluster {cluster}: {len(cluster_frames)} frames written to {output_file}')

        if write_top_pdb:
            u.trajectory[cluster_frames[0]]
            u.atoms.write(f'{output_dir}/{output_base_name}_{cluster}_top.pdb')
