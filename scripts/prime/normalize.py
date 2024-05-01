import numpy as np
import sys
sys.path.insert(0, '../../')
from src.inputs.preprocess import gen_traj_numpy, normalize_file, Normalizer
import re
import glob
import os

# System info - EDIT THESE
input_top = '../../examples/md/aligned_tau.pdb'
unnormed_cluster_dir = '../outputs/labels_*'
output_dir = 'normed_clusters'
output_base_name = 'normed_clusttraj'
atomSelection = 'resid 3 to 12 and name N CA C O H'
n_clusters = 6

if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    list_clusttraj = sorted(glob.glob(unnormed_cluster_dir), 
                            key=lambda x: int(re.findall("\d+", x)[0]))
    list_clusttraj = list_clusttraj[:n_clusters]
    all_clusttraj = []
    for clusttraj in list_clusttraj:
        traj_numpy = gen_traj_numpy(input_top, clusttraj, atomSelection)
        all_clusttraj.append(traj_numpy)
    concat_clusttraj = np.concatenate(all_clusttraj)
    normed_data, min, max, avg = normalize_file(concat_clusttraj, norm_type='v3')
    np.save(f'{output_dir}/normed_data.npy', normed_data)
    for i, traj in enumerate(all_clusttraj):
        norm = Normalizer(data=traj, custom_min=min, custom_max=max)
        normed_frame = norm.get_v3_norm()
        np.save(f'{output_dir}/{output_base_name}.c{i}.npy', normed_frame)
    