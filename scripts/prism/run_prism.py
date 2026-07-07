import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
from mdance import data
from mdance.cluster.prism import PRISM
from pathlib import Path


# System Parameters
data_dir = Path(data.__file__).parent / 'ala_pathways'
file_pattern = str(data_dir / '*.csv')  # Pattern pointing to trajectory files (each file = one pathway)
metric = 'MSD'

# PRISM Parameters
option = 2                          # Medoid construction: 1=global, 2=per-traj, 3=two-stage
k = 20                              # Number of clusters for representative medoid construction
k_final = k                         # Second-stage cluster count (used only if option == 3)
weight_scheme = 'weighted_avg'      # Average Hausdorff normalization scheme

t = 2
criterion = 'maxclust'
linkage_method = 'ward'

# Output Parameters
save_medoid_indices = True         # Save medoid frame indices as CSV


if __name__ == '__main__':
    frames_all = []

    # Modify parsing below based on your file structure and data format.
    # Each trajectory must be stored as (unique_id, 2D numpy array) with shape (n_frames, n_features).
    # All trajectories must have the same number of features (columns).
    for file in glob.glob(file_pattern):
        traj = file.split('_')[-1].split('.')[0]
        frame = np.genfromtxt(file, delimiter=',')
        frames_all.append((traj, frame))

    # for file in glob.glob(file_pattern, recursive=True):
    #     traj = file.split('/')[-1].split('.')[0]
    #     frame = np.load(file)
    #     frames_all.append((traj, frame))

    suffix = f'prism_opt{option}_k{k}' + (f'_kfinal{k_final}' if option == 3 else '')

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

    mod.run()

    # Build cluster assignments
    traj_keys = list(mod.pathways.keys())
    assignments = sorted(
        [(traj_id, int(cl) - 1) for traj_id, cl in zip(traj_keys, mod.clusters)],
        key=lambda x: (x[1], x[0])
    )

    # Summary
    unique_clusters = sorted({cl for _, cl in assignments})
    print(f"\nCluster summary ({len(unique_clusters)} clusters):")
    for cl in unique_clusters:
        members = [traj_id for traj_id, c in assignments if c == cl]
        print(f"  Cluster {cl}: {members}")

    # Save cluster assignments 
    cluster_path = f'{suffix}_labels.csv'
    with open(cluster_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['traj_id', 'cluster'])
        for traj_id, cl in assignments:
            writer.writerow([traj_id, cl])
    print(f"Saved cluster assignments to {cluster_path}")

    # Plot the dendrogram
    plt.figure(figsize=(2.5, 2))
    ax = mod.plot()
    ax_cur = plt.gca()
    ax_cur.xaxis.label.set_size(7)
    ax_cur.yaxis.label.set_size(7)
    ax_cur.tick_params(axis='both', labelsize=6)
    plt.savefig(f'{suffix}.png', bbox_inches='tight', dpi=500, pad_inches=0.1)
    plt.close()

    # Save medoid indices
    if save_medoid_indices:
        csv_path = f'{suffix}_medoid_indices.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['assigned_traj', 'source_traj', 'frame_index'])
            for traj_id, meds in mod.medoids_per_traj.items():
                for med in meds:
                    for source_id, source_frames in mod.pathways.items():
                        matches = np.where(np.all(source_frames == med, axis=1))[0]
                        if len(matches) > 0:
                            writer.writerow([traj_id, source_id, int(matches[0])])
                            break
        print(f"Saved medoid indices to {csv_path}")
