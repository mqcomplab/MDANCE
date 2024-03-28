import numpy as np
from ..modules.kmeansNANI import KmeansNANI
from ..tools.bts import extended_comparison, calculate_medoid
import os
from argparse import _SubParsersAction
from typing import Literal

type InitiatorStrings = Literal['div_select', 'comp_sim', 'k-means++', 'random']
type MetricStrings = Literal['MSD', 'BUB', 'Fai', 'Gle', 'Ja', 'JT', 'RT', 'RR', 'SM', 'SS1', 'SS2']

def assign_labels(input_traj_numpy : str, N_atoms : int = 50, sieve : int = 1, 
                  n_clusters : int = 6, init_type : InitiatorStrings = 'comp_sim', 
                  metric : MetricStrings = 'MSD', n_structures : int = 11, 
                  output_dir : str = 'outputs'):
    """
    I am most likely not the best one to do the descriptions here, but here is the template - φ

    Parameters
    ----------
    input_traj_numpy : str
        
    N_atoms : int, optional
        , by default 50
    sieve : int, optional
        , by default 1
    n_clusters : int, optional
        , by default 6
    init_type : InitiatorStrings, optional
        , by default 'comp_sim'
    metric : MetricStrings, optional
        , by default 'MSD'
    n_structures : int, optional
        , by default 11
    output_dir : str, optional
        Data output directory, by default 'outputs'
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    traj_numpy = np.load(input_traj_numpy)[::sieve]
    mod = KmeansNANI(data=traj_numpy, n_clusters=n_clusters, N_atoms=N_atoms, init_type=init_type, 
                     metric=metric, percentage=10)
    labels, centers, n_iter = mod.execute_kmeans_all()
    sort_labels_by_size = np.argsort(np.bincount(labels))[::-1]
    labels = np.array([np.where(sort_labels_by_size == i)[0][0] for i in labels])
    best_frames = []
    cluster_msd = []

    # Save best frames indices for each cluster
    for i, label in enumerate(np.unique(labels)):
        cluster = np.where(labels == label)[0]
        if len(cluster) > 1:
            medoid_index = calculate_medoid(traj_numpy[cluster], metric=metric, N_atoms=N_atoms)
            medoid = traj_numpy[cluster][medoid_index]
            msd_to_medoid = []
            for j, frame in enumerate(traj_numpy[cluster]):
                msd_to_medoid.append((j, extended_comparison(
                    np.array([frame, medoid]), data_type='full', metric=metric, N_atoms=N_atoms)))
            msd_to_medoid = np.array(msd_to_medoid)
            sorted_indices = np.argsort(msd_to_medoid[:, 1])
            best_n_structures = traj_numpy[cluster][sorted_indices[:n_structures]]
            best_frames.append(best_n_structures)
    
    best_frames_indices = []
    for i, frame in enumerate(traj_numpy):
        i = i * sieve
        for j, cluster in enumerate(best_frames):
            if np.any(np.all(cluster == frame, axis=1)):
                best_frames_indices.append((i, j))
    best_frames_indices = np.array(best_frames_indices)
    best_frames_indices = best_frames_indices[best_frames_indices[:, 1].argsort()]
    np.savetxt(f'{output_dir}/best_frames_indices_{n_clusters}.csv', best_frames_indices, delimiter=',', fmt='%s', 
               header=f'Numer of clusters,{n_clusters}\nFrame Index,Cluster Index')
    
    # Save cluster labels
    with open(f'{output_dir}/labels_{n_clusters}.csv', 'w') as f:
        f.write(f'# init_type: {init_type}, Number of clusters: {n_clusters}\n')
        f.write('# Frame Index, Cluster Index\n')
        for i, row in enumerate(labels):
            f.write(f'{i * sieve},{row}\n')
    
    # Calculate population of each cluster
    with open(f'{output_dir}/summary_{n_clusters}.csv', 'w') as f:
        f.write(f'# Number of clusters, {n_clusters}\n')
        f.write('# Cluster Index, Fraction out of total pixels\n')
        for i, row in enumerate(np.bincount(labels)):
            f.write(f'{i},{row/len(labels)}\n')


def al_parser(subparser : _SubParsersAction):
    """
    Command-line arguments for the assign_labels script

    Parameters
    ----------
    subparser : _SubParsersAction
        Subparser object containing the assign_labels parser
    """
    al = subparser.add_parser('ASSIGN', help='Run the \"assign_labels\" script')

    # I'm not too sure what is best to write for the help descriptions here - φ

    ###############################
    # Arguments for assign_labels #
    ###############################

    al.add_argument(
        '-natoms', '--n-atoms', metavar='INTEGER', type=int, default=50, 
        help=''
    )
    
    al.add_argument(
        '-sv', '-sieve',metavar='INTEGER',type=int,default=1, dest='sieve',
        help='' 
    )
    
    al.add_argument(
        '-nclusters', '--n-clusters', metavar="INTEGER", type=int, default=6,
        help=''
    )

    al.add_argument(
        '-init', '--init-type', nargs='?',
        choices=['div_select', 'comp_sim', 'k-means++', 'random'], 
        default='comp_sim', const='comp_sim',
        help=''
    )

    al.add_argument(
        '-metric', nargs='?',
        choices=['MSD', 'BUB', 'Fai', 'Gle', 'Ja', 'JT', 'RT', 'RR', 'SM', 'SS1', 'SS2'],
        default='MSD', const='MSD',
        help=''
    )

    al.add_argument(
        '-nstructures', '--n-structures', metavar="INTEGER", type=int, default=11,
        help=''
    )

    al.add_argument(
        '-out', '-output', '--output-dir', metavar="DIRECTORY", type=str, nargs='?', 
        default='outputs', const='outputs', dest='output_dir',
        help=''
    )


if __name__ == '__main__':
    assign_labels('../../examples/md/backbone.npy')