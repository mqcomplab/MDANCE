import numpy as np
from ..modules.kmeansNANI import KmeansNANI, compute_scores
from ..tools.bts import extended_comparison
import os
from argparse import _SubParsersAction
from typing import Literal

type InitiatorStrings = Literal['div_select', 'comp_sim', 'k-means++', 'random']
type MetricStrings = Literal['MSD', 'BUB', 'Fai', 'Gle', 'Ja', 'JT', 'RT', 'RR', 'SM', 'SS1', 'SS2']


def screen_nani(input_traj_numpy : str, N_atoms : int = 50, sieve : int = 1, 
                  init_types : list[InitiatorStrings] = ['comp_sim'], 
                  metric : MetricStrings = 'MSD', start_n_clusters : int = 5,
                  end_n_clusters : int = 30, output_dir : str = 'outputs'):
    """
    I am most likely not the best one to do the descriptions here, but here is the template - φ

    Parameters
    ----------
    input_traj_numpy : str
        
    N_atoms : int, optional
        , by default 50
    sieve : int, optional
        , by default 1
    init_types : list[InitiatorStrings], optional
        , by default ['comp_sim']
    metric : MetricStrings, optional
        , by default 'MSD'
    start_n_clusters : int, optional
        , by default 5
    end_n_clusters : int, optional
        , by default 30
    output_dir : str, optional
        Data output directory, by default 'outputs'
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    traj_numpy = np.load(input_traj_numpy)[::sieve]
    for init_type in init_types:
        if init_type in ['k-means++', 'random', 'vanilla_kmeans++']:
            percentage = ''
        
        # `comp_sim` and `div_select` are ran only once to get the initiators
        elif init_type in ['comp_sim', 'div_select']:
            percentage = 10
            mod = KmeansNANI(data=traj_numpy, n_clusters=end_n_clusters, metric=metric, 
                             N_atoms=N_atoms, init_type=init_type, percentage=percentage)
            initiators = mod.initiate_kmeans()
        
        snl_scores = []
        for i in range(start_n_clusters, end_n_clusters+1):
            totsn = 0

            # Run k-means clustering
            if init_type in ['comp_sim', 'div_select']:
                mod = KmeansNANI(data=traj_numpy, n_clusters=i, metric=metric, 
                                 N_atoms=N_atoms, init_type=init_type, percentage=percentage)
                labels, centers, n_iter = mod.kmeans_clustering(initiators)
            elif init_type in ['k-means++', 'random']:
                mod = KmeansNANI(data=traj_numpy, n_clusters=i, metric=metric, 
                                 N_atoms=N_atoms, init_type=init_type)
                labels, centers, n_iter = mod.kmeans_clustering(initiators=init_type)
            elif init_type == 'vanilla_kmeans++':
                mod = KmeansNANI(data=traj_numpy, n_clusters=i, metric=metric, 
                                 N_atoms=N_atoms, init_type=init_type)
                initiators = mod.initiate_kmeans()
                labels, centers, n_iter = mod.kmeans_clustering(initiators=initiators)
            
            
            # Compute scores
            ch_score, db_score = compute_scores(data=traj_numpy, labels=labels)
            
            # Csnculate MSD for each cluster
            dict = {}
            for j in range(i):
                dict[j] = np.where(labels == j)[0]
                dict[j] = traj_numpy[dict[j]]
            for key in dict:
                msd = extended_comparison(np.array(dict[key]), traj_numpy_type='full', 
                                          metric=metric, N_atoms=N_atoms)
                totsn += msd
            snl_scores.append((i, n_iter, ch_score, db_score, totsn/i))
        
        snl_scores = np.array(snl_scores)
        header = f'init_type: {init_type}, percentage: {percentage}, metric: {metric}, sieve: {sieve}\n'
        header += 'Number of clusters, Number of iterations, Csninski-Harabasz score, Davies-Bouldin score, Average MSD'
        np.savetxt(f'{output_dir}/{percentage}{init_type}_summary.csv', snl_scores, 
                   delimiter=',', header=header, fmt='%s')


def sn_parser(subparser : _SubParsersAction):
    """
    Command-line arguments for the assign_labels script

    Parameters
    ----------
    subparser : _SubParsersAction
        Subparser object containing the assign_labels parser
    """
    sn = subparser.add_parser('SCREEN', help='Run the \"screen_nani\" script')

    # I'm not too sure what is best to write for the help descriptions here - φ

    #############################
    # Arguments for screen_nani #
    #############################

    sn.add_argument(
        '-natoms', '--n-atoms', metavar='INTEGER', type=int, default=50, 
        help=''
    )
    
    sn.add_argument(
        '-sv', '-sieve',metavar='INTEGER',type=int,default=1, dest='sieve',
        help='' 
    )

    sn.add_argument(
        '-inits', '--init-types', nargs='*',
        choices=['div_select', 'comp_sim', 'k-means++', 'random'], 
        default=['comp_sim'], dest='inits',
        help=''
    )

    sn.add_argument(
        '-metric', nargs='?',
        choices=['MSD', 'BUB', 'Fai', 'Gle', 'Ja', 'JT', 'RT', 'RR', 'SM', 'SS1', 'SS2'],
        default='MSD', const='MSD',
        help=''
    )

    sn.add_argument(
        '-snc', '--start-n-clusters', metavar="INTEGER", 
        type=int, default=5, dest='snc',
        help='At least 2 clusters'
    )

    sn.add_argument(
        '-enc', '--end-n-clusters', metavar="INTEGER",
        type=int, default=30, dest='enc',
        help=''
    )

    sn.add_argument(
        '-out', '-output', '--output-dir', metavar="DIRECTORY", type=str, nargs='?', 
        default='outputs', const='outputs', dest='output_dir',
        help=''
    )


if __name__ == '__main__':
    screen_nani('../../examples/md/backbone.npy')