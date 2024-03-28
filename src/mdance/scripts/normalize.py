import numpy as np
from ..inputs.preprocess import normalize_file
import os
from argparse import _SubParsersAction


def normalize(data_file : str, delimiter : str = ',' , output_base_name : str = 'array'):
    """
    I am most likely not the best one to do the descriptions here, but here is the template - φ

    Parameters
    ----------
    data_file : str
        
    delimiter : str, optional
        Delimiter used for parsing the data_file (e.g. commas for csv), by default ','
    output_base_name : str, optional
        Output npy file path/name without npy extension, by default 'array'
    """
    array = np.genfromtxt(data_file, delimiter=delimiter)
    output_name = f'{output_base_name}.npy'
    normed_data, min, max, avg = normalize_file(array, norm_type='v3')
    np.save(output_name, normed_data)

def nm_parser(subparser : _SubParsersAction):
    """
    Command-line arguments for the assign_labels script

    Parameters
    ----------
    subparser : _SubParsersAction
        Subparser object containing the assign_labels parser
    """
    nm = subparser.add_parser('NORMALIZE', help='Run the \"normalize\" script')

    # I'm not too sure what is best to write for the help descriptions here - φ

    ###########################
    # Arguments for normalize #
    ###########################

    nm.add_argument(
        '-delim', '-delimeter', metavar='CHAR', 
        type=str, dest='delimeter', default=',',
        help='')

    nm.add_argument(
        '-out', '-output', '--output-dir', metavar="DIRECTORY", type=str, nargs='?', 
        default='outputs', const='outputs', dest='output_dir',
        help=''
    )


if __name__ == '__main__':
    normalize('../../examples/2D/blob_disk.csv')