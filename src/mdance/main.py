import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction
from typing import Callable
import numpy as np

type MDANCE_Scripts = dict[str ,tuple[Callable[..., None], Callable[[_SubParsersAction[ArgumentParser]], None]]]

MASL = ('https://userguide.mdanalysis.org/stable/selections.html', 'MDAnalysis Atom Selections Language')

# ADD SCRIPTS HERE #
# Include script function name, then script function, then argument parser for the script as well
from mdance.scripts.assign_labels import assign_labels, al_parser
from mdance.scripts.screen_nani import screen_nani, sn_parser
from mdance.scripts.normalize import normalize, nm_parser

SCRIPTS : MDANCE_Scripts = {
    'ASSIGN' : (assign_labels, al_parser),
    'SCREEN' : (screen_nani, sn_parser),
    'NORMALIZE' : (normalize, nm_parser),
}

def parse_command_line_args(arg_list : list[str]) -> Namespace:
    """
    Convert command-line user input into Namespace object for greater utility

    Parameters
    ----------
    arg_list : list[str]
        List of arguments for parser, typically from system's argv

    Returns
    -------
    args : Namespace
        Simple object holding command-line parameters and
        their designated attributes
    """
    # Program title and description, initialization of parser
    parser = ArgumentParser(
        prog='MDANCE', 
        description='Molecular Dynamics Analysis with N-ary Clustering Ensembles (MDANCE) \
                     is a flexible n-ary clustering package that provides a set of tools \
                     for clustering Molecular Dynamics trajectories.')

    # Additional help argument
    parser.add_argument('-help', action='help')
    
    # Input argument for specifying the file-path
    parser.add_argument(
        '-in', '--input', 
        metavar="File",
        type=str,
        help='Designate input file(s) for MDANCE processing \
              (e.g. npy, csv)')
    
    # Topology for preprocessing mode
    parser.add_argument(
        '-top', '-topology',
        metavar="File",
        type=str,
        help='Designate input topology to preprocess'
    )

    # Trajectory for preprocessing mode
    parser.add_argument(
        '-traj','-trajectory',
        metavar='File',
        type=str,
        help='Designate input trajectory to preprocess'
    )

    # Add a subparser for each possible script before running
    subparser = parser.add_subparsers(title="Processing", dest='proc')

    # Iterate through script argument parsers
    for script in SCRIPTS.values():
        script[1](subparser)

    # Return args Namespace object
    args = parser.parse_args()
    
    # Ensure that both topology and trajectory are provided
    if bool(args.top) != bool(args.traj):
        parser.error('Only one of two required files were inputted; Please input both topology and trajectory files')

    # Ensure input exists before returning
    if not args.input and not (args.traj and args.top):
        parser.error('No file input, please input at least one file.')

    # Ensure script is chosen, throw error
    if not args.proc:
        parser.error('No script chosen, use \'-h\' option to see list of possible scripts.')

    return args
    
def _link(uri : str, label : str = "") -> str:
    """
    Generate a hyperlink for the user to select

    Parameters
    ----------
    uri : str
        uri or url of the website
    label : str, optional
        label to desplay the hyperlink, by default ""

    Returns
    -------
    str
        Designated hyperlink
    """
    if not label: 
        label = uri
    parameters = ''

    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)

def entry():
    # Call argument parsing function to treat input
    args = parse_command_line_args(sys.argv[1:])

    file_input = args.input

    # If the input is empty, generate input from the topology and trajectory files instead
    if not file_input:
        from mdance.inputs.preprocess import gen_traj_numpy
        npy_output = input("Type an output file name (without file extension): ")
        atomSelection = input(f"Type the atom selection for clustering ({_link(*MASL)}): ")

        atomSelection = atomSelection.strip("'")
        traj_numpy = gen_traj_numpy(args.top, args.traj, atomSelection)
        output_name = npy_output + '.npy'
        np.save(output_name, traj_numpy)

        file_input = output_name

    # Add your script and the parameters it requires from args here
    match args.proc:
        case "ASSIGN":
            SCRIPTS['ASSIGN'][0](file_input, args.n_atoms, 
                                 args.sieve, args.n_clusters,
                                 args.init_type, args.metric,
                                 args.n_structures, args.output_dir)
        case "SCREEN":
            SCRIPTS['SCREEN'][0](file_input, args.n_atoms,
                                 args.sieve, args.inits,
                                 args.metric, args.snc,
                                 args.enc, args.output_dir)
        case "NORMALIZE":
            SCRIPTS['NORMALIZE'][0]()
        case _:
            pass

if __name__ == "__main__":
    entry()