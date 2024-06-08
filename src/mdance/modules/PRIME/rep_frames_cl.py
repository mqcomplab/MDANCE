"""Call the `gen_one_method_max` to generate the method max for a single method 
or `gen_all_methods_max` to generate the method max for all methods.

Example usage:
----------------
No trim, RR index
>>> python scripts/rep_frames.py
10% trim, RR index
>>> python scripts/rep_frames.py -t 0.1
20% trim, SM index
>>> python scripts/rep_frames.py -t 0.2 -i SM
"""
import sys
sys.path.insert(0, '../../')
from src.modules.PRIME.rep_frames import gen_one_method_max, gen_all_methods_max
import argparse

parser = argparse.ArgumentParser(description='Generate method max with optional trim and n_ary')
parser.add_argument('-m', '--method', type=str, help='method to use')
parser.add_argument('-s', '--sim_folder', type=str, help='folder to access')
parser.add_argument('-t', '--trim_frac', type=float, default=None,
                    help='Trim parameter for gen_method_max method')
parser.add_argument('-i', '--index', type=str, default='RR',
                    help='n_ary parameter for gen_method_max method')
parser.add_argument('-d', '--norm_folder', type=str, help='norm_folder to access')

args = parser.parse_args()
if args.method:
    gen_one_method_max(method=args.method, sim_folder=args.sim_folder, norm_folder=args.norm_folder, 
                       trim_frac=args.trim_frac, n_ary=args.index)
else:
    gen_all_methods_max(sim_folder=args.sim_folder, norm_folder=args.norm_folder, 
                        trim_frac=args.trim_frac, n_ary=args.index)