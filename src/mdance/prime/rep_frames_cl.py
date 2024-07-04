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
from mdance.prime.rep_frames import gen_one_method_max, gen_all_methods_max
import argparse

parser_dict = {
    'method': {'type': str, 'help': 'method to use'},
    'sim_folder': {'type': str, 'help': 'folder to access'},
    'trim_frac': {'type': float, 'default': None, 'help': 'Trim parameter for gen_method_max method'},
    'index': {'type': str, 'default': 'RR', 'help': 'n_ary parameter for gen_method_max method'},
    'norm_folder': {'type': str, 'help': 'norm_folder to access'}
}

parser = argparse.ArgumentParser(description='Generate method max with optional trim and n_ary')
for key, value in parser_dict.items():
    parser.add_argument(f'-{key[0]}', f'--{key}', **value) 

args = parser.parse_args()

if args.method:
    gen_one_method_max(method=args.method, sim_folder=args.sim_folder, norm_folder=args.norm_folder, 
                       trim_frac=args.trim_frac, n_ary=args.index)
else:
    gen_all_methods_max(sim_folder=args.sim_folder, norm_folder=args.norm_folder, 
                        trim_frac=args.trim_frac, n_ary=args.index)