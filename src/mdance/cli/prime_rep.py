import argparse

from mdance.prime.rep_frames import gen_one_method_max, gen_all_methods_max


def main():
    """Main function to run the command line interface for generating
    method max values.
    
    Returns
    -------
    txt file with method max values.
    """
    parser_dict = {
        'method': {'flags': ['-m', '--method'], 'kwargs': {'type': str, 'help': 'method to use', 'required': True}},
        'sim_folder': {'flags': ['-s', '--sim_folder'], 'kwargs': {'type': str, 'help': 'folder to access'}},
        'trim_frac': {'flags': ['-t', '--trim_frac'], 'kwargs': {'type': float, 'default': None, 'help': 'Trim parameter for gen_method_max method', 'required': True}},
        'index': {'flags': ['-i', '--index'], 'kwargs': {'type': str, 'default': 'RR', 'help': 'n_ary parameter for gen_method_max method', 'required': True}},
        'norm_folder': {'flags': ['-d', '--norm_folder'], 'kwargs': {'type': str, 'help': 'norm_folder to access'}}
    }
    parser = argparse.ArgumentParser(description='Generate method max with optional trim and n_ary')
    for key, value in parser_dict.items():
        parser.add_argument(*value['flags'], **value['kwargs'])
    args = parser.parse_args()

    if args.method:
        gen_one_method_max(method=args.method, sim_folder=args.sim_folder, norm_folder=args.norm_folder, 
                        trim_frac=args.trim_frac, n_ary=args.index)
    else:
        gen_all_methods_max(sim_folder=args.sim_folder, norm_folder=args.norm_folder, 
                            trim_frac=args.trim_frac, n_ary=args.index)


if __name__ == '__main__':
    main()