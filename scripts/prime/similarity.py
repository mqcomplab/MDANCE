"""A script to calculate the similarity between clusters using different methods.
Example usage:
>>> python similarity.py -m medoid -n 11 -i RR
"""
import sys
sys.path.insert(0, '../../')
import argparse
import modules as mod
import json
import time
import os

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', help='Method to use for similarity calculation. \
                    (pairwise, union, medoid, outlier)', required=True)
parser.add_argument('-n', '--n_clusters', type=int, help='Number of clusters for analysis', 
                    required=True)
parser.add_argument('-i', '--index', help='Similarity Index to use (e.g. RR or SM)', 
                    required=True)
parser.add_argument('-t', '--trim_frac', type=float, help='Fraction of outliers to trim. \
                    (e.g. 0.1, default: None)', default=None)
parser.add_argument('-w', '--weighted_by_frames', help='Weighing clusters by frames it contains. \
                    (default: True)', default=True)
parser.add_argument('-d', '--cluster_folder', help='Location of the cluster files directory', 
                    default="new_clusters/")
parser.add_argument('-s', '--summary_file', help='Location of CPPTRAJ cluster summary file', 
                    default="summary")
args = parser.parse_args()

# Calculate similarities
start = time.perf_counter()
lib = mod.FrameSimilarity(cluster_folder=args.cluster_folder, summary_file=args.summary_file, 
                               n_clusters=args.n_clusters, trim_frac=args.trim_frac, 
                               n_ary=args.index, weighted_by_frames=args.weighted_by_frames)
method_func = getattr(lib, f'calculate_{args.method}')
new_sims = method_func()

if args.weighted_by_frames:
    w = "w"

else:
    w = "nw"

dir_name = 'outputs'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

if args.trim_frac:
    with open(f'{dir_name}/{w}_{args.method}_{args.index}_t{int(float(args.trim_frac) * 100)}.txt', 
              'w') as file:
        file.write(json.dumps(new_sims, indent=4))
    end = time.perf_counter()
    print(f"{w}_{args.method}_{args.index}_t{int(float(args.trim_frac) * 100)}: \
          Finished in {round(end-start,2)} second")

else:
    with open(f'{dir_name}/{w}_{args.method}_{args.index}.txt', 'w') as file:
        file.write(json.dumps(new_sims, indent=4))
    end = time.perf_counter()
    print(f"{w}_{args.method}_{args.index}: Finished in {round(end-start,2)} second")

if __name__ == '__main__':
    os.system(f"python similarity.py -m medoid -n 11 -i RR")