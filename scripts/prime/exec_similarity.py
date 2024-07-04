"""PRIME is a command line tool for structure prediction

-h - for help with the argument options.
-m - methods (required)
-n - number of clusters (required)
-i - similarity index (required)
-t - Fraction of outliers to trim in decimals.
-w - Weighing clusters by frames it contains.
-d - directory where the `normed_clusttraj.c*` files are located.
-s - location where summary with population file is located

Potential Error:
 w_dict[key] = [old_list[i] * v for i, v in enumerate(weights)]
IndexError: list index out of range
Check -n argument to ensure it is not greater than the number of clusters in the directory.
"""
import os
if __name__ == '__main__':
    os.system('python ../../src/modules/PRIME/similarity_cl.py -m union -n 6 -i SM -t 0.1  \
              -d normed_clusters -s ../nani/outputs/summary_6.csv')