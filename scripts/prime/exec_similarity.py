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
import subprocess
import sys

from mdance import data


if __name__ == '__main__':
    if not os.path.isdir('outputs/normed_clusters'):
        sys.exit('outputs/normed_clusters not found. Run normalize.py first.')
    subprocess.run([sys.executable, '-m', 'mdance.cli.prime_sim', '-m', 'union', '-n', '6',
                    '-i', 'SM', '-t', '0.1', '-d', 'outputs/normed_clusters',
                    '-s', str(data.summary_6)], check=True)
