"""PRIME is a command line tool for structure prediction

-m - methods (for one method, None for all methods)
-s - folder to access for `w_union_SM_t10.txt` file
-i - similarity index (required)
-t - Fraction of outliers to trim in decimals.
-d - directory where the `normed_clusttraj.c*` files are located.

Note:
-----
Output will generate the index of the representative frames for the given method.
Python uses 0-based indexing, so the index will be 1 less than the actual frame number.
"""

import os
import subprocess
import sys


if not os.path.isfile('outputs/w_union_SM_t10.txt'):
    sys.exit('outputs/w_union_SM_t10.txt not found. Run normalize.py and exec_similarity.py first.')
subprocess.run([sys.executable, '-m', 'mdance.cli.prime_rep', '-m', 'union', '-s', 'outputs',
                '-d', 'outputs/normed_clusters', '-t', '0.1', '-i', 'SM'], check=True)
