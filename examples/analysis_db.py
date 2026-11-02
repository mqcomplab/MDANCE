"""
Cluster Analysis
===============================================

MDANCE provides a pipeline to screen the optimal number of clusters for a given dataset.

The pwd of this script is ``$PATH/MDANCE/examples``.
"""
###############################################################################
# To begin with, let's import the function we will use:

from mdance import data
from mdance.outputs.plot import plot_scores

###############################################################################
# ``scores_csv`` is a screening csv from `screen_nani.py`. Here we use the sample screening included with MDANCE.
#   - ``x_min``: only values greater than or equal to ``x_min`` are suggested (``None`` to consider all).

scores_csv = data.nani_screening
x_min = None

###############################################################################
# The function below will plot the Davies-Bouldin and Calinski-Harabasz indices and suggest the number of clusters.
#   - The Davies-Bouldin index suggests its minimum and the minima with the largest 2nd derivative.
#   - The Calinski-Harabasz index suggests its maximum and the maxima with the most negative 2nd derivative.

suggested, figs = plot_scores(scores_csv, x_min=x_min, output_prefix='nani_comp_sim_scores')
for index, values in suggested.items():
    print(f'{index}: {values}')