"""
Learn NANI in 60 seconds!
===============================================
How to use NANI in 60 seconds? Say no more! 

**Note: This is the simplest example to get you started.**

The main idea is to use the NANI to optimize initial centroids so *k*-means is 100% 
deterministic, converges faster, and finds better solutions.
"""

###############################################################################
# Imports
#   - ``numpy <https://numpy.org/>`` for manipulating arrays.
#   - ``sci-kit-learn <https://scikit-learn.org/stable/>`` for *k*-means clustering.
#   - ``mdance.cluster.nani <https://mdance.readthedocs.io/en/latest/api/mdance.cluster.nani.html>``

import numpy as np
from sklearn.cluster import KMeans
from mdance.cluster.nani import KmeansNANI
###############################################################################
# Data
#   - Load the data from a file, must be array of shape (n_samples, n_features).
#   - In this example, we use a numpy file.

data = np.load('data.npy')
###############################################################################
# NANI
#   - Create an instance of KmeansNANI.
#   - ``data``: data to cluster.
#   - ``n_clusters``: number of clusters.


mod = KmeansNANI(data=data, n_clusters=4, metric='MSD', N_atoms=1, 
                 init_type='comp_sim', percentage=10)
initiators = mod.initiate_kmeans()
###############################################################################
# K-means
#   - Create an instance of KMeans.
#   - ``n_clusters``: number of clusters.
#   - ``init``: initial centroids.
#  - ``n_init``: NANI only needs one initialization!
#  - ``random_state``: We don't need this because NANI is 100% deterministic!

kmeans = KMeans(n_clusters=4, init=initiators, n_init=1, random_state=None)
kmeans.fit(data)
kmeans_labels = kmeans.labels_

###############################################################################
# That's it! You have successfully used NANI to optimize initial centroids for *k*-means clustering.
#   - ``kmeans_labels``: cluster labels assigned by *k*-means using NANI.

# For more advance usage, please look at the NANI Tutorial<https://mdance.readthedocs.io/en/latest/nani.html>!
# Why? Because NANI can also predict number of clusters, work with Molecular Dynamics data, and more!