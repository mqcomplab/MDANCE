Getting Started
===============

Molecular Dynamics Analysis with *N*-ary Clustering Ensembles (MDANCE) is a flexible *n*-ary 
clustering package.

Installation
------------
.. code-block:: bash

   pip install mdance

To check for proper installation, run the following command:

.. code-block:: python
    
   import mdance
   print(mdance.__version__)

Usage
-----
To use MDANCE in a project, here is a simple example:

.. code-block:: python

   from mdance.cluster.nani import KmeansNANI
   from sklearn.cluster import KMeans
   import numpy as np

   data = np.load('data.npy')
   mod = KmeansNANI(data=data, n_clusters=4, metric='MSD', N_atoms=1, 
                    init_type='comp_sim', percentage=10)
   initiators = mod.initiate_kmeans()
   kmeans = KMeans(n_clusters, init=initiators, n_init=1, random_state=None)
   kmeans.fit(data)
   labels = kmeans.labels_