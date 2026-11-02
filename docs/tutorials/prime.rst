.. raw:: html

    <h3 align="center"> 
        <img src="../_static/logo.png" width="800" height=auto align="center"></a></h3>


PRIME Tutorial
==============

.. contents::
   :local:
   :depth: 2

Overview
--------

This clustering tutorial is meant for datasets Molecular Dynamics
Trajectory. PRIME assumes a MD trajectory that has a well-sampled
ensemble of conformations. The PRIME algorithm predicts the native
structure of a protein from simulation or clustering data. These methods
perfectly mapped all the structural motifs in the studied systems and
required unprecedented linear scaling.

Tutorial
--------

The following tutorial will guide you through the process of determining
the native structure of a biomolecule using the PRIME algorithm. If you
do not have clustered data. Please refer to other clustering algorithms
such as `NANI <nani.html>`__ to cluster your data, follow all steps.

1. Clone the MDANCE Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First things first, clone the MDANCE repository if you haven't already.

.. code:: bash

    $ git clone https://github.com/mqcomplab/MDANCE.git
    $ cd MDANCE/scripts/prime

2. Cluster Normalization
~~~~~~~~~~~~~~~~~~~~~~~~

`normalize.py <https://github.com/mqcomplab/MDANCE/blob/main/scripts/prime/normalize.py>`__ With
already clustered data, this script will normalize the trajectory data
between :math:`[0,1]` using the Min-Max Normalization.

PRIME needs two files from a clustering algorithm such as `NANI <nani.html>`__:

- a labels file with the cluster of each frame (frame index, cluster index), with clusters numbered by size (cluster 0 being the largest), e.g. ``nani_labels_6.csv`` from ``scripts/nani/assign_labels.py``;
- a summary file with the population of each cluster (cluster index, fraction of total frames), e.g. ``nani_summary_6.csv``.

The scripts use the samples included with MDANCE, ``mdance.data.labels_6`` and ``mdance.data.summary_6``. Run the steps in order; each step uses the outputs of the previous one.

::

   # System parameters
   input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
   cluster_labels = data.labels_6                      # Labels file from a clustering script
   output_dir = 'outputs/normed_clusters'              # Folder for the normalized cluster files
   output_base_name = 'normed_clusttraj'               # Prefix for the normalized cluster files
   n_clusters = 6                                      # Number of clusters in the labels file

Inputs
^^^^^^

System parameters
'''''''''''''''''

| ``input_traj_numpy`` is the numpy array used in the clustering.
| ``cluster_labels`` is the labels file from the clustering.
| ``output_dir`` is the directory where the normalized clustering files will be saved. 
| ``output_base_name`` is the base name for the output files. 
| ``n_clusters`` is the number of clusters used in the PRIME. If number less than total number of cluster, it will take top *n* number of clusters.

Execution
^^^^^^^^^

Make sure your pwd is ``$PATH/MDANCE/scripts/prime``.

.. code:: bash

   $ python normalize.py

Outputs
^^^^^^^

| ``outputs/normed_clusters/normed_clusttraj.c*.npy`` files, normalized clustering files.
| ``outputs/normed_clusters/normed_data.npy``, appended all normed files together.

3. Similarity Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~

``prime_sim`` generates a similarity dictionary from running PRIME.

| ``-h``    Help with the argument options.
| ``-m``    Methods, {pairwise, union, medoid, outlier} (*required*).
| ``-n``    Number of clusters (*required*).
| ``-i``    Similarity index, {RR or SM} (*required*).
| ``-t``    Fraction of outliers to trim in decimals (default is None).
| ``-w``    Weighing clusters by frames it contains (default is True).
| ``-d``    Directory where the ``normed_clusttraj.c*.npy`` files are located (*required*)
| ``-s``    Location where ``summary`` file is located with population of each cluster (*required*)

Execution
^^^^^^^^^

Make sure your pwd is ``$PATH/MDANCE/scripts/prime``.

.. code:: bash

    $ prime_sim -m union -n 6 -i SM -t 0.1 -d outputs/normed_clusters -s ../nani/outputs/nani_summary_6.csv

To generate a similarity dictionary using the normalized clusters from step 2 (make sure you
are in the prime directory) using the ``union`` method (2.2 in *Fig 2*) and
Sokal Michener index. In addition, 10% of the outliers were trimmed. You can either ``python exec_similarity.py``
(which uses the sample summary ``mdance.data.summary_6``) or run the example above. 

Outputs
^^^^^^^

| ``outputs/w_union_SM_t10.txt`` file with the similarity dictionary. 
| The result is a dictionary organized as followes: 

.. code:: text

    {
        "frame_0": [   
            0.7,    # cluster 1 similarity.
            0.9,    # cluster 2 similarity.
            ...,
            0.8     # average similarity of all above similarities.
        ] 
    }
    
4. Representative Frames
~~~~~~~~~~~~~~~~~~~~~~~~

``prime_rep`` will determine the native structure of the protein using
the similarity dictionary generated in step 3.

| ``-h``    for help with the argument options. 
| ``-m``    methods (for one method, None for all methods).
| ``-s``    folder to access for ``w_union_SM_t10.txt`` file.
| ``-i``    similarity index (*required*)
| ``-t``    Fraction of outliers to trim in decimals (default is None).
| ``-d``    directory where the ``normed_clusttraj.c*`` files are located (required if method is None)

Execution
^^^^^^^^^

Make sure your pwd is ``$PATH/MDANCE/scripts/prime``.

.. code:: bash

    $ prime_rep -m union -s outputs -d outputs/normed_clusters -t 0.1 -i SM

Outputs
^^^^^^^

``outputs/w_rep_SM_t10_union.txt`` file with the representative frames index. You can either ``python exec_rep_frames.py`` or run the example above.

Further Reading
---------------

For more information on the PRIME algorithm, please refer to the `PRIME
paper <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00362>`__.

Please Cite

.. code:: bibtex

    @article{chen_protein_2024,
        title = {Protein {Retrieval} via {Integrative} {Molecular} {Ensembles} ({PRIME}) through {Extended} {Similarity} {Indices}},
        author = {Chen, Lexin and Mondal, Arup and Perez, Alberto and Miranda-Quintana, Ramón Alain},
        journal = {Journal of Chemical Theory and Computation},
        volume = {20},
        number = {14},
        pages = {6303--6315},
        year = {2024},
        doi = {10.1021/acs.jctc.4c00362},
        url = {https://doi.org/10.1021/acs.jctc.4c00362},
    }

.. image:: ../img/methods.jpg
  :width: 500
  :alt: Alternative text

*Fig 2. Six techniques of protein refinement. Blue is top cluster.*
