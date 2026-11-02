HELM - Hierarchical Extended Linkage Method
===========================================

.. contents::
   :local:
   :depth: 2

Overview
--------
HELM is a hierarchical agglomerative clustering algorithm that uses the *n*-ary similarity to merge clusters at each level. It transformed from the traditional hierarchical clustering algorithm to be more efficient and scalable turning a :math:`O(N^2)` algorithm to :math:`O(N)`. It specializes in recognizing dominant conformations within an ensemble and is often used alongside NANI to achieve a balance between efficiency and precision.

This clustering tutorial is meant for datasets for all applications (2D fingerprints, mass spectrometry imaging data, etc). Molecular Dynamics Trajectory has a different treatment. If a specific step is only for Molecular Dynamics trajectory, it will be specified. Otherwise, it is applicable for all datasets.

Tutorial
--------

1. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~

Clone the MDANCE repository if you haven't already.

.. code:: bash

   $ git clone https://github.com/mqcomplab/MDANCE.git
   $ cd MDANCE/scripts/nani

2. Input Preparations
~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>

.. raw:: html

   <summary>

Preparation for Molecular Dynamics Trajectory

.. raw:: html

   </summary>

Prepare a valid topology file (e.g. ``.pdb``, ``.prmtop``), trajectory
file (e.g. ``.dcd``, ``.nc``), and the atom selection. This step will
convert a Molecular Dynamics trajectory to a numpy ndarray. **Make sure
the trajectory is already aligned and/or centered if needed!**

`Preprocessing Notebook <../examples/preprocessing.html>`__ 
contains step-by-step tutorial to prepare the input for NANI. 

A copy of this notebook can be found in ``$PATH/MDANCE/scripts/inputs/preprocessing.ipynb``.

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

Preparation for all other datasets (OPTIONAL)

.. raw:: html

   </summary>

This step is **optional**. If you are using a metric that is NOT the
mean-square deviation (MSD)–default metric, you will need to normalize
the dataset. Otherwise, you can skip this step.

`normalize.py <https://github.com/mqcomplab/MDANCE/blob/main/scripts/inputs/normalize.py>`__ will
normalize the dataset. The following parameters to be specified in the
script:

::

   # System parameters
   data_file = data.blob_disk                          # Input data as a CSV file
   array = np.genfromtxt(data_file, delimiter=',')
   output_base_name = 'array'                          # Prefix for the output files

Inputs
^^^^^^

System parameters
'''''''''''''''''

| ``data_file`` is your input file with a 2D array. 
| ``array`` is the array is the loaded dataset from ``data_file``. This step can be changed according to the type of file format you have. However, ``array`` must be an array-like in the shape (number of samples, number of features).
| ``output_base_name`` is the base name for the output file. The output file will be saved as ``outputs/{output_base_name}.npy``. 

.. raw:: html

   </details>

3. Prior Clustering (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HELM requires a set of initial clusters to start with. You can start from any clustering method. An example is with the NANI clustering also among this tutorial set. All you need is to have the cluster labels similar to this format (frame index, cluster index); ``mdance.data.labels_60`` is a sample with 60 clusters from NANI:

:: 

   #frame,cluster
   0,0
   1,0
   2,1
   3,1
   4,2

4. HELM clustering
~~~~~~~~~~~~~~~~~~

`scripts/helm/run_helm.py <https://github.com/mqcomplab/MDANCE/blob/main/scripts/helm/run_helm.py>`_ will run HELM clustering on the dataset. The following parameters need to be specified in the script:

::

   # System parameters
   input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
   cluster_labels = data.labels_60                     # Labels of the initial clusters (e.g. from NANI)
   sieve = 1                                           # Use every sieve-th frame (1 = all frames)
   N_atoms = 50                                        # Number of atoms in the system

   # HELM parameters
   metric = 'MSD'                                      # Distance metric for clustering
   merge_scheme = 'intra'                              # Merge scheme {'inter', 'intra', 'half'}
   link = None                                         # Linkage method {None, 'ward'}
   n_clusters = 1                                      # Either n_clusters or eps must be set
   eps = None                                          # Either n_clusters or eps must be set
   trim_start = False                                  # Remove the least compact initial clusters first
   trim_val = None                                     # Either trim_val or trim_k must be set
   trim_k = 5                                          # Either trim_val or trim_k must be set
   save_pairwise_sim = False                           # Save the pairwise similarity matrix
   output_dir = 'outputs'                              # Folder for output files

   # Analysis parameters
   plot_metrics = True                                 # Plot DBI and CHI, and suggest values
   x_min = 5                                           # Only suggest values >= x_min (None = all)

Inputs
^^^^^^

System parameters
'''''''''''''''''
| ``input_traj_numpy``: The numpy array prepared from step 2.
| ``cluster_labels``: The cluster labels from step 3.
| ``sieve``: Reading every ``sieve`` frames from the trajectory.
| ``N_atoms``: The number of atoms used in the clustering.

HELM parameters
'''''''''''''''
| ``metric``: The metric used to calculate the similarity between frames (See ``mdance.tools.bts.extended_comparisons`` for details).
| ``merge_scheme``: The merging scheme to use. {``inter``, ``intra``, ``half``}. ``inter`` merges clusters with lowest interdistance. ``intra`` merges clusters with lowest intradistance. ``half`` merges clusters with the lowest combined MSD minus the average MSD of the two clusters.
| ``link`` *(optional)*: ``'ward'`` to merge the clusters with Ward linkage instead of ``merge_scheme``. Default is None.
| ``n_clusters``: The number of clusters to stop at. Either ``n_clusters`` or ``eps`` must be set.
| ``eps``: The largest distance at which two clusters can still be merged. Either ``n_clusters`` or ``eps`` must be set.
| ``trim_start`` *(optional)*: A boolean to remove the least compact initial clusters before merging, using ``trim_val`` (an MSD cutoff) or ``trim_k`` (the number of clusters to remove). Default is False.
| ``save_pairwise_sim`` *(optional)*: A boolean variable to indicate whether to save the pairwise similarity matrix. Default is False.
| ``output_dir``: The directory to store the results.

Analysis parameters
'''''''''''''''''''
| ``plot_metrics``: A boolean to plot the clustering scores and mark the suggested number of clusters (see step 5). If False, no analysis is done.
| ``x_min``: The smallest number of clusters that can be suggested. The whole hierarchy is still plotted. ``None`` considers all values.

Execution
^^^^^^^^^

.. code:: bash

   $ python run_helm.py

Outputs
^^^^^^^

Files are named after ``link``, or ``merge_scheme`` if ``link`` is None (e.g. ``intra``):

* ``outputs/helm_intra.pkl`` containing the clustering results.
* ``outputs/helm_intra_scores.csv`` containing the Calinski-Harabasz score, Davies-Bouldin score, and average MSD of the clusters for each number of clusters.

5. Get most optimal number of clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The clustering screening results are analyzed using the
Davies-Bouldin index (DBI) and the Calinski-Harabasz index (CHI). There are
two criteria to select the number of clusters: 

1. lowest DBI or highest CHI
2. maximum 2nd derivative of DBI or minimum 2nd derivative of CHI.

When ``plot_metrics = True``, ``run_helm.py`` does this analysis after clustering: it saves the DBI and CHI plots (e.g. ``outputs/helm_intra_scores_dbi.png`` and ``outputs/helm_intra_scores_chi.png``) with the suggested number of clusters marked.

The same analysis can be run on any scores csv with ``mdance.outputs.plot.plot_scores``.

6. Cluster Assignment
~~~~~~~~~~~~~~~~~~~~~~

`assign_labels.py <https://github.com/mqcomplab/MDANCE/blob/main/scripts/helm/assign_labels.py>`_ will assign cluster labels to the trajectory. The following parameters need to be specified in the script:

::

   # System parameters
   input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
   N_atoms = 50                                        # Number of atoms in the system
   sieve = 1                                           # Use every sieve-th frame (1 = all frames)

   # HELM parameters
   n_clusters = 10                                     # Number of clusters to assign
   pre_cluster_labels = data.labels_60                 # Labels of the initial clusters (e.g. from NANI)
   pickle_file = 'outputs/helm_intra.pkl'              # HELM output from run_helm.py
   metric = 'MSD'                                      # Distance metric for clustering
   n_structures = 11                                   # Frames saved per cluster, closest to its medoid
   output_dir = 'outputs'                              # Folder for output files

Inputs
^^^^^^

System parameters
'''''''''''''''''

| ``input_traj_numpy``: The numpy array prepared from step 2.
| ``N_atoms``: The number of atoms used in the clustering.
| ``sieve``: Reading every ``sieve`` frames from the trajectory.

HELM parameters
'''''''''''''''

| ``n_clusters``: The number of clusters to assign labels to. Use the most optimal number of clusters from analysis in step 5.
| ``pre_cluster_labels``: The cluster labels from step 3.
| ``pickle_file``: The clustering results from step 4.
| ``metric``: The metric used to calculate the similarity between frames (See ``mdance.tools.bts.extended_comparisons`` for details).
| ``n_structures``: The number of structures closest to the medoid to extract from each cluster.
| ``output_dir``: The directory to store the results.

Execution
^^^^^^^^^

.. code-block:: bash

    python assign_labels.py

Outputs
^^^^^^^

All files are saved in ``outputs/``, with ``{n_clusters}`` in the name (e.g. ``n_clusters = 10``). Clusters are numbered by size, cluster 0 being the largest.

* ``helm_labels_10.csv``: Contains the cluster of each frame, in frame order (frame index, cluster index). Frames not in any cluster are left out.
* ``helm_best_frames_10.csv``: Contains the ``n_structures`` frames closest to the medoid of each cluster, closest first (frame index, cluster index).
* ``helm_summary_10.csv``: Contains the population and MSD of each cluster (cluster index, fraction of total frames, number of frames, MSD); the header also gives the total number of frames.

7. Extract frames for each cluster (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`postprocessing.ipynb <../examples/postprocessing.html>`__
will use the indices from last step to extract the designated frames
from the original trajectory for each cluster.

A copy of this notebook can be found in ``$PATH/MDANCE/scripts/outputs/postprocessing.ipynb``.

Further Reading
~~~~~~~~~~~~~~~

For more information on the HELM algorithm, please refer to the `HELM
paper <https://doi.org/10.1021/acs.jcim.5c00539>`__.

Please Cite

.. code:: bibtex

   @article{chen_hierarchical_2025,
       title = {Hierarchical {Extended} {Linkage} {Method} ({HELM})'s {Deep} {Dive} into {Hybrid} {Clustering} {Strategies}},
       author = {Chen, Lexin and Santos, Jherome Brylle Woody and Gaza, Jokent and Perez, Alberto and Miranda-Quintana, Ramón Alain},
       journal = {Journal of Chemical Information and Modeling},
       volume = {65},
       number = {12},
       pages = {6209--6220},
       year = {2025},
       doi = {10.1021/acs.jcim.5c00539},
       url = {https://doi.org/10.1021/acs.jcim.5c00539},
   }
