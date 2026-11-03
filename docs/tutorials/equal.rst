.. image:: ../img/algorithms/eQual.png
   :width: 150

eQual - Extended Quality Clustering
===================================

Overview
--------
eQual is a quality clustering algorithm that use the radial threshold to grow the cluster to maximize similarity between members in a cluster. It is an extension of the Radial Threshold Clustering algorithm (`Daura and Oscar Conchillo-Solé <https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.2c01079>`_). eQual has improved with new seed selection methods and tie-breaking criteria.

eQual selects the seed to start the clustering. It then grows the cluster by adding the neighbors within a threshold away from the seed. This threshold is calculated using the mean-square deviation from the seed. The iteration continues until it runs out of neighbors and chosen neighbors will be removed from original dataset. A new iteration begins and selects medoid and its neighbors from the available dataset. If user selects multiple medoids, then the medoid that proposed the densest and most similar cluster will be chosen. The process repeats and the algorithm terminates when the dataset is empty.

This clustering tutorial is meant for datasets for all applications (2D fingerprints, mass spectrometry imaging data, etc). Molecular Dynamics Trajectory has a different treatment. If specific step is only for Molecular Dynamics trajectory, it will be specified. Otherwise, it is applicable for all datasets.

Tutorial
--------

1. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~

Clone the MDANCE repository if you haven't already.

.. code:: bash

   $ git clone https://github.com/mqcomplab/MDANCE.git
   $ cd MDANCE/scripts/equal


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

3. eQual Screening
~~~~~~~~~~~~~~~~~~
`scripts/equal/screen_equal.py <https://github.com/mqcomplab/MDANCE/blob/master/scripts/equal/screen_equal.py>`_ will screen eQual clustering for multiple thresholds and give the most optimal threshold. For the best result, we recommend screening eQual with a wide range of threshold values.  
*Depending on the number of samples or features, consider sieving over wide threshold range. For large dataset, please submit this as a job instead of running on command line.* 
The following parameters to be specified in the script:

::

    # System parameters
    input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
    N_atoms = 50                                        # Number of atoms in the system
    sieve = 1                                           # Use every sieve-th frame (1 = all frames)

    # eQual parameters
    metric = 'MSD'                                      # Distance metric for clustering
    n_seeds = 3                                         # Seeds tried per iteration (int, or fraction of the data)
    check_sim = True                                    # Reject clusters less similar than sim_threshold
    reject_lowd = True                                  # Reject clusters smaller than min_samples
    sim_threshold = 16                                  # Largest similarity value a cluster may have
    min_samples = 10                                    # Minimum cluster size

    # Screening parameters
    start_threshold = 5                                 # First threshold to screen
    end_threshold = 6                                   # Last threshold to screen
    step = 0.1                                          # Step between thresholds
    save_clusters = False                               # Save the clusters of each threshold as .pkl
    output_dir = 'outputs'                              # Folder for output files

    # Analysis parameters
    plot_metrics = True                                 # Plot DBI and CHI, and suggest values
    x_min = None                                        # Only suggest values >= x_min (None = all)

Inputs
^^^^^^
System parameters
'''''''''''''''''

| `input_traj_numpy` is the numpy array prepared from step 1, if not it will be your loaded dataset. 
| `N_atoms` is the number of atoms used in the clustering. **For all non-Molecular Dynamics datasets, this is 1.** 
| `sieve` takes every sieve-th frame from the trajectory for analysis. 

eQual params
^^^^^^^^^^^^

| `metric` is the metric used to calculate the similarity between frames (See ``mdance.tools.bts.extended_comparisons`` for details). 
| `n_seeds` is the is the number of seeds selected per iteration. If `n_medoids` is greater than 1, then multiple clusters will be proposed; the cluster with the densest and greatest similarity of members will be selected. Performance time will increase with more seeds. 
| `check_sim` is boolean to check the similarity of the seed to the cluster. 
| `reject_lowd` is boolean to reject low density clusters. `sim_threshold` needs to be specified. 
| `sim_threshold` is the similarity threshold to reject less compact clusters. 
| `min_samples` is the minimum cluster size to reject low density clusters. Default is 10. 

Radial threshold screening params
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| `start_threshold` is the starting value `r_theshold` for screening range. 
| `end_threshold` is the ending value of `r_theshold` screening range. 
| `step` is the increment of the `r_theshold` screening range. 
| `save_clusters` is boolean to save the cluster dictionary of each threshold in ``outputs/clusters_dict/``. Default is False.
| `output_dir` is the directory to store the results. 

Analysis parameters
^^^^^^^^^^^^^^^^^^^

| `plot_metrics` is boolean to plot the screening results and mark the suggested thresholds (see step 4). If False, no analysis is done. 
| `x_min` is the smallest threshold that can be suggested. The whole screening is still plotted. `None` considers all values. 

Execution
^^^^^^^^^
.. code:: bash

    $ python screen_equal.py

Outputs
^^^^^^^
- ``outputs/equal_screening.csv`` with the number of clusters and the population of the top 10 clusters for each threshold value.
- ``outputs/equal_scores.csv`` with the number of clusters, Calinski-Harabasz (CH) score, Davies-Bouldin (DB) score (two cluster quality indices), and average MSD of the clusters for each threshold value.
- ``outputs/clusters_dict/equal_threshold_{threshold}.pkl`` with the clusters of each threshold, if ``save_clusters = True``.

4. eQual Screening Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The clustering screening results are analyzed using the
Davies-Bouldin index (DBI) and the Calinski-Harabasz index (CHI). There are
two criteria to select the number of clusters: 

1. lowest DBI or highest CHI
2. maximum 2nd derivative of DBI or minimum 2nd derivative of CHI.

When ``plot_metrics = True``, ``screen_equal.py`` analyzes the screening results after the screening:

- ``outputs/equal_scores_dbi.png`` and ``outputs/equal_scores_chi.png`` plot the DBI and CHI for each threshold. The suggested thresholds are marked.
- ``outputs/equal_screening_nclusters.png`` plots the number of clusters for each threshold and marks the threshold with the most clusters.

The same analysis can be run on any screening csv with ``mdance.outputs.plot`` (``plot_scores`` and ``plot_n_clusters``).

5. Assign labels to the frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`scripts/equal/assign_labels.py <https://github.com/mqcomplab/MDANCE/blob/master/scripts/equal/assign_labels.py>`_ will assign cluster for each frame. The following parameters to be specified in the script:

::

    # System parameters
    input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
    N_atoms = 50                                        # Number of atoms in the system
    sieve = 1                                           # Use every sieve-th frame (1 = all frames)

    # eQual parameters
    metric = 'MSD'                                      # Distance metric for clustering
    n_seeds = 3                                         # Seeds tried per iteration (int, or fraction of the data)
    check_sim = True                                    # Reject clusters less similar than sim_threshold
    reject_lowd = True                                  # Reject clusters smaller than min_samples
    sim_threshold = 16                                  # Largest similarity value a cluster may have
    min_samples = 10                                    # Minimum cluster size

    # Output parameters
    threshold = 5.80                                    # Cluster radius threshold (pick from screen_equal.py)
    n_structures = 11                                   # Frames saved per cluster, closest to its medoid
    open_clusters = None                                # Saved clusters .pkl to reuse (None = recluster)
    output_dir = 'outputs'                              # Folder for output files

Inputs - New parameters
^^^^^^^^^^^^^^^^^^^^^^^

| `threshold` is the threshold value to use for clustering, e.g. chosen from the screening plots. 
| `n_structures` is the number of closest structures (from the medoid) to extract from each cluster. 
| `open_clusters` is a saved cluster dictionary to open (e.g. ``outputs/clusters_dict/equal_threshold_5.8.pkl`` from the screening with ``save_clusters = True``). If `None`, it will run the clustering algorithm. 
| `output_dir` is the directory to store the results. 

Execution
^^^^^^^^^
.. code:: bash

    python assign_labels.py

Outputs
^^^^^^^

All files are saved in ``outputs/``, with the ``threshold`` in the name (e.g. ``threshold = 5.8``). Clusters are numbered by size, cluster 0 being the largest.

| ``equal_best_frames_5.8.csv`` contains the ``n_structures`` most representative frames of each cluster, closest to the medoid first (frame index, cluster index). 
| ``equal_labels_5.8.csv`` contains the cluster of each frame, in frame order (frame index, cluster index). Frames rejected as low density are not assigned to any cluster and are left out.
| ``equal_summary_5.8.csv`` contains the population and MSD of each cluster (cluster index, fraction of total frames, number of frames, MSD); the header also gives the total number of frames. 

6. Extract frames for each cluster (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`postprocessing.ipynb <../examples/postprocessing.html>`__
will use the indices from last step to extract the designated frames
from the original trajectory for each cluster.

A copy of this notebook can be found in ``$PATH/MDANCE/scripts/outputs/postprocessing.ipynb``.


Further Reading
~~~~~~~~~~~~~~~

For more information on the eQual algorithm, please refer to the `eQual
paper <https://pubs.acs.org/doi/10.1021/acs.jcim.4c02341>`__.

Please Cite

.. code:: bibtex

   @article{chen_extended_2025,
       title = {Extended {Quality} ({eQual}): {Radial} {Threshold} {Clustering} {Based} on n-ary {Similarity}},
       author = {Chen, Lexin and Smith, Micah and Roe, Daniel R. and Miranda-Quintana, Ramón Alain},
       journal = {Journal of Chemical Information and Modeling},
       volume = {65},
       number = {10},
       pages = {5062--5070},
       year = {2025},
       doi = {10.1021/acs.jcim.4c02341},
       url = {https://doi.org/10.1021/acs.jcim.4c02341},
   }
    