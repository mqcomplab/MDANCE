.. image:: ../img/algorithms/NANI.png
   :width: 150


*k*-means NANI Tutorial
=======================

.. tip::

   New to NANI? Try the interactive `NANI in 60 seconds <../nani-in-60-seconds.html>`__ first.

.. youtube:: Y2DNnLseq9E
   :title: NANI video tutorial (MDANCE Ep. 1)

Overview
--------
This clustering tutorial is meant for datasets for **all** applications
(2D fingerprints, mass spectrometry imaging data, etc). Molecular
Dynamics Trajectory has a different treatment. If specific step is only
for Molecular Dynamics trajectory, it will be specified. Otherwise, it
is applicable for all datasets.

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

3. NANI Screening
~~~~~~~~~~~~~~~~~

`screen_nani.py <https://github.com/mqcomplab/MDANCE/blob/main/scripts/nani/screen_nani.py>`__ will
run NANI for a range of clusters and calculate cluster quality metrics.
For the best result, we recommend running NANI over a wide range of
number of clusters. The following parameters to be specified in the
script:

::

   # System parameters
   input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
   N_atoms = 50                                        # Number of atoms in the system
   sieve = 1                                           # Use every sieve-th frame (1 = all frames)

   # NANI parameters
   output_dir = 'outputs'                              # Folder for output files
   init_type = 'strat_all'                             # Seeding method {'strat_all', 'strat_reduced', 'comp_sim', 'div_select', 'k-means++', 'random'}
   metric = 'MSD'                                      # Distance metric for clustering
   start_n_clusters = 5                                # At least 2 clusters
   end_n_clusters = 30                                 # Maximum number of clusters

   # Analysis parameters
   plot_metrics = True                                 # Plot DBI and CHI, and suggest values
   x_min = None                                        # Only suggest values >= x_min (None = all)

Inputs
^^^^^^

System parameters
'''''''''''''''''

| ``input_traj_numpy`` is the numpy array prepared from step 1, if not it will be your loaded dataset. 
| ``N_atoms`` is the number of atoms used in the clustering. **For all non-Molecular Dynamics datasets, ``N_atoms=1``.**
| ``sieve`` takes every sieve-th frame from the trajectory for analysis.

NANI parameters
''''''''''''''''

| ``output_dir`` is the directory to store the clustering results.
| ``init_type`` is the selected seed selectors (See ``mdance.cluster.nani.KmeansNANI`` for details). 
| ``metric`` is the metric used to calculate the similarity between frames (See ``mdance.tools.bts.extended_comparisons`` for details).
| ``start_n_clusters`` is the starting number for screening. **This number must be at least 2**.
| ``end_n_clusters`` is the ending number for screening.

Analysis parameters
''''''''''''''''''''

| ``plot_metrics`` is a boolean to plot the screening scores and mark the suggested number of clusters (see step 4). If False, no analysis is done.
| ``x_min`` is the smallest number of clusters that can be suggested. The whole screening is still plotted. ``None`` considers all values.

Execution
^^^^^^^^^

Make sure your pwd is ``$PATH/MDANCE/scripts/nani``.

.. code:: bash

   $ python screen_nani.py

Outputs
^^^^^^^

``outputs/nani_{init_type}_scores.csv`` (e.g. ``outputs/nani_strat_all_scores.csv``) containing
the number of clusters, Calinski-Harabasz score, Davies-Bouldin score, average mean-square deviation (MSD) of the clusters, and number of iterations for that seed selector.

4. Analysis of NANI Screening Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The clustering screening results are analyzed using the
Davies-Bouldin index (DBI) and the Calinski-Harabasz index (CHI). There are
two criteria to select the number of clusters: 

1. lowest DBI or highest CHI
2. maximum 2nd derivative of DBI or minimum 2nd derivative of CHI.

When ``plot_metrics = True``, ``screen_nani.py`` does this analysis after the screening: it saves the DBI and CHI plots next to the screening csv (e.g. ``outputs/nani_strat_all_scores_dbi.png`` and ``outputs/nani_strat_all_scores_chi.png``) with the suggested number of clusters marked.

The same analysis can be run on any screening csv with ``mdance.outputs.plot.plot_scores``, as shown in the `cluster analysis example <../examples/analysis_db.html>`__.

5. Cluster Assignment
~~~~~~~~~~~~~~~~~~~~~

`assign_labels.py <https://github.com/mqcomplab/MDANCE/blob/master/scripts/nani/assign_labels.py>`__
will assign labels to the clusters for *k*-means clustering using the
initialization methods. The following parameters to be specified in the
script:

::

   # System parameters
   input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
   N_atoms = 50                                        # Number of atoms in the system
   sieve = 1                                           # Use every sieve-th frame (1 = all frames)

   # NANI parameters
   n_clusters = 6                                      # Number of clusters (pick from screen_nani.py)
   init_type = 'strat_all'                             # Seeding method, same as in screen_nani.py
   metric = 'MSD'                                      # Distance metric for clustering
   n_structures = 11                                   # Frames saved per cluster, closest to its medoid
   output_dir = 'outputs'                              # Folder for output files

Inputs
^^^^^^

System parameters
'''''''''''''''''

| ``input_traj_numpy`` is the numpy array prepared from step 1, if not it will be your loaded dataset. 
| ``N_atoms`` is the number of atoms used in the clustering. **For all non-Molecular Dynamics datasets, ``N_atoms=1``.**
| ``sieve`` takes every ``sieve``\ th frame from the trajectory for analysis.

*k*-means params
''''''''''''''''

| ``n_clusters`` is the number of clusters for labeling.
| ``init_type`` is the seed selector to use (See ``mdance.cluster.nani.KmeansNANI`` for details). 
| ``metric`` is the metric used to calculate the similarity between frames (See ``mdance.tools.bts.extended_comparisons`` for details).
| ``n_structures`` is the number of frames to extract from each cluster.
| ``output_dir`` is the directory to store the clustering results.

Execution
^^^^^^^^^

Make sure your pwd is ``$PATH/MDANCE/scripts/nani``.

.. code:: bash

   $ python assign_labels.py

Outputs
^^^^^^^

All files are saved in ``outputs/``, with ``{n_clusters}`` in the name (e.g. ``n_clusters = 6``). Clusters are numbered by size, cluster 0 being the largest.

* ``nani_best_frames_6.csv``: the ``n_structures`` frames closest to the medoid of each cluster (frame index, cluster index).
* ``nani_labels_6.csv``: the cluster label of each frame (frame index, cluster index).
* ``nani_summary_6.csv``: the population and MSD of each cluster (cluster index, fraction of total frames, number of frames, MSD); the header also gives the total number of frames.

6. Extract frames for each cluster (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`postprocessing.ipynb <../examples/postprocessing.html>`__
will use the indices from last step to extract the designated frames
from the original trajectory for each cluster.

A copy of this notebook can be found in ``$PATH/MDANCE/scripts/outputs/postprocessing.ipynb``.

Further Reading
~~~~~~~~~~~~~~~

For more information on the NANI algorithm, please refer to the `NANI
paper <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00308>`__.

Please Cite

.. code:: bibtex

   @article{chen_k-means_2024,
       title = {k-{Means} {NANI}: {An} {Improved} {Clustering} {Algorithm} for {Molecular} {Dynamics} {Simulations}},
       author = {Chen, Lexin and Roe, Daniel R. and Kochert, Matthew and Simmerling, Carlos and Miranda-Quintana, Ramón Alain},
       journal = {Journal of Chemical Theory and Computation},
       volume = {20},
       number = {13},
       pages = {5583--5597},
       year = {2024},
       doi = {10.1021/acs.jctc.4c00308},
       url = {https://doi.org/10.1021/acs.jctc.4c00308},
   }