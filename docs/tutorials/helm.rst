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

   # System info - EDIT THESE
   data_file = '../data/2D/blob_disk.csv'
   array = np.genfromtxt(data_file, delimiter=',')
   output_base_name = 'output_base_name'

Inputs
^^^^^^

System info
'''''''''''

| ``data_file`` is your input file with a 2D array. 
| ``array`` is the array is the loaded dataset from ``data_file``. This step can be changed according to the type of file format you have. However, ``array`` must be an array-like in the shape (number of samples, number of features).
| ``output_base_name`` is the base name for the output file. The output file will be saved as ``output_base_name.npy``. 

.. raw:: html

   </details>

3. Prior Clustering (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HELM requires a set of initial clusters to start with. You can start from any clustering method. An example is with the NANI clustering also among this tutorial set. All you need is to have the cluster labels similar to this format:

:: 

   #frame,cluster
   0,0
   1,0
   2,1
   3,1
   4,2

4. HELM clustering
~~~~~~~~~~~~~~~~~~

`scripts/helm/intra/run_helm.py <https://github.com/mqcomplab/MDANCE/blob/main/scripts/helm/intra/run_helm_intra.py>`_ will run HELM clustering on the dataset. The following parameters need to be specified in the script:

::

   # System info - EDIT THESE
   input_traj_numpy = data.sim_traj_numpy
   cluster_labels = '../labels_60.csv'
   sieve = 1
   N_atoms = 50                                    # Number of atoms in the system

   # HELM params - EDIT THESE
   metric = 'MSD'                                  # Default  
   N0 = 60                                         # How many clusters to start with
   final_target = 1                                # How many clusters to end with
   align_method = None                             # Default
   save_pairwise_sim = False                       # Default
   merging_scheme = 'inter'                        # {'inter', 'intra'}

.. _system-info-1:

Inputs
^^^^^^

System info
'''''''''''
| ``input_file``: The trajectory from step 1.
| ``cluster_labels``: The cluster labels from step 2.
| ``sieve``: Reading every ``sieve`` frames from the trajectory.
| ``N_atoms``: The number of atoms used in the clustering.

- **HELM params**
| ``metric``: The metric used to calculate the similarity between frames (See ``mdance.tools.bts.extended_comparisons`` for details).
| ``N0``: The number of clusters to start with.
| ``final_target``: The number of clusters to end with.
| ``align_method`` *(optional)*: The method to align the clusters. Default is None.
| ``save_pairwise_sim`` *(optional)*: A boolean variable to indicate whether to save the pairwise similarity matrix. Default is False.
| ``merging_scheme`` *(optional)*: The merging scheme to use. {``inter``, ``intra``}. ``inter`` merges clusters with lowest interdistance. ``intra`` merges clusters with lowest intradistance. Default is ``inter``.

Execution
^^^^^^^^^

.. code:: bash

   $ python run_helm_intra.py

Outputs
^^^^^^^

* Pickle file containing the clustering results.
* CSV file containing the Calinski-Harabasz and Davies-Bouldin scores for each number of clusters.

5. Get most optimal number of clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The clustering screening results will be analyzed using the
Davies-Bouldin index (DB). There are two criteria to select the number
of clusters: 

1. lowest DB
2. maximum 2nd derivative of DB.

`analysis notebook <https://github.com/mqcomplab/MDANCE/blob/main/scripts/helm/intra/analysis_db.ipynb>`__
contains step-by-step tutorial to analyze clustering screening results.

6. Cluster Assignment
~~~~~~~~~~~~~~~~~~~~~~

`assign_labels_intra.py <https://github.com/mqcomplab/MDANCE/blob/main/scripts/helm/intra/assign_labels_intra.py>`_ will assign cluster labels to the trajectory. The following parameters need to be specified in the script:

::

   # System info - EDIT THESE
   input_traj_numpy = data.sim_traj_numpy
   N_atoms = 50
   sieve = 1

   # HELM params - EDIT THESE
   n_clusters = 10
   pre_cluster_labels = '../labels_60.csv'
   pickle_file = 'inter-helm.pkl'
   metric = 'MSD'                                                      # Default
   extract_type = 'top'                                                # Default
   n_structures = 11                                                   # Default

.. _inputs-1:

Inputs
^^^^^^

.. _system-info-2:

System info
'''''''''''

| ``input_traj_numpy``: The normalized trajectory from step 1.
| ``N_atoms``: The number of atoms used in the clustering.
| ``sieve``: Reading every ``sieve`` frames from the trajectory.

HELM params
'''''''''''

| ``n_clusters``: The number of clusters to assign labels to. Use the most optimal number of clusters from analysis in step 4.
| ``pre_cluster_labels``: The cluster labels from step 2.
| ``pickle_file``: The clustering results from step 3.
| ``metric``: The metric used to calculate the similarity between frames (See ``mdance.tools.bts.extended_comparisons`` for details).
| ``extract_type``: The type of extraction method to use. {``top``, ``random``}. ``top`` means to extract the top ``n_structures`` from each cluster. ``random`` means to extract ``n_structures`` random structures from each cluster.
| ``n_structures``: The number of structures to extract from each cluster.

Execution
^^^^^^^^^

.. code-block:: bash

    python assign_labels_intra.py

Outputs
^^^^^^^

* ``helm_cluster_labels.csv``: Contains the cluster labels for each frame.
* ``helm_best_frames_indices.csv``: Contains the indices of the best or random frames to extract from each cluster.
* ``helm_summary.csv``: Contains the summary of the clustered population.

6. Extract frames for each cluster (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`postprocessing.ipynb <../examples/postprocessing.html>`__
will use the indices from last step to extract the designated frames
from the original trajectory for each cluster.

A copy of this notebook can be found in ``$PATH/MDANCE/scripts/outputs/postprocessing.ipynb``.

Further Reading
---------------

For more information on the HELM algorithm, please refer to the `HELM
paper <https://www.biorxiv.org/content/10.1101/2025.03.05.641742v1>`__.

Please Cite