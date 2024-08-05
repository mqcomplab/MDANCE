.. raw:: html

    <p align="center">
    <img src="../_static/nani-logo.PNG" width="150" height=auto align="center"></a></p>


*k*-means NANI Tutorial
=======================

.. contents::
   :local:
   :depth: 2

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

3. NANI Screening
~~~~~~~~~~~~~~~~~

`screen_nani.py <https://github.com/mqcomplab/MDANCE/blob/main/scripts/nani/screen_nani.py>`__ will
run NANI for a range of clusters and calculate cluster quality metrics.
For the best result, we recommend running NANI over a wide range of
number of clusters. The following parameters to be specified in the
script:

::

   # System info
   input_traj_numpy = '../../data/md/backbone.npy'
   N_atoms = 50
   sieve = 1

   # NANI parameters
   output_dir = 'outputs'                        
   init_types = ['comp_sim']
   metric = 'MSD'
   start_n_clusters = 2
   end_n_clusters = 30

.. _system-info-1:

Inputs
^^^^^^

System info
'''''''''''

| ``input_traj_numpy`` is the numpy array prepared from step 1, if not it will be your loaded dataset. 
| ``N_atoms`` is the number of atoms used in the clustering. **For all non-Molecular Dynamics datasets, ``N_atoms=1``.**
| ``sieve`` takes every sieve-th frame from the trajectory for analysis.

NANI parameters
''''''''''''''''

| ``output_dir`` is the directory to store the clustering results.
| ``init_types`` is a **list** of selected seed selectors. User can input one or multiple. Each seed selector will have results in a separate file. ``metric`` is the metric used to calculate the similarity between frames (See ``mdance.tools.bts.extended_comparisons`` for details).
| ``start_n_clusters`` is the starting number for screening. **This number must be greater than 2**.
| ``end_n_clusters`` is the ending number for screening.

Execution
^^^^^^^^^

Make sure your pwd is ``$PATH/MDANCE/scripts/nani``.

.. code:: bash

   $ python screen_nani.py

Outputs
^^^^^^^

csv file containing the number of clusters and the corresponding number
of iterations, Callinski-Harabasz score, Davies-Bouldin score, and
average mean-square deviation for that seed selector.

4. Analysis of NANI Screening Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The clustering screening results will be analyzed using the
Davies-Bouldin index (DB). There are two criteria to select the number
of clusters: 

1. lowest DB
2. maximum 2nd derivative of DB.

`analysis notebook <../examples/analysis_db.html>`__
contains step-by-step tutorial to analyze clustering screening results.

A copy of this notebook can be found in ``$PATH/MDANCE/scripts/nani/analysis_db.ipynb``.

5. Cluster Assignment
~~~~~~~~~~~~~~~~~~~~~

`assign_labels.py <https://github.com/mqcomplab/MDANCE/blob/master/scripts/nani/assign_labels.py>`__
will assign labels to the clusters for *k*-means clustering using the
initialization methods. The following parameters to be specified in the
script:

::

   # System info - EDIT THESE
   input_traj_numpy = '../../data/md/backbone.npy'
   N_atoms = 50
   sieve = 1

   # K-means params - EDIT THESE
   n_clusters = 6
   init_type = 'comp_sim'                                              
   metric = 'MSD'                                                      
   n_structures = 11                                                   
   output_dir = 'outputs'                                              

.. _inputs-1:

Inputs
^^^^^^

.. _system-info-2:

System info
'''''''''''

| ``input_traj_numpy`` is the numpy array prepared from step 1, if not it will be your loaded dataset. 
| ``N_atoms`` is the number of atoms used in the clustering. **For all non-Molecular Dynamics datasets, ``N_atoms=1``.**
| ``sieve`` takes every ``sieve``\ th frame from the trajectory for analysis.

*k*-means params
''''''''''''''''

| ``n_clusters`` is the number of clusters for labeling.
| ``init_type`` is the seed selector to use.
| ``metric`` is the metric used to calculate the similarity between frames (See
| ``mdance.tools.bts.extended_comparisons`` for details).
| ``n_structures`` is the number of frames to extract from each cluster.
| ``output_dir`` is the directory to store the clustering results.

.. _execution-1:

Execution
^^^^^^^^^

Make sure your pwd is ``$PATH/MDANCE/scripts/nani``.

.. code:: bash

   $ python assign_labels.py

.. _outputs-1:

Outputs
^^^^^^^

* csv file containing the indices of the best frames in each cluster.
* csv file containing the cluster labels for each frame.
* csv file containing the population of each cluster.

6. Extract frames for each cluster (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`postprocessing.ipynb <../examples/postprocessing.html>`__
will use the indices from last step to extract the designated frames
from the original trajectory for each cluster.

A copy of this notebook can be found in ``$PATH/MDANCE/scripts/outputs/postprocessing.ipynb``.