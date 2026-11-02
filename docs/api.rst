mdance.cluster
======================

mdance.cluster.nani
--------------------------

.. automodule:: mdance.cluster.nani
   :members:
   :undoc-members:
   :show-inheritance:

mdance.cluster.equal
--------------------------
.. automodule:: mdance.cluster.equal
   :members:
   :undoc-members:
   :show-inheritance:

mdance.cluster.helm
--------------------------
.. automodule:: mdance.cluster.helm
   :members:
   :undoc-members:
   :show-inheritance:

mdance.cluster.shine
--------------------------
.. automodule:: mdance.cluster.shine
   :members:
   :undoc-members:
   :show-inheritance:

mdance.cluster.divine
--------------------------
.. automodule:: mdance.cluster.divine
   :members:
   :undoc-members:
   :show-inheritance:

mdance.cluster.mdbirch
--------------------------
.. automodule:: mdance.cluster.mdbirch
   :members:
   :undoc-members:
   :show-inheritance:

mdance.cluster.prism
--------------------------
.. automodule:: mdance.cluster.prism
   :members:
   :undoc-members:
   :show-inheritance:


mdance.cli
====================

mdance.cli.prime_sim
-----------------------

.. automodule:: mdance.cli.prime_sim
   :members:
   :undoc-members:
   :show-inheritance:

mdance.cli.prime_rep
-----------------------

.. automodule:: mdance.cli.prime_rep
   :members:
   :undoc-members:
   :show-inheritance:

mdance.data
====================

mdance.data.top
---------------

Sample topology files for testing Molecular Dynamics systems. Works with ``mdance.data.traj``.

mdance.data.traj
----------------

Sample trajectory files for testing Molecular Dynamics systems. Works with ``mdance.data.top``.

mdance.data.sim_traj_numpy
--------------------------

Numpy array of sample trajectory data for testing Molecular Dynamics systems. Extracted from ``mdance.data.top`` and ``mdance.data.traj``.

mdance.data.cc_sim
---------------------

Complementary similarity of sample trajectory data for testing Molecular Dynamics systems. 

mdance.data.trimmed_sim
-----------------------

Trimmed outliers of sample trajectory data for testing Molecular Dynamics systems.

mdance.data.blob_disk
---------------------

2D dataset with the multiple blobs shaped of a disk for testing 2D systems.

mdance.data.diamonds
---------------------
2D dataset shape of nine diamonds for testing 2D systems.

mdance.data.ellipses
---------------------
2D dataset shape of multiple ellipses for testing 2D systems.

mdance.data.labels_6
---------------------
Cluster labels of ``mdance.data.sim_traj_numpy`` from NANI *k*-means with 6 clusters (``init_type='comp_sim'``). Columns: frame index, cluster index.

mdance.data.labels_6_strat_all
------------------------------
Cluster labels of ``mdance.data.sim_traj_numpy`` from NANI *k*-means with 6 clusters (``init_type='strat_all'``). Columns: frame index, cluster index.

mdance.data.labels_6_strat_reduced
----------------------------------
Cluster labels of ``mdance.data.sim_traj_numpy`` from NANI *k*-means with 6 clusters (``init_type='strat_reduced'``). Columns: frame index, cluster index.

mdance.data.labels_60
---------------------
Cluster labels of ``mdance.data.sim_traj_numpy`` from NANI *k*-means with 60 clusters. Used as the initial clusters for HELM. Columns: frame index, cluster index.

mdance.data.best_frames_6
-------------------------
The 11 frames closest to the medoid of each cluster in ``mdance.data.labels_6``. Columns: frame index, cluster index. Used by the postprocessing example.

mdance.data.summary_6
---------------------
Population of each cluster in ``mdance.data.labels_6``. Columns: cluster index, fraction of total frames. Used by the PRIME example.

mdance.data.nani_screening
--------------------------
NANI screening of ``mdance.data.sim_traj_numpy`` for 5 to 30 clusters (``init_type='comp_sim'``). Columns: number of clusters, number of iterations, Calinski-Harabasz score, Davies-Bouldin score, average MSD. Can be plotted with ``mdance.outputs.plot.plot_scores``.

mdance.data.ala_pathways
------------------------
Folder of 80 alanine dipeptide pathways (``pathway_dih_0.csv`` to ``pathway_dih_79.csv``) with the phi and psi dihedral angles of each frame, for testing pathway clustering (SHINE and PRISM). Load them with ``glob.glob(str(mdance.data.ala_pathways / '*.csv'))``.

mdance.inputs
=====================

mdance.inputs.preprocess
-------------------------------

.. autofunction:: mdance.inputs.preprocess.gen_traj_numpy
.. autofunction:: mdance.inputs.preprocess.normalize_file



mdance.outputs
======================

mdance.outputs.postprocess
---------------------------------

.. automodule:: mdance.outputs.postprocess
   :members:
   :undoc-members:
   :show-inheritance:

mdance.outputs.plot
---------------------------------

.. automodule:: mdance.outputs.plot
   :members:
   :show-inheritance:


mdance.prime
====================

mdance.prime.sim\_calc
-----------------------------

.. automodule:: mdance.prime.sim_calc
   :members:
   :undoc-members:
   :show-inheritance:


mdance.prime.rep\_frames
-------------------------------

.. automodule:: mdance.prime.rep_frames
   :members:
   :undoc-members:
   :show-inheritance:


mdance.tools
====================

mdance.tools.bts
-----------------------

.. automodule:: mdance.tools.bts
   :members:
   :undoc-members:
   :show-inheritance:

mdance.tools.isim
---------------------------------

.. automodule:: mdance.tools.isim
   :members:
   :undoc-members:
   :show-inheritance:

mdance.tools.esim
---------------------------------

.. automodule:: mdance.tools.esim
   :members:
   :undoc-members:
   :show-inheritance: