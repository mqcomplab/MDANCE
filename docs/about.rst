.. image:: _static/mdance.png
   :width: 500
   :alt: MDANCE
   :class: only-light

.. image:: _static/mdance-dark.png
   :width: 500
   :alt: MDANCE
   :class: only-dark

MDANCE: Molecular Dynamics Analysis with *N*-ary Clustering Ensembles
=====================================================================

Cluster millions of MD frames in minutes, with the same result on every run.

|pypi| |python| |docs| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/mdance
   :target: https://pypi.org/project/mdance/
   :alt: PyPI version
.. |python| image:: https://img.shields.io/badge/python-%E2%89%A5%203.10-blue
   :target: https://pypi.org/project/mdance/
   :alt: Python 3.10 or newer
.. |docs| image:: https://readthedocs.org/projects/mdance/badge/?version=latest
   :target: https://mdance.readthedocs.io/en/latest/
   :alt: Documentation
.. |license| image:: https://img.shields.io/github/license/mqcomplab/MDANCE
   :target: https://github.com/mqcomplab/MDANCE/blob/main/LICENSE
   :alt: License

Why MDANCE
----------

Most clustering tools compare frames two at a time. That breaks down on modern simulations:

- **Quadratic cost.** Comparing every frame with every other frame means doubling the trajectory quadruples the work.
- **Memory.** A full RMSD matrix for a million-frame trajectory takes terabytes before clustering even starts.
- **Poor initialization.** Random seeding gives suboptimal clusters, and different clusters on every run.
- **Hard follow-up questions.** Finding the dominant pathways, or the native structure, in a large ensemble is difficult.

MDANCE is built on *n*-ary (extended) similarity, which compares all frames at once and finds high- and low-density regions in linear time. Its algorithms:

- scale linearly with the number of frames, from O(*N*\ :sup:`2`) to O(*N*), without pairwise distance matrices;
- use deterministic seeding, so the same input gives the same clusters;
- are designed for structural biology and validated against experimental structures.

Researchers use MDANCE to find biologically relevant conformations for drug discovery, study mechanisms through pathway analysis, validate models against experimental structures, and analyze very large simulation datasets.

Installation
------------

.. code-block:: bash

   pip install mdance

MDANCE requires Python 3.10 or newer. To install from source:

.. code-block:: bash

   git clone https://github.com/mqcomplab/MDANCE.git
   cd MDANCE
   pip install -e .

Quick start
-----------

Cluster the example trajectory included with MDANCE (6,001 frames of 50 atoms) with NANI seeding and *k*-means:

.. code-block:: python

   import numpy as np
   from sklearn.cluster import KMeans
   from mdance import data
   from mdance.cluster.nani import KmeansNANI

   X = np.load(data.sim_traj_numpy)

   # Deterministic NANI seeds
   nani = KmeansNANI(data=X, n_clusters=6, metric='MSD', N_atoms=50,
                     init_type='strat_reduced', percentage=10)
   centroids = nani.initiate_kmeans()

   # k-means from the NANI seeds gives the same labels every run
   labels = KMeans(n_clusters=6, init=centroids, n_init=1).fit(X).labels_

To use your own trajectory, convert it to a NumPy array first (see :doc:`examples/preprocessing`). Every algorithm also has ready-to-run scripts in `scripts/ <https://github.com/mqcomplab/MDANCE/tree/main/scripts>`__.

Algorithms
----------

.. |nani| image:: img/algorithms/NANI.png
   :width: 64
.. |helm| image:: img/algorithms/HELM.png
   :width: 64
.. |divine| image:: img/algorithms/DIVINE.png
   :width: 64
.. |mdbirch| image:: img/algorithms/mdBIRCH.png
   :width: 64
.. |shine| image:: img/algorithms/SHINE.png
   :width: 64
.. |prism| image:: img/algorithms/PRISM.png
   :width: 64
.. |equal| image:: img/algorithms/eQual.png
   :width: 64
.. |prime| image:: img/algorithms/PRIME.png
   :width: 64

.. list-table::
   :header-rows: 1
   :widths: 16 34 20 12 18
   :class: algo-table

   * - Algorithm
     - What it does
     - Best for
     - Tutorial
     - Paper
   * - |nani|

       **NANI**
     - Deterministic *k*-means seeding that finds metastable states
     - Simple analysis
     - :doc:`Tutorial <tutorials/nani>`
     - `JCTC 2024 <https://doi.org/10.1021/acs.jctc.4c00308>`__, `JCIM 2026 <https://doi.org/10.1021/acs.jcim.5c02741>`__
   * - |helm|

       **HELM**
     - Hierarchical clustering at *k*-means cost
     - High-resolution hierarchies
     - :doc:`Tutorial <tutorials/helm>`
     - `JCIM 2025 <https://doi.org/10.1021/acs.jcim.5c00539>`__
   * - |divine|

       **DIVINE**
     - Top-down divisive clustering, every resolution in one pass
     - Fast hierarchies
     - `Tutorial <https://github.com/mqcomplab/MDANCE/blob/main/tutorials/divine.md>`__
     - `JCIM 2026 <https://doi.org/10.1021/acs.jcim.5c02740>`__
   * - |mdbirch|

       **mdBIRCH**
     - Online clustering with an RMSD-calibrated merge test
     - Very large or streaming trajectories
     - `Tutorial <https://github.com/mqcomplab/MDANCE/blob/main/tutorials/mdbirch.md>`__
     - `JCTC 2026 <https://doi.org/10.1021/acs.jctc.6c00221>`__
   * - |shine|

       **SHINE**
     - Finds the dominant pathways in enhanced-sampling data
     - Pathway exploration
     - `Tutorial <https://github.com/mqcomplab/MDANCE/blob/main/tutorials/shine.md>`__
     - `JCIM 2025 <https://doi.org/10.1021/acs.jcim.5c00240>`__
   * - |prism|

       **PRISM**
     - Compares pathways by their structural medoids
     - Multi-pathway comparison
     - `Tutorial <https://github.com/mqcomplab/MDANCE/blob/main/tutorials/prism.md>`__
     - `Digital Discovery 2026 <https://doi.org/10.1039/d6dd00288a>`__
   * - |equal|

       **eQual**
     - Radial threshold clustering in linear time
     - Clusters of a set radius
     - :doc:`Tutorial <tutorials/equal>`
     - `JCIM 2025 <https://doi.org/10.1021/acs.jcim.4c02341>`__
   * - |prime|

       **PRIME**
     - Retrieves the native structure from a clustered ensemble
     - Structure selection
     - :doc:`Tutorial <tutorials/prime>`
     - `JCTC 2024 <https://doi.org/10.1021/acs.jctc.4c00362>`__

NANI: *k*-means initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Solves the seed selection problem of *k*-means: picks diverse initial centroids from high-density regions using *n*-ary comparisons.
- Gives compact, well-separated clusters that find metastable states, with consistent populations across replicates.
- Clusters 1.5 million HP35 frames in about 40 minutes.

HELM: hybrid hierarchical clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Combines *k*-means efficiency with hierarchical flexibility using *n*-ary difference functions.
- Keeps the cost of *k*-means while allowing any partition: the hierarchy is built without a pairwise distance matrix, so any number of clusters can be taken from one run.
- Clusters 1.5 million frames in about 34 minutes, where classic hierarchical clustering takes 29 hours.

DIVINE: divisive hierarchical clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Splits clusters top-down, choosing the next cluster to split by a weighted variance criterion.
- Uses NANI anchors, so every split is deterministic, and never builds a pairwise distance matrix.
- Explores every resolution in a single pass.
- Matches or exceeds bisecting *k*-means quality in less time.

mdBIRCH: online clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Adapts the BIRCH CF-tree to MD data and processes frames as they arrive.
- The merge test is calibrated to RMSD, so its threshold has a physical meaning.
- Avoids pairwise distance matrices and scales near-linearly with the number of frames: hundreds of thousands of frames on a single CPU core in seconds.
- Two protocols: runs anchored to a chosen RMSD, and a blind sweep over thresholds.

SHINE: pathway clustering
~~~~~~~~~~~~~~~~~~~~~~~~~

- Groups pathways from multiple simulations, such as enhanced-sampling runs, and picks the most representative pathway in each group.
- Combines *n*-ary similarity with tools from cheminformatics to reveal the dominant transformation mechanisms.
- Costs less than Fréchet-distance approaches.
- Applied to alanine dipeptide and adenylate kinase.

PRISM: pathway representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Summarizes each pathway by a compact set of structural medoids.
- Clusters pathways by weighted average Hausdorff distance with hierarchical clustering.
- Offers three ways to build the medoids: shared global medoids, per-pathway medoids, or a two-stage refinement.
- Designed for comparing and classifying many pathways, for example from enhanced sampling.

eQual: radial threshold clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Turns O(*N*\ :sup:`2`) radial threshold clustering into an O(*N*) algorithm with new seed selection and tie-breaking.
- Selects seeds with *k*-means++ and breaks ties with extended similarity indices, for deterministic results.
- Uses no memory-intensive pairwise RMSD matrix.
- Gives compact, well-separated clusters of the same quality as the original method.

PRIME: native structure prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Uses *n*-ary similarity across clusters to pick the most native-like frame of an ensemble.
- Mapped all structural motifs in benchmark studies, and consistently identified native structures within 2 Å RMSD of experimental data.

.. figure:: img/2k2e.png
   :width: 300
   :align: center
   :alt: PRIME structure of 2k2e superposed on the experimental structure

   2k2e: structure from PRIME (yellow) superposed on the experimental native structure (blue).

Citing MDANCE
-------------

If you use MDANCE, please cite the paper of each algorithm you use (linked in the table above). BibTeX entries for all papers are in `CITATION.bib <https://github.com/mqcomplab/MDANCE/blob/main/CITATION.bib>`__.

Contributing
------------

We welcome collaborations and contributions, whether you are a computational biologist with new analysis needs, a method developer who wants to extend the framework, or a structural biologist with a challenging dataset.

- Open an `issue <https://github.com/mqcomplab/MDANCE/issues>`__ for bug reports or feature requests.
- Submit a pull request with improvements.
- Get in touch to discuss a research collaboration.

Funding
-------

This research was supported by the National Institute of General Medical Sciences of the National Institutes of Health under award number R35GM150620.

License
-------

MDANCE is released under the `MIT License <https://github.com/mqcomplab/MDANCE/blob/main/LICENSE>`__.
