About MDANCE
======================================================================
Molecular Dynamics Analysis with *N*-ary Clustering Ensembles (MDANCE) is a flexible 
*n*-ary clustering package that provides a set of tools for clustering Molecular 
Dynamics trajectories. The package is designed to be modular and extensible, allowing 
for the addition of new clustering algorithms and similarity metrics. Research contained 
in this package was supported by the National Institute of General Medical Sciences of 
the National Institutes of Health under award number R35GM150620.

.. contents::
   :local:
   :depth: 2

Background
----------

Clustering Algorithms
---------------------
NANI
~~~~

*k*-Means *N*-Ary Natural Initiation (NANI) is an algorithm for
selecting initial centroids for *k*-Means clustering. NANI is an
extension of the *k*-Means++ algorithm. NANI stratifies the data to high
density region and perform diversity selection on top of the it to
select the initial centroids. This is a deterministic algorithm that
will always select the same initial centroids for the same dataset and
improve on *k*-means++ by reducing the number of iterations required to
converge and improve the clustering quality.

For more information on the NANI algorithm, please refer to the `NANI
paper <https://doi.org/10.1021/acs.jctc.4c00308>`__.

Clustering Postprocessing
-------------------------
PRIME
~~~~~

Protein Retrieval via Integrative Molecular Ensembles (PRIME) is a novel
algorithm that predicts the native structure of a protein from
simulation or clustering data. These methods perfectly mapped all the
structural motifs in the studied systems and required unprecedented
linear scaling.

For more information on the PRIME algorithm, please refer to the `PRIME
paper <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00362>`__.

Contributing
------------
Contributions to MDANCE are welcome! 

License
-------
MDANCE is released under the MIT License. See the LICENSE file in the project repository for more details.

Citing MDANCE
-------------
If you use MDANCE in your research, please cite the following paper:

.. code-block:: bibtex

    @article{chen_k-means_2024,
        title = {k-Means NANI: An Improved Clustering Algorithm for Molecular Dynamics Simulations},
        issn = {1549-9618},
        url = {https://doi.org/10.1021/acs.jctc.4c00308},
        doi = {10.1021/acs.jctc.4c00308},
        journal = {Journal of Chemical Theory and Computation},
        author = {Chen, Lexin and Roe, Daniel R. and Kochert, Matthew and Simmerling, Carlos and Miranda-Quintana, Ramón Alain},
        month = jun,
        year = {2024},
        note = {Publisher: American Chemical Society},
    }

Contact
-------
For questions or support, please contact us.