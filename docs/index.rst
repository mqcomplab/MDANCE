.. image:: _static/mdance.png
   :width: 420
   :alt: MDANCE
   :class: only-light

.. image:: _static/mdance-dark.png
   :width: 420
   :alt: MDANCE
   :class: only-dark

Molecular Dynamics Analysis with *N*\ ‑ary Clustering Ensembles
===============================================================

MDANCE is a flexible *n*-ary clustering package for Molecular Dynamics trajectories. Cluster millions of frames in minutes, with the same result on every run.

.. code-block:: bash
   :class: install-cmd

   pip install mdance

.. container:: hero-buttons

   .. button-ref:: about
      :ref-type: doc
      :color: primary

      Get started

   .. button-ref:: api
      :ref-type: doc
      :color: primary
      :outline:

      API reference

Algorithms
----------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: NANI
      :img-top: img/algorithms/NANI.png
      :link: tutorials/nani
      :link-type: doc
      :class-card: algo-card

      Deterministic *k*-means seeding

   .. grid-item-card:: HELM
      :img-top: img/algorithms/HELM.png
      :link: tutorials/helm
      :link-type: doc
      :class-card: algo-card

      Hierarchical clustering at *k*-means cost

   .. grid-item-card:: DIVINE
      :img-top: img/algorithms/DIVINE.png
      :link: https://github.com/mqcomplab/MDANCE/blob/main/tutorials/divine.md
      :class-card: algo-card

      Top-down divisive clustering

   .. grid-item-card:: mdBIRCH
      :img-top: img/algorithms/mdBIRCH.png
      :link: https://github.com/mqcomplab/MDANCE/blob/main/tutorials/mdbirch.md
      :class-card: algo-card

      Online clustering for large trajectories

   .. grid-item-card:: SHINE
      :img-top: img/algorithms/SHINE.png
      :link: https://github.com/mqcomplab/MDANCE/blob/main/tutorials/shine.md
      :class-card: algo-card

      Dominant pathways in enhanced sampling

   .. grid-item-card:: PRISM
      :img-top: img/algorithms/PRISM.png
      :link: https://github.com/mqcomplab/MDANCE/blob/main/tutorials/prism.md
      :class-card: algo-card

      Pathway comparison by structural medoids

   .. grid-item-card:: eQual
      :img-top: img/algorithms/eQual.png
      :link: tutorials/equal
      :link-type: doc
      :class-card: algo-card

      Radial threshold clustering in linear time

   .. grid-item-card:: PRIME
      :img-top: img/algorithms/PRIME.png
      :link: tutorials/prime
      :link-type: doc
      :class-card: algo-card

      Native structure from a clustered ensemble

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   About <about>
   Getting started <getting_started>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Algorithms

   NANI <tutorials/nani>
   HELM <tutorials/helm>
   DIVINE <https://github.com/mqcomplab/MDANCE/blob/main/tutorials/divine.md>
   mdBIRCH <https://github.com/mqcomplab/MDANCE/blob/main/tutorials/mdbirch.md>
   SHINE <https://github.com/mqcomplab/MDANCE/blob/main/tutorials/shine.md>
   PRISM <https://github.com/mqcomplab/MDANCE/blob/main/tutorials/prism.md>
   eQual <tutorials/equal>
   PRIME <tutorials/prime>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   NANI in 60 seconds <examples/nani101>
   Preprocessing <examples/preprocessing>
   Cluster analysis <examples/analysis_db>
   Postprocessing <examples/postprocessing>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   API reference <api>
