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

   .. button-link:: nani-in-60-seconds.html
      :color: primary
      :outline:

      Learn NANI in 60 seconds

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
      :link: tutorials/divine
      :link-type: doc
      :class-card: algo-card

      Top-down divisive clustering

   .. grid-item-card:: mdBIRCH
      :img-top: img/algorithms/mdBIRCH.png
      :link: tutorials/mdbirch
      :link-type: doc
      :class-card: algo-card

      Online clustering for large trajectories

   .. grid-item-card:: SHINE
      :img-top: img/algorithms/SHINE.png
      :link: tutorials/shine
      :link-type: doc
      :class-card: algo-card

      Dominant pathways in enhanced sampling

   .. grid-item-card:: PRISM
      :img-top: img/algorithms/PRISM.png
      :link: tutorials/prism
      :link-type: doc
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
   DIVINE <tutorials/divine>
   mdBIRCH <tutorials/mdbirch>
   SHINE <tutorials/shine>
   PRISM <tutorials/prism>
   eQual <tutorials/equal>
   PRIME <tutorials/prime>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   NANI in 60 seconds <https://mdance.readthedocs.io/en/latest/nani-in-60-seconds.html>
   NANI example (code) <examples/nani101>
   Preprocessing <examples/preprocessing>
   Cluster analysis <examples/analysis_db>
   Postprocessing <examples/postprocessing>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   API reference <api>
