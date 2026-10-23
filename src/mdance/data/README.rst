About the simulation data
=========================

``aligned_tau.pdb`` in this directory corresponds to simulation data from this GitHub repository:
`github.com/LQCT/BitQT/tree/master/examples <https://github.com/LQCT/BitQT/tree/master/examples>`_.

``aligned_1000_tau.dcd`` is the original trajectory from that repository aligned to the 1000th frame using CPPTRAJ in AmberTools.

``backbone.npy`` uses ``scripts/inputs/preprocessing.ipynb`` to create a numpy array of the backbone atoms of the protein using MDAnalysis.