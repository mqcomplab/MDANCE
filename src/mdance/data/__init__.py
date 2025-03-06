from __future__ import absolute_import

import importlib.resources


top = importlib.resources.files(__name__) / 'aligned_tau.pdb'
traj = importlib.resources.files(__name__) / 'aligned_1000_tau.dcd'
sim_traj_numpy = importlib.resources.files(__name__) / 'backbone.npy'
labels_6 = importlib.resources.files(__name__) / 'labels_6.csv'

cc_sim = importlib.resources.files(__name__) / 'cc_sim.npy'
trimmed_sim = importlib.resources.files(__name__) / 'trimmed_sim.npy'

blob_disk = importlib.resources.files(__name__) / 'blob_disk.csv'
diamonds = importlib.resources.files(__name__) / 'diamond9.csv'
ellipses = importlib.resources.files(__name__) / 'ellipses.csv'

labels_60 = importlib.resources.files(__name__) / 'labels_60.csv'