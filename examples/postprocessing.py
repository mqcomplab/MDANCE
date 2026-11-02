"""
Clustering Results Postprocessing
==================================

MDANCE provides a set of tools to postprocess the clustering results. 
This snippet demonstrates how to write out trajectories for each cluster.

The pwd of this script is ``$PATH/MDANCE/examples``.
"""

###############################################################################
# Imports
#   - `numpy <https://numpy.org/>`_ for manipulating arrays. 
#   - `MDAnalysis <https://www.mdanalysis.org/>`_ for reading and writing trajectory files.

import MDAnalysis as mda
import numpy as np

from mdance import data

###############################################################################
# Read the original trajectory file with MDAnalysis.
#   - ``input_top`` is the path to the topology file. Check `here <https://userguide.mdanalysis.org/1.0.0/formats/index.html>`_ for all accepted formats.
#   - ``input_traj`` is the path to the trajectory file. Check `here <https://userguide.mdanalysis.org/1.0.0/formats/index.html>`_ for all accepted formats.

input_top = data.top
input_traj = data.traj

u = mda.Universe(input_top, input_traj)
print(f'Number of atoms in the trajectory: {u.atoms.n_atoms}')

###############################################################################
# Extract frames for each cluster using the cluster assignments from the previous step.
#   - ``cluster_assignment`` is the path to the cluster assignment.
#   - It will take this list of frame and convert to a trajectory for each unique cluster. 
#   - Here we use the sample best frames included with MDANCE. This also works with a labels file, e.g. ``data.labels_6``.

cluster_assignment = data.best_frames_6
###############################################################################
# Define the frames to extract 
#   - ``x`` is the frame number.
#   - ``y`` is the cluster number.
#   - Output will be written to a DCD file for each cluster. Check `here <https://userguide.mdanalysis.org/1.0.0/formats/index.html>`_ for all accepted formats.

x, y = np.loadtxt(cluster_assignment, delimiter=',', dtype=int, unpack=True)

# get x value in a list for every unique y value
frames = [x[y == i] for i in np.unique(y)]

for i, frame in enumerate(frames):
    # write trajectory with only the selected frames in frames[i]
    with mda.Writer(f'best_frames_{i}.dcd', u.atoms.n_atoms) as W:
        for ts in u.trajectory[frame]:
            W.write(u.atoms)