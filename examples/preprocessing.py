"""
Preprocessing of Molecular Dynamics Data
===============================================

MDANCE provides a set of tools to preprocess molecular dynamics trajectories before clustering. 
This includes reading the trajectories, normalizing them, and aligning them. 
This snippet demonstrates how to read a trajectory and save it as a numpy array.
"""

###############################################################################
# Imports
#   - `numpy <https://numpy.org/>`_ for manipulating and saving arrays.
#   - ``gen_traj_numpy`` for using the `MDAnalysis <https://www.mdanalysis.org/>`_ library to read the trajectories and save them as numpy arrays.

import numpy as np

from mdance import data
from mdance.inputs.preprocess import gen_traj_numpy

###############################################################################
#Inputs
#   - ``input_top`` is the path to the topology file. Check `here <https://userguide.mdanalysis.org/1.0.0/formats/index.html>`_ for all accepted formats.
#   - ``input_traj`` is the path to the trajectory file. Check `here <https://userguide.mdanalysis.org/1.0.0/formats/index.html>`_ for all accepted formats.
#       - **The trajectory file should be aligned and centered beforehand if needed!**
#   - ``output_name`` is the name of the output file. The output file will be saved as ``{output_name}.npy`` for faster loading in the future.
#   - ``atomSelection`` is the atom selection used for clustering that must be compatible with the `MDAnalysis Atom Selections Language <https://userguide.mdanalysis.org/stable/selections.html>`_.
#   - ``gen_traj_numpy`` will convert the trajectory to a numpy array with the shape ``(n_frames, n_atoms * 3)`` for comparison purposes.

input_top = data.top
input_traj = data.traj
output_base_name = 'backbone'
atomSelection = 'resid 3 to 12 and name N CA C O H'

traj_numpy = gen_traj_numpy(input_top, input_traj, atomSelection)

###############################################################################
# Outputs
#   - The output is a numpy array of shape ``(n_frames, n_atoms * 3)``.

output_name = output_base_name + '.npy'
np.save(output_name, traj_numpy)