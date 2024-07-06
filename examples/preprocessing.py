"""
Preprocessing of Molecular Dynamics Trajectories

Imports
^^^^^^^

* `numpy <https://numpy.org/>`_ for manipulating and saving arrays.
* ``gen_traj_numpy`` for using the `MDAnalysis <https://www.mdanalysis.org/>`_ library to read the trajectories and save them as numpy arrays.
* ``normalize_files`` for normalizing the trajectories.
* ``align_traj`` for aligning the trajectories.
"""
import numpy as np
from mdance.inputs.preprocess import gen_traj_numpy

"""
Inputs
^^^^^^

* ``input_top`` is the path to the topology file. 
Check `here <https://userguide.mdanalysis.org/1.0.0/formats/index.html>`_ for all accepted formats.
* ``input_traj`` is the path to the trajectory file. 
Check `here <https://userguide.mdanalysis.org/1.0.0/formats/index.html>`_ for all accepted formats.

  * **Note**: The trajectory file should be already aligned and centered beforehand if needed!

* ``output_name`` is the name of the output file. The output file will be saved as ``{output_name}.npy`` for faster loading in the future.
* ``atomSelection`` is the atom selection used for clustering that must be compatible with the `MDAnalysis Atom Selections Language <https://userguide.mdanalysis.org/stable/selections.html>`_.

``gen_traj_numpy`` will convert the trajectory to a numpy array with the shape ``(n_frames, n_atoms * 3)`` for comparison purposes.
``normalize_file`` will normalize trajectory between ``[0, 1]`` to be compatible with extended similarity indices.
"""
input_top = '../../data/md/aligned_tau.pdb'
input_traj = '../../data/md/aligned_1000_tau.dcd'
output_base_name = '../../data/md/backbone'
atomSelection = 'resid 3 to 12 and name N CA C O H'

traj_numpy = gen_traj_numpy(input_top, input_traj, atomSelection)

"""
#### Outputs
The output is a numpy array of shape (n_frames, n_atoms * 3).
"""
output_name = output_base_name + '.npy'
np.save(output_name, traj_numpy)