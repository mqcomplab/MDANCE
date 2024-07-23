import glob
from itertools import chain
import MDAnalysis as mda
import numpy as np
import re


def gen_traj_numpy(prmtopFileName, trajFileName, atomSel):
    """Reads in a trajectory and returns a 2D numpy array of the coordinates 
    of the selected atoms.
    
    Parameters
    ----------
    prmtopFileName : str
        The file path of the topology file.
    trajFileName : str
        The file path of the trajectory file.
    atomSel : str
        The atom selection string. For example, ``resid 3:12 and name N H CA C O``.
        View details in the `MDAnalysis documentation`_.

    Returns
    -------
    traj_numpy : np.ndarray
        The 2D numpy array of shape (n_frames, n_atoms*3) containing the coordinates
        of the selected atoms.
        
    Examples
    --------
    >>> traj_numpy = gen_traj_numpy('aligned_tau.pdb', 'aligned_tau.dcd', 
                                    'resid 3:12 and name N CA C')

    .. _MDAnalysis documentation:
         https://docs.mdanalysis.org/stable/documentation_pages/selections.html
    """
    coord = mda.Universe(prmtopFileName,trajFileName)
    print('Number of atoms in trajectory:', coord.atoms.n_atoms)
    print('Number of frames in trajectory:', coord.trajectory.n_frames)
    atomSel = coord.select_atoms(atomSel)
    print('Number of atoms in selection:', atomSel.n_atoms)
    # Create traj data of the atom selection
    traj_numpy = np.empty((coord.trajectory.n_frames,atomSel.n_atoms, 3), dtype=float)
    # Loop every frame and store the coordinates of the atom selection
    for ts in coord.trajectory:
        traj_numpy[ts.frame,:] = atomSel.positions
    # Flatten 3D array to 2D array
    traj_numpy = traj_numpy.reshape(traj_numpy.shape[0],-1)
    return traj_numpy


class Normalizer:
    """A class for normalizing data from cpptraj CRD/MDCRD files.

    Parameters
    ----------
    file_path : str, optional
        The file path of the input data. If provided, the data is read from
        the file. Defaults to None.
    data : array_like of shape (n_samples, n_features), optional
        The input data as a numpy array. If provided, the file_path argument
        is ignored. Defaults to None.
    custom_min : {float, None}, optional
        The minimum value to use for normalization. If not provided, 
        the minimum value of the input data is used. Defaults to None.
    custom_max : {float, None}, optional
        The maximum value to use for normalization. If not provided,
        the maximum value of the input data is used. Defaults to None.
    custom_avg : {float, None}, optional
        The average value to use for normalization. If not provided,
        the average value of the input data is used. Defaults to None.
            
    Attributes
    ----------
    file_path : str, optional
        The file path of the input data. If provided, the data is read from
        the file. Defaults to None.
    data : array_like of shape (n_samples, n_features), optional
        The input data as a numpy array. If provided, the file_path argument
        is ignored. Defaults to None.
    custom_min : float or None, optional
        The minimum value to use for normalization. If not provided, 
        the minimum value of the input data is used. Defaults to None.
    custom_max : float or None, optional
        The maximum value to use for normalization. If not provided,
        the maximum value of the input data is used. Defaults to None.
    normed_data : np.ndarray
        The normalized input data as a numpy array.
    c_total : np.ndarray
        The sum of columns of the normalized input data.
    min : float
        The minimum value of the input data.
    max : float
        The maximum value of the input data.
        
    Notes
    -----
    Used for non-Molecular Dynamics data.
    Please use ``gen_traj_numpy`` for all Molecular Dynamics data.
    """ 
    def __init__(self, file_path=None, data=None, custom_min=None, custom_max=None, custom_avg=None):
        if file_path:
            self.file_path = file_path
            self.data = np.genfromtxt(self.file_path)
        elif data is not None:
            self.data = data
        if custom_min and custom_max:
            self.min = custom_min
            self.max = custom_max
        else:
            self.min = np.min(self.data)
            self.max = np.max(self.data)

        self.v3_norm = (self.data - self.min) / (self.max - self.min)
        if custom_avg is not None:
            self.avg = custom_avg
        else:
            self.avg = np.mean(self.v3_norm, axis=0)
        self.v2_norm = 1 - np.abs(self.v3_norm - self.avg)
        self.c_total = np.sum(1 - np.abs(self.v3_norm - np.mean(self.v3_norm, axis=0)), axis=0)
    
    def get_min_max(self):
        """Returns the minimum and maximum values of the input data."""
        return self.min, self.max, self.avg
    
    def get_v2_norm(self):
        """Returns the ``v2`` normalized data."""
        return self.v2_norm
    
    def get_v3_norm(self):
        """Returns the ``v3`` normalized data."""
        return self.v3_norm
    
    def get_c_total(self):
        """Returns the ``c_total`` values."""
        return self.c_total


def read_cpptraj(break_line=None, norm_type=None, min=None, max=None, avg=None, normalize=False):
    """Read multiple AMBER CRD files to convert to numpy ndarray formatting and normalize the data.
    
    Parameters
    ----------
    break_line : int
        The number of columns per line of the input file. (have to n-1 because ignore first line)
    norm_type : str
        The type of normalization to use. Can be ``v2`` or ``v3``.
    min : float or None, optional
        The minimum value to use for normalization. If not provided,
        the minimum value of the input data is used. Defaults to None.
    max : float or None, optional
        The maximum value to use for normalization. If not provided,
        the maximum value of the input data is used. Defaults to None.
    avg : float or None, optional
        The average value to use for normalization. If not provided,
        the average value of the input data is used. Defaults to None.
    normalize : bool, optional
        Whether to normalize the input data. If True, the data is
        normalized to the range [0, 1]. Defaults to False.
    
    Returns
    -------
    np.ndarray
        The concatenated input data as a numpy array.
        
    Notes
    -----
    Not recommended due to inefficiency and 3-decimal precision loss.
    Please use ``gen_traj_numpy`` for all Molecular Dynamics data.
    """
    input_files = sorted(glob.glob("clusttraj.c*"), key=lambda x: int(re.findall("\d+", x)[0]))
    break_line = break_line
    frames_list = []
    count_frames = []
    for file in input_files:
        with open(file, 'r') as infile:
            lines = [line.rstrip() for line in infile][1:]
        sep_lines = [[line[i:i+8] for i in range(0, len(line), 8)] for line in lines]
        chunks = [sep_lines[i:i+break_line] for i in range(0, len(sep_lines), break_line)]
        str_frames = [list(chain.from_iterable(chunk)) for chunk in chunks]
        str_frames = [' '.join(frame) for frame in str_frames]
        frames = np.array([np.fromstring(frame, dtype='float32', sep=' ') for frame in str_frames])
        if normalize:
            norm = Normalizer(data=frames, custom_min=min, custom_max=max, custom_avg=avg)
            if norm_type == "v2":
                normed_frame = norm.get_v2_norm()
            elif norm_type == "v3":
                normed_frame = norm.get_v3_norm()
            np.savetxt(f"normed_{file}", normed_frame)
        else:
            frames_list.append(frames)
        count_frames.append(len(frames))
    if not normalize:
        data = np.concatenate(frames_list, axis=0)
        return data


def normalize_file(file, break_line=None, norm_type=None): 
    """Normalize a single file and output the normalized data to a new file.
    
    Parameters
    ----------
    file : str
        The file path of the input data.
    output : str
        The file path of the output data.
    break_line : int
        The number of columns per line of the input file. (have to n-1 because ignore first line)
    norm_type : str
        The type of normalization to use. Can be ``v2`` or ``v3``.
    min : float or None, optional
        The minimum value to use for normalization. If not provided,
        the minimum value of the input data is used. Defaults to None.
    max : float or None, optional
        The maximum value to use for normalization. If not provided,
        the maximum value of the input data is used. Defaults to None.
    avg : float or None, optional
        The average value to use for normalization. If not provided,
        the average value of the input data is used. Defaults to None.
    
    Returns
    -------
    tuple
        The minimum, maximum, and average values of the input data.
    """
    if file is not isinstance(file, str):
        frames = file
    if break_line:
        break_line = break_line
        with open(file, 'r') as infile:
            lines = [line.rstrip() for line in infile][1:]
        sep_lines = [[line[i:i+8] for i in range(0, len(line), 8)] for line in lines]
        chunks = [sep_lines[i:i+break_line] for i in range(0, len(sep_lines), break_line)]
        str_frames = [list(chain.from_iterable(chunk)) for chunk in chunks]
        str_frames = [' '.join(frame) for frame in str_frames]
        frames = np.array([np.fromstring(frame, dtype='float32', sep=' ') for frame in str_frames])
    if norm_type == "v2":
        norm = Normalizer(data=frames)
        normed_frame = norm.get_v2_norm()
    elif norm_type == "v3":
        norm = Normalizer(data=frames)
        normed_frame = norm.get_v3_norm()
    min, max, avg = norm.get_min_max()
    return normed_frame, min, max, avg