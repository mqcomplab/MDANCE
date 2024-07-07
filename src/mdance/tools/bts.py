from shapeGMMTorch import torch_align
import MDAnalysis as mda
import numpy as np
import os
import random
import subprocess
import torch
import warnings
from mdance.tools.esim_modules import gen_sim_dict
from mdance.tools.isim import gen_sim_dict
from mdance.inputs.preprocess import gen_traj_numpy

def mean_sq_dev(matrix, N_atoms):
    """*O(N)* Mean square deviation (MSD) calculation for n-ary objects.
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        Input data matrix.
    N_atoms : int
        Number of atoms in the system.
    
    Returns
    -------
    float
        normalized MSD value.
    """
    N = len(matrix)
    sq_data = matrix ** 2
    c_sum = np.sum(matrix, axis=0)
    sq_sum = np.sum(sq_data, axis=0)
    msd = np.sum(2 * (N * sq_sum - c_sum ** 2)) / (N ** 2)
    norm_msd = msd / N_atoms
    return norm_msd

def msd_condensed(c_sum, sq_sum, N, N_atoms):
    """Condensed version of 'mean_sq_dev'.

    Parameters
    ----------
    c_sum : array-like of shape (n_features,)
        Column sum of the data. 
    sq_sum : array-like of shape (n_features,)
        Column sum of the squared data.
    N : int
        Number of data points.
    N_atoms : int
        Number of atoms in the system.
    
    Returns
    -------
    float
        normalized MSD value.
    """
    msd = np.sum(2 * (N * sq_sum - c_sum ** 2)) / (N ** 2)
    norm_msd = msd / N_atoms
    return norm_msd

def extended_comparison(matrix, data_type='full', metric='MSD', N=None, N_atoms=1, 
                        **kwargs):
    """*O(N)* Extended comparison function for n-ary objects. 
    
    Parameters
    ----------
    matrix : {array-like of shape (n_samples, n_features), tuple/list of length 1 or 2}
        Input data matrix.
            - ``full``: use numpy.ndarray of shape (n_samples, n_features).
            - ``condensed``: use tuple/list of length 1 (c_sum) or 2 (c_sum, sq_sum).
    data_type : {'full', 'condensed'}, optional
        Type of data inputted. Defaults to 'full'. Options:
            - ``full``: Use numpy.ndarray of shape (n_samples, n_features).
            - ``condensed``: Use tuple/list of length 1 (c_sum) or 2 (c_sum, sq_sum).
    metric : str, optional
        Metric to use for the extended comparison. Defaults to ``MSD``.
        Additional metrics:
            - ``AC``: Austin-Colwell, ``BUB``: Baroni-Urbani-Buser, 
            - ``CTn``: Consoni-Todschini n, ``Fai``: Faith, 
            - ``Gle``: Gleason, ``Ja``: Jaccard, 
            - ``Ja0``: Jaccard 0-variant, ``JT``: Jaccard-Tanimoto, 
            - ``RT``: Rogers-Tanimoto, ``RR``: Russel-Rao,
            - ``SM``: Sokal-Michener, ``SSn``: Sokal-Sneath n
    N : int, optional
        Number of data points. Defaults to None.
    N_atoms : int, optional
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for all non Molecular Dynamics datasets.
    c_threshold : int, optional
        Coincidence threshold. Defaults to None.
    w_factor : {'fraction', 'power_n'}, optional
        Type of weight function that will be used. Defaults to 'fraction'.
        See ``mdance.tools.esim_modules.calculate_counters`` for more information.
    
    Raises
    ------
    TypeError
        If data is not a numpy.ndarray or tuple/list of length 2.
    
    Returns
    -------
    float
        Extended comparison value.
    """
    if data_type == 'full':
        if not isinstance(matrix, np.ndarray):
            raise TypeError('data must be a numpy.ndarray')
        c_sum = np.sum(matrix, axis=0)
        if not N:
            N = len(matrix)
        if metric == 'MSD':
            sq_data = matrix ** 2
            sq_sum = np.sum(sq_data, axis=0)
        
    elif data_type == 'condensed':
        if not isinstance(matrix, (tuple, list)):
            raise TypeError('data must be a tuple or list of length 1 or 2')
        c_sum = matrix[0]
        if metric == 'MSD':
            sq_sum = matrix[1]
    if metric == 'MSD':
        return msd_condensed(c_sum, sq_sum, N, N_atoms)
    else:
            if 'c_threshold' in kwargs:
                c_threshold = kwargs['c_threshold']
            else:
                c_threshold = None
            if 'w_factor' in kwargs:
                w_factor = kwargs['w_factor']
            else:
                w_factor = 'fraction'
            esim_dict = gen_sim_dict(c_sum, n_objects=N, c_threshold=c_threshold, w_factor=w_factor)
            
            return 1 - esim_dict[metric]

def calculate_comp_sim(matrix, metric, N_atoms=1):
    """*O(N)* Complementary similarity calculation for n-ary objects.
    
    Parameters
    ----------
    matrix : array-like
        Data matrix.
    metric : {'MSD', 'JT', etc}
        Metric used for extended comparisons. 
        See ``mdance.tools.bts.extended_comparison`` for details.
    N_atoms : int, optional
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for all non Molecular Dynamics datasets.
    
    Returns
    -------
    numpy.ndarray
        Array of complementary similarities for each object.
    """
    if metric == 'MSD' and N_atoms == 1:
        warnings.warn('N_atoms is being specified as 1. Please change if N_atoms is not 1.')
    N = len(matrix)
    sq_data_total = matrix ** 2
    c_sum_total = np.sum(matrix, axis = 0)
    sq_sum_total = np.sum(sq_data_total, axis=0)
    comp_sims = []
    for i, object in enumerate(matrix):
        object_square = object ** 2
        value = extended_comparison([c_sum_total - object, sq_sum_total - object_square],
                                    data_type='condensed', metric=metric, 
                                    N=N - 1, N_atoms=N_atoms)
        comp_sims.append((i, value))
    comp_sims = np.array(comp_sims)
    return comp_sims

def calculate_medoid(matrix, metric, N_atoms=1):
    """*O(N)* medoid calculation for *n*-ary objects.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        Data matrix.
    metric : {'MSD', 'JT', etc}
        Metric used for extended comparisons. 
        See ``mdance.tools.bts.extended_comparison`` for details.
    N_atoms : int, optional
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for all non Molecular Dynamics datasets.
    
    Returns
    -------
    int
        The index of the medoid in the dataset.
    """
    if metric == 'MSD' and N_atoms == 1:
        warnings.warn('N_atoms is being specified as 1. Please change if N_atoms is not 1.')
    N = len(matrix)
    sq_data_total = matrix ** 2
    c_sum_total = np.sum(matrix, axis=0)
    sq_sum_total = np.sum(sq_data_total, axis=0)  
    index = len(matrix) + 1
    max_dissim = -1
    for i, object in enumerate(matrix):
        object_square = object ** 2
        value = extended_comparison([c_sum_total - object, sq_sum_total - object_square],
                                    data_type='condensed', metric=metric, 
                                    N=N - 1, N_atoms=N_atoms)
        if value > max_dissim:
            max_dissim = value
            index = i
        else:
            pass
    return index

def calculate_outlier(matrix, metric, N_atoms=1):
    """*O(N)* Outlier calculation for *n*-ary objects.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        Data matrix.
    metric : {'MSD', 'JT', etc}
        Metric used for extended comparisons. 
        See ``mdance.tools.bts.extended_comparison`` for details.
    N_atoms : int, optional
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for all non Molecular Dynamics datasets.
    
    Returns
    -------
    int
        The index of the outlier in the dataset.
    """
    if metric == 'MSD' and N_atoms == 1:
        warnings.warn('N_atoms is being specified as 1. Please change if N_atoms is not 1.')
    N = len(matrix)
    sq_data_total = matrix ** 2
    c_sum_total = np.sum(matrix, axis=0)
    sq_sum_total = np.sum(sq_data_total, axis=0)  
    index = len(matrix) + 1
    min_dissim = np.Inf
    for i, object in enumerate(matrix):
        object_square = object ** 2
        value = extended_comparison([c_sum_total - object, sq_sum_total - object_square],
                                    data_type='condensed', metric=metric, 
                                    N=N - 1, N_atoms=N_atoms)
        if value < min_dissim:
            min_dissim = value
            index = i
        else:
            pass
    return index

def trim_outliers(matrix, n_trimmed, metric, N_atoms, criterion='comp_sim'):
    """Trims a desired percentage of outliers (most dissimilar) from the dataset 
    by calculating largest complement similarity. *O(N)* time complexity.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        Data matrix.
    n_trimmed : float or int
        The desired fraction of outliers to be removed or the number of outliers to be removed.
        float : Fraction of outliers to be removed.
        int : Number of outliers to be removed.
    metric : {'MSD', 'JT', etc}
        Metric used for extended comparisons.
        See ``mdance.tools.bts.extended_comparison`` for details.
    N_atoms : int
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for all non Molecular Dynamics datasets.
    criterion : {'comp_sim', 'sim_to_medoid'}, optional
        Criterion to use for data trimming. Defaults to 'comp_sim'.
        ``comp_sim`` removes the most dissimilar objects based on the complement similarity.
        ``sim_to_medoid`` removes the most dissimilar objects based on the similarity to the medoid.
        
    Returns
    -------
    numpy.ndarray
        A ndarray with desired fraction of outliers removed.
    
    Notes
    -----
    ``criterion='comp_sim'``: the lowest indices are removed 
        because they are the most outlier.
    ``criterion='sim_to_medoid'``: the highest indices are removed 
        because they are farthest from the medoid.
    """
    N = len(matrix)
    if isinstance(n_trimmed, int):
        cutoff = n_trimmed
    elif 0 < n_trimmed < 1:
        cutoff = int(np.floor(N * float(n_trimmed)))
    if criterion == 'comp_sim':
        c_sum = np.sum(matrix, axis = 0)
        sq_sum_total = np.sum(matrix ** 2, axis=0)
        comp_sims = []
        for i, row in enumerate(matrix):
            c = c_sum - row
            sq = sq_sum_total - row ** 2
            value = extended_comparison([c, sq], data_type='condensed', metric=metric, 
                                        N=N - 1, N_atoms=N_atoms)
            comp_sims.append((i, value))
        comp_sims = np.array(comp_sims)
        lowest_indices = np.argsort(comp_sims[:, 1])[:cutoff]
        matrix = np.delete(matrix, lowest_indices, axis=0)
    elif criterion == 'sim_to_medoid':
        medoid_index = calculate_medoid(matrix, metric, N_atoms=N_atoms)
        medoid = matrix[medoid_index]
        np.delete(matrix, medoid_index, axis=0)
        values = []
        for i, frame in enumerate(matrix):
            value = extended_comparison(np.array([frame, medoid]), data_type='full', 
                                        metric=metric, N_atoms=N_atoms)
            values.append((i, value))
        values = np.array(values)
        highest_indices = np.argsort(values[:, 1])[-cutoff:]
        matrix = np.delete(matrix, highest_indices, axis=0)
    return matrix

def diversity_selection(matrix, percentage: int, metric, start='medoid', N_atoms=1):
    """Selects a diverse subset of the data using the complementary similarity. 
    *O(N)* time complexity.
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        Data matrix.
    percentage : int
        Percentage of the data to select.
    metric : {'MSD', 'JT', etc}
        Metric used for extended comparisons.
        See ``mdance.tools.bts.extended_comparison`` for details.
    start : {'medoid', 'outlier', 'random', list}, optional
        Seed of diversity selection. Defaults to 'medoid'.
    N_atoms : int, optional
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for all non Molecular Dynamics datasets.
    
    Raises
    ------
    ValueError
        If start is not ``medoid``, ``outlier``, ``random``, or a list.
    ValueError
        If percentage is too high.
    
    Returns
    -------
    list
        List of indices of the selected data.
    """
    n_total = len(matrix)
    total_indices = np.array(range(n_total))
    if start =='medoid':
        seed = calculate_medoid(matrix, metric=metric, N_atoms=N_atoms)
        selected_n = [seed]
    elif start == 'outlier':
        seed = calculate_outlier(matrix, metric=metric, N_atoms=N_atoms)
        selected_n = [seed]
    elif start == 'random':
        seed = random.randint(0, n_total - 1)
        selected_n = [seed]
    elif isinstance(start, list):
        selected_n = start
    else:
        raise ValueError('Select a correct starting point: medoid, outlier, random or outlier')

    n = len(selected_n)
    n_max = int(np.floor(n_total * percentage / 100))
    if n_max > n_total:
        raise ValueError('Percentage is too high')
    selection = [matrix[i] for i in selected_n] 
    selection = np.array(selection)
    selected_condensed = np.sum(selection, axis=0)
    if metric == 'MSD':
        sq_selection = selection ** 2
        sq_selected_condensed = np.sum(sq_selection, axis=0)
    
    while len(selected_n) < n_max:
        select_from_n = np.delete(total_indices, selected_n)
        if metric == 'MSD':
            new_index_n = get_new_index_n(matrix, metric=metric, selected_condensed=selected_condensed,
                                          sq_selected_condensed=sq_selected_condensed, n=n, 
                                          select_from_n=select_from_n, N_atoms=N_atoms)
            sq_selected_condensed += matrix[new_index_n] ** 2
        else:
            new_index_n = get_new_index_n(matrix, metric=metric, selected_condensed=selected_condensed, 
                                          n=n, select_from_n=select_from_n)
        selected_condensed += matrix[new_index_n]
        selected_n.append(new_index_n)
        n = len(selected_n)
    return selected_n

def get_new_index_n(matrix, metric, selected_condensed, n, select_from_n, **kwargs):
    """Function to get the new index to add to the selected indices
    
    Parameters
    ----------
    matrix : array-like
        Data matrix.
    metric : {'MSD', 'JT', etc}
        Metric used for extended comparisons.
        See ``mdance.tools.bts.extended_comparison`` for details.
    selected_condensed : array-like of shape (n_features,)
        Condensed sum of the selected fingerprints.
    n : int
        Number of selected objects.
    select_from_n : array-like of shape (n_samples,)
        Array of indices to select from.
    sq_selected_condensed : array-like of shape (n_features,), optional
        Condensed sum of the squared selected fingerprints. Defaults to None.
    N_atoms : int, optional
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for all non Molecular Dynamics datasets.
    
    Returns
    -------
    int
        index of the new fingerprint to add to the selected indices.
    """
    if 'sq_selected_condensed' in kwargs:
        sq_selected_condensed = kwargs['sq_selected_condensed']
    if 'N_atoms' in kwargs:
        N_atoms = kwargs['N_atoms']
    
    # Number of fingerprints already selected and the new one to add
    n_total = n + 1
    
    # Placeholders values
    min_value = -1
    index = len(matrix) + 1
    
    # Calculate MSD for each unselected object and select the index with the highest value.
    for i in select_from_n:
        if metric == 'MSD':
            sim_index = extended_comparison([selected_condensed + matrix[i], sq_selected_condensed + (matrix[i] ** 2)],
                                            data_type='condensed', metric=metric, N=n_total, N_atoms=N_atoms) 
        else:
            sim_index = extended_comparison([selected_condensed + matrix[i]], data_type='condensed', 
                                            metric=metric, N=n_total)
        if sim_index > min_value:
            min_value = sim_index
            index = i
        else:
            pass
    return index

def align_traj(data, N_atoms, align_method=None):
    """Aligns trajectory using uniform or kronecker alignment.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Matrix of data to be aligned
    N_atoms : int
        Number of atoms in the system.
    align_method : {'uni', 'kron'}, optional
        Alignment method, Defaults to None.
            - ``uni``or ``uniform``: Uniform alignment.
            - ``kron`` or ``kronecker``: Kronecker alignment.
    
    Raises
    ------
    ValueError
        if align_method is not ``uni``, ``kron``, or None
    
    Returns
    -------
    array-like of shape (n_samples, n_features)
        matrix of aligned data
    """
    if not align_method:
        return data
    data = data.reshape(len(data), N_atoms, 3)
    device = torch.device('cpu')
    dtype = torch.float32
    traj_tensor = torch.tensor(data, device=device, dtype=dtype)
    torch_align.torch_remove_center_of_geometry(traj_tensor)
    if align_method == 'uni' or align_method == 'uniform':
        uniform_aligned_traj_tensor, uniform_avg_tensor, uniform_var_tensor = torch_align.torch_iterative_align_uniform(
            traj_tensor, device=device, dtype=dtype, verbose=True)
        aligned_traj_numpy = uniform_aligned_traj_tensor.cpu().numpy()
    elif align_method == 'kron' or align_method == 'kronecker':
        kronecker_aligned_traj_tensor, kronecker_avg_tensor, kronecker_precision_tensor, kronecker_lpdet_tensor = torch_align.torch_iterative_align_kronecker(
            traj_tensor, device=device, dtype=dtype, verbose=True)
        aligned_traj_numpy = kronecker_aligned_traj_tensor.cpu().numpy()
    else:
        raise ValueError('Please select a correct alignment method: uni, kron, or None')
    reshaped = aligned_traj_numpy.reshape(aligned_traj_numpy.shape[0], -1)
    return reshaped

def equil_align(indices, sieve, input_top, input_traj, mdana_atomsel, cpptraj_atomsel, ref_index):
    """ Aligns the frames in the trajectory to the reference frame.
    
    Parameters
    ----------
    indices : list
        List of indices of the data points in the cluster.
    input_top : str
        Path to the input topology file.
    input_traj : str
        Path to the input trajectory file.
    mdana_atomsel : str
        Atom selection string for MDAnalysis.
    cpptraj_atomsel : str
        Atom selection string for cpptraj.
    ref_index : int
        Index of the reference frame.
    
    Returns
    -------
    aligned_traj_numpy : numpy.ndarray
        Numpy array of the aligned trajectory.
    """
    u = mda.Universe(input_top, input_traj)
    with mda.Writer(f'unaligned_traj.pdb', u.atoms.n_atoms) as W:
        for ts in u.trajectory[[i * sieve for i in indices]]:
            W.write(u.atoms)
    with open('cpptraj.in', 'w') as outfile:
        outfile.write(f'parm {input_top}\n')
        outfile.write(f'trajin unaligned_traj.pdb\n')
        outfile.write('autoimage\n')
        outfile.write(f'reference {input_traj} frame {ref_index}\n')
        outfile.write(f'rms ToAvg reference {cpptraj_atomsel}\n')
        outfile.write('trajout aligned_traj.pdb nobox\n')
        outfile.write('run\n')
    subprocess.run(['cpptraj', '-i', 'cpptraj.in'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Read aligned trajectory
    aligned_traj_numpy = gen_traj_numpy(input_top, 'aligned_traj.pdb', atomSel=mdana_atomsel)

    # Clean up
    os.remove('cpptraj.in')
    os.remove('unaligned_traj.pdb')
    os.remove('aligned_traj.pdb')
    return aligned_traj_numpy