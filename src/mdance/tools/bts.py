import os
import random
import subprocess

import MDAnalysis as mda
import numpy as np
from shapeGMMTorch import torch_align
import torch

import mdance.tools.esim as esim
from mdance.inputs.preprocess import gen_traj_numpy


def mean_sq_dev(matrix, N_atoms):
    """*O(N)* Mean square deviation (MSD) calculation for *n*-ary objects.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    N_atoms : int
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1``
        for non-MD systems.
    
    Returns
    -------
    float
        normalized MSD value.
    
    See Also
    --------
    msd_condensed : Condensed version of MSD calculation for *n*-ary objects.
    extended_comparisons : *n*-ary similarity calculation for all indices/metrics.

    Examples
    --------
    >>> from mdance.tools import bts
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8]])
    >>> bts.mean_sq_dev(X, N_atoms=1)
    32.8
    """
    N = len(matrix)
    if N == 1:
        return 0
    sq_data = matrix ** 2
    c_sum = np.sum(matrix, axis=0)
    sq_sum = np.sum(sq_data, axis=0)
    msd = np.sum(2 * (N * sq_sum - c_sum ** 2)) / (N ** 2)
    norm_msd = msd / N_atoms
    return norm_msd


def msd_condensed(c_sum, sq_sum, N, N_atoms):
    """Condensed version of Mean square deviation (MSD) calculation 
    for *n*-ary objects. 

    Parameters
    ----------
    c_sum : array-like of shape (n_features,)
        A feature array of the column-wsie sum of the data. 
    sq_sum : array-like of shape (n_features,)
        A feature array of the column-wise sum of the squared data. 
    N : int
        Number of data points.
    N_atoms : int
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1`` 
        for non-MD systems.
    
    Returns
    -------
    float
        normalized MSD value.

    See Also
    --------
    mean_sq_dev : Full version of MSD calculation for *n*-ary objects.
    extended_comparisons : *n*-ary similarity calculation for all indices/metrics.

    Examples
    --------
    >>> from mdance.tools import bts
    >>> import numpy as np
    >>> c_sum = np.array([21, 22])
    >>> sq_sum = np.array([137, 130])
    >>> bts.msd_condensed(c_sum, sq_sum, N=5, N_atoms=1)
    32.8
    """
    if N == 1:
        return 0
    msd = np.sum(2 * (N * sq_sum - c_sum ** 2)) / (N ** 2)
    norm_msd = msd / N_atoms
    return norm_msd


def extended_comparison(matrix, data_type='full', metric='MSD', N=None, 
                        N_atoms=1, **kwargs):
    """*O(N)* Extended comparison function for *n*-ary objects. 
    
    Valid values for metric are:

    ``MSD``: Mean Square Deviation.
    
    Extended or Instant Similarity Metrics : 
    
    | ``AC``: Austin-Colwell, ``BUB``: Baroni-Urbani-Buser, 
    | ``CTn``: Consoni-Todschini n, ``Fai``: Faith, 
    | ``Gle``: Gleason, ``Ja``: Jaccard, 
    | ``Ja0``: Jaccard 0-variant, ``JT``: Jaccard-Tanimoto, 
    | ``RT``: Rogers-Tanimoto, ``RR``: Russel-Rao,
    | ``SM``: Sokal-Michener, ``SSn``: Sokal-Sneath n.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features) or tuple/list of \
        length 1 or 2}
        A feature array of shape (n_samples, n_features) if ``data_type='full'``. 
        Otherwise, tuple or list of length 1 (c_sum) or 2 (c_sum, sq_sum) 
        if ``data_type='condensed'``.
    data_type : {'full', 'condensed'}, default='full'
        Type of data inputted.
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N : int, optional, default=None
        Number of data points. 
    N_atoms : int, default=1
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1``
        for non-MD systems.
    c_threshold : int, default=None
        Coincidence threshold for calculating extended similarity. It must 
        be an options allowed by :func:`mdance.tools.esim.calculate_counters`.
    w_factor : {'fraction', 'power_n'}, default='fraction'
        The type of weight function for calculating extended similarity. It must 
        be an options allowed by :func:`mdance.tools.esim.calculate_counters`.
    
    Raises
    ------
    TypeError
        If data is not a numpy.ndarray or tuple/list of length 2.
    
    Returns
    -------
    float
        Extended comparison value.

    Examples
    --------
    >>> from mdance.tools import bts
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8]])
    >>> bts.extended_comparison(X, data_type='full', metric='MSD', N_atoms=1)
    32.8
    """
    if not N:
        N = len(matrix)
    if data_type == 'full':
        if not isinstance(matrix, np.ndarray):
            raise TypeError('data must be a numpy.ndarray')
        if matrix.ndim != 2:
            raise ValueError('Input must be numpy ndarray of shape (n_samples, n_features)')
        c_sum = np.sum(matrix, axis=0)
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
            esim_dict = esim.gen_sim_dict(c_sum, n_objects=N, 
                                          c_threshold=c_threshold, 
                                          w_factor=w_factor)
            return 1 - esim_dict[metric]


def calculate_comp_sim(matrix, metric, N_atoms=1):
    """*O(N)* Complementary similarity calculation for *n*-ary objects.
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    metric : str
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int, default=1
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1`` 
        for non-MD systems.

    Returns
    -------
    numpy.ndarray
        Array of complementary similarities for each object.

    Examples
    --------
    >>> from mdance.tools import bts
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8]])
    >>> bts.calculate_comp_sim(X, metric='MSD', N_atoms=1)
    array([31, 34.375, 36.75, 27.75, 23.875])
    """
    N = len(matrix)
    sq_data = matrix ** 2
    c_sum = np.sum(matrix, axis=0)
    sq_sum = np.sum(sq_data, axis=0)
    comp_csum = c_sum - matrix
    comp_sqsum = sq_sum - sq_data
    
    if metric == 'MSD':
        comp_msd = np.sum(2 * ((N-1) * comp_sqsum - comp_csum ** 2), axis=1) / (N-1)**2
        comp_sims = comp_msd / N_atoms
    
    else:
        comp_sims = []
        for object in matrix:
            object_square = object ** 2
            value = extended_comparison([c_sum - object, sq_sum - object_square],
                                        data_type='condensed', metric=metric, 
                                        N=N-1, N_atoms=N_atoms)
            comp_sims.append(value)
        comp_sims = np.array(comp_sims)

    return comp_sims


def calculate_medoid(matrix, metric, N_atoms=1):
    """*O(N)* medoid calculation for *n*-ary objects.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    metric : str
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int, default=1
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1`` 
        for non-MD systems.
    
    Returns
    -------
    int
        The index of the medoid in the dataset.
    
    Examples
    --------
    >>> from mdance.tools import bts
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8]])
    >>> bts.calculate_medoid(X, metric='MSD', N_atoms=1)
    2
    """
    return np.argmax(calculate_comp_sim(matrix, metric, N_atoms))


def calculate_outlier(matrix, metric, N_atoms=1):
    """*O(N)* Outlier calculation for *n*-ary objects.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int, default=1
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1`` 
        for non-MD systems.
    
    Returns
    -------
    int
        The index of the outlier in the dataset.

    Examples
    --------
    >>> from mdance.tools import bts
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8]])
    >>> bts.calculate_outlier(X, metric='MSD', N_atoms=1)
    4
    """
    return np.argmin(calculate_comp_sim(matrix, metric, N_atoms))


def trim_outliers(matrix, n_trimmed, metric, N_atoms, criterion='comp_sim'):
    """*O(N)* method of trimming a desired percentage of outliers 
    (most dissimilar) from a data matrix through complementary similarity.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    n_trimmed : float or int
        The desired fraction of outliers to be removed or the number of outliers to be removed.
        float : Fraction of outliers to be removed.
        int : Number of outliers to be removed.
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int, default=1
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1`` 
        for non-MD systems.
    criterion : {'comp_sim', 'sim_to_medoid'}, default='comp_sim'
        Criterion to use for data trimming. ``comp_sim`` criterion removes the most 
        dissimilar objects based on the complement similarity. ``sim_to_medoid`` 
        criterion removes the most dissimilar objects based on the similarity to 
        the medoid.
    
    Returns
    -------
    numpy.ndarray
        A ndarray with desired fraction of outliers removed.
    
    Examples
    --------
    >>> from mdance.tools import bts
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    >>> output = bts.trim_outliers(X, n_trimmed=0.6, metric='MSD', N_atoms=1)
    >>> output
    array([[2, 3], [8, 7], [8, 8]])
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
            value = extended_comparison([c, sq], data_type='condensed', 
                                        metric=metric, N=N - 1, 
                                        N_atoms=N_atoms)
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


def diversity_selection(matrix, percentage: int, metric, N_atoms=1, 
                        start='medoid'):
    """*O(N)* method of selecting the most diverse subset of a data 
    matrix using the complementary similarity. 
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    percentage : int
        Percentage of the data to select.
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int, default=1
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for non-Molecular Dynamics datasets.
    start : {'medoid', 'outlier', 'random', list}, default='medoid'
        The initial seed for initiating diversity selection. Either 
        from one of the options or a list of indices are valid inputs.

    Raises
    ------
    ValueError
        If ``start`` is not ``medoid``, ``outlier``, ``random``, or a list.
    ValueError
        If ``percentage`` is too high.
    
    Returns
    -------
    list
        List of indices of the diversity selected data.

    Examples
    --------
    >>> from mdance.tools import bts
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8]])
    >>> bts.diversity_selection(X, percentage=10, metric='MSD', N_atoms=1)
    [2]
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
        raise ValueError('Select a correct starting point: medoid, outlier, \
                         random or outlier')

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
            new_index_n = get_new_index_n(matrix, metric=metric, 
                                          selected_condensed=selected_condensed,
                                          sq_selected_condensed=sq_selected_condensed, 
                                          n=n, select_from_n=select_from_n, 
                                          N_atoms=N_atoms)
            sq_selected_condensed += matrix[new_index_n] ** 2
        else:
            new_index_n = get_new_index_n(matrix, metric=metric, 
                                          selected_condensed=selected_condensed, 
                                          n=n, select_from_n=select_from_n)
        selected_condensed += matrix[new_index_n]
        selected_n.append(new_index_n)
        n = len(selected_n)
    return selected_n


def get_new_index_n(matrix, metric, selected_condensed, n, select_from_n, **kwargs):
    """Extract the new index to add to the list of selected indices.
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    selected_condensed : array-like of shape (n_features,)
        Condensed sum of the selected fingerprints.
    n : int
        Number of selected objects.
    select_from_n : array-like of shape (n_samples,)
        Array of indices to select from. 
    sq_selected_condensed : array-like of shape (n_features,), optional
        Condensed sum of the squared selected fingerprints. (**kwargs)
    N_atoms : int, optional
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for non-Molecular Dynamics datasets. (**kwargs)
    
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
                                            data_type='condensed', metric=metric, 
                                            N=n_total, N_atoms=N_atoms) 
        else:
            sim_index = extended_comparison([selected_condensed + matrix[i]], 
                                            data_type='condensed', 
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
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    N_atoms : int
        Number of atoms in the system.
    align_method : {'uni', 'kron'}, default=None
        Alignment method. ``uni`` or ``uniform``: Uniform alignment.
        ``kron`` or ``kronecker``: Kronecker alignment.
    
    Raises
    ------
    ValueError
        if align_method is not ``uni``, ``kron``, or ``None``.
    
    Returns
    -------
    numpy.ndarray
        matrix of aligned data.
    
    References
    ----------
    Klem, H., Hocky, G. M., and McCullagh M., `"Size-and-Shape Space Gaussian 
    Mixture Models for Structural Clustering of Molecular Dynamics Trajectories"`_.
    *Journal of Chemical Theory and Computation* **2022** 18 (5), 3218-3230

    .. _"Size-and-Shape Space Gaussian Mixture Models for Structural Clustering of Molecular Dynamics Trajectories":
        https://pubs.acs.org/doi/abs/10.1021/acs.jctc.1c01290
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
    """Aligns the frames in the trajectory to the reference frame.
    
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
    subprocess.run(['cpptraj', '-i', 'cpptraj.in'], stdout=subprocess.DEVNULL, 
                   stderr=subprocess.DEVNULL)
    
    # Read aligned trajectory
    aligned_traj_numpy = gen_traj_numpy(input_top, 'aligned_traj.pdb', 
                                        atomSel=mdana_atomsel)

    # Clean up
    os.remove('cpptraj.in')
    os.remove('unaligned_traj.pdb')
    os.remove('aligned_traj.pdb')
    return aligned_traj_numpy


def rep_sample(data, metric='MSD', N_atoms=1, n_bins=10, n_samples=100, 
               hard_cap=True):
    """Representative sampling according to comp_sim values.
    
    Divides the range of comp_sim values in nbins and then
    uniformly selects n_samples molecules, consecutively
    taking one from each bin
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data to be sampled.
    metric : str, default='MSD'
        The metric to be used for the comparison.
    N_atoms : int, default=1
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1`` 
        for non-MD systems.
    n_bins : int, default=10
        Number of bins to divide the comp_sim values.
    n_samples : int, default=100
        Number of samples to be selected.
    hard_cap : bool, default=True
        If True, the number of samples will be exactly n_samples.
        If False, the number of samples may not be exactly n_samples.
        
    Returns
    -------
    sampled_mols : list
        List of indices of the sampled objects in the original data
    """
    n = len(data)
    if n_samples < 1:
        n_samples = int(n * n_samples)
    cs = calculate_comp_sim(data, metric=metric, N_atoms=N_atoms)
    tups = []
    for i, comp in enumerate(cs):
        tups.append((i, comp))
    comp_sims = np.sort(cs)
    mi = np.min(comp_sims)
    ma = np.max(comp_sims)
    D = ma - mi
    step = D / n_bins
    bins = []
    indices = np.array(list(range(n)))
    for i in range(n_bins - 1):
        low = mi + i * step
        up = mi + (i + 1) * step
        bins.append(indices[(comp_sims >= low) * (comp_sims < up)])
    bins.append(indices[(comp_sims >= up) * (comp_sims <= ma)])
    order_sampled = []
    i = 0
    while len(order_sampled) < n_samples:
        for b in bins:
            if len(b) > i:
                order_sampled.append(b[i])
                if hard_cap:
                    if len(order_sampled) >= n_samples:
                        break
            else:
                pass
        i += 1
    tups.sort(key = lambda tups : tups[1])
    sampled_mols = []
    for i in order_sampled:
        sampled_mols.append(tups[i][0])
    return sampled_mols


def refine_dis_matrix(matrix):
    """Refine a distance matrix by setting the diagonal to zero and
    symmetrizing the matrix.
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    
    Returns
    -------
    numpy.ndarray
        A refined 2D matrix.
    """
    distances = np.array(matrix)
    if distances.ndim != 2:
        raise ValueError('Matrix must be 2D')
    if distances.shape[0] != distances.shape[1]:
        raise ValueError('Matrix must be square')
    distances += distances.T
    distances *= 0.5
    distances -= np.min(distances)
    np.fill_diagonal(distances, 0)
    return distances