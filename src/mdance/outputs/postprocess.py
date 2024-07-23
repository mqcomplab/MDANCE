def unnormalize_data(norm_data, min, max):
    """Unnormalize data from 0 to 1 to the original range.
    
    Parameters
    ----------
    norm_data : array_like of shape (n_samples, n_features)
        The normalized data.
    min : float
        The minimum value of the original range.
    max : float
        The maximum value of the original range.
    
    Returns
    -------
    array_like of shape (n_samples, n_features)
        The unnormalized data.
    """
    unnorm_data = norm_data * (max - min) + min
    return unnorm_data


def numpy_array_to_crd_traj(matrix, num_columns=10):
    """Convert a numpy array to a AMBER CRD trajectory.

    Parameters:
    -----------
    array : array_like of shape (n_samples, n_features)
        The data to be converted.
    num_columns : int, optional
        The number of columns per line. Defaults to 10.
    
    Returns
    -------
    str
        The string representation of the trajectory.
    """
    num_atoms = len(matrix)
    traj_lines = []
    for i in range(0, num_atoms, num_columns):
        atom_line = ' ' + ' '.join([f'{x:.3f}'.rjust(7) for x in matrix[i:i+num_columns]])
        traj_lines.append(atom_line)
    return '\n'.join(traj_lines)