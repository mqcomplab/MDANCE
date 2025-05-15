from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import squareform

from mdance.tools.bts import diversity_selection
from mdance.tools.bts import extended_comparison
from mdance.tools.bts import rep_sample
from mdance.tools.bts import refine_dis_matrix


class Shine:
    """
    SHINE (Sampling Hierarchical Intrinsic *N*-ary Ensembles) is a class that
    performs hierarchical clustering on a set of pathways. It uses the
    *n*-ary similarity framework to sample/calculate the pairwise distances between
    the trajectories. The class also provides a method to generate a dendrogram
    plot of the clustering results.
    
    Parameters
    ----------
    trajs: list
        List of tuples containing (idx, traj) where idx is the trajectory index 
        and traj is the trajectory data
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int, default=1) 
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1`` 
        for non-MD systems.   
    link : str, default='ward'
        The linkage algorithm to use. See the `Linkage Methods`_ for full descriptions.
    t : scalar
        For criteria 'inconsistent', 'distance' or 'monocrit',
         this is the threshold to apply when forming flat clusters.
        For 'maxclust' or 'maxclust_monocrit' criteria,
         this would be max number of clusters requested. See `fcluster`_
        for the full description.
    criterion : str, optional
        The criterion to use in forming flat clusters. This can
        be any of the following values: 'inconsistent', 'distance',
        'maxclust', 'monocrit', 'maxclust_monocrit'. See `fcluster`_
        for the full description.
    merge_scheme : str, default='intra'
        The scheme to merge the distances between the trajectories.
        Possible values are ``'intra'``, ``'inter'``, ``'semi_sum'``,
        ``'max'``, ``'min'``, ``'haus'``.
    sampling : str, default='diversity'
        The sampling scheme to use. Possible values are ``'diversity'``,
        ``'quota'``, ``None``.
    frame_cutoff : int, default=50
        Minimum number of frames to perform sampling.
    frac : float, default=0.2
        Fraction of the data to be sampled from each trajectory.
    
    .. _Linkage Methods: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    
    .. _fcluster: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
    """
    def __init__(self, trajs, metric, N_atoms, t, criterion, link='ward',
                 merge_scheme='intra', sampling='diversity', **kwargs):
        self.trajs = trajs
        self.metric = metric
        self.N_atoms = N_atoms
        self.t = t
        self.criterion = criterion
        self.link = link
        self.merge_scheme = merge_scheme
        self.sampling = sampling
        self.frame_cutoff = kwargs.get('frame_cutoff', 50)
        self.frac = kwargs.get('frac', 0.2)
        self.pathways = {}
        self._check_merge_scheme()
        self._check_sampling_scheme()
    
    def _check_merge_scheme(self):
        """
        Check if the merge scheme is valid.
        
        Raises
        ------
        ValueError
            If the merge scheme is not valid.
        """
        if self.merge_scheme not in ['intra', 'inter', 'semi_sum', 'max', 'min', 'haus']:
            raise ValueError(f"Invalid merge scheme: {self.merge_scheme}")
    
    def _check_sampling_scheme(self):
        """
        Check if the sampling scheme is valid.
        
        Raises
        ------
        ValueError
            If the sampling scheme is not valid.
        """
        if self.sampling not in ['diversity', 'quota', None]:
            raise ValueError(f"Invalid sampling scheme: {self.sampling}")
    
    def _check_frac(self):
        """
        Check if the fraction is valid.
        
        Raises
        ------
        ValueError
            If the fraction is not valid.
        """
        if not 0 < self.frac <= 1:
            raise ValueError(f"Invalid fraction: {self.frac}")
        
    def process_trajs(self):
        """
        Generates the trajectory dictionary and applies the sampling scheme.
        
        Returns
        -------
        pathways : dict
            Dictionary containing the sampled trajectories
        """
        for traj_idx, traj in self.trajs:
            traj_idx = int(traj_idx)
            if self.sampling:
                if self.sampling == 'diversity':
                    if self.frame_cutoff <= len(traj):
                        div_idxs = diversity_selection(traj, self.frac * 100, 
                                                       self.metric, self.N_atoms,
                                                       'comp_sim', 'medoid')
                        self.pathways[traj_idx] = traj[div_idxs]
                    else:
                        self.pathways[traj_idx] = traj
                elif self.sampling == 'quota':
                    if self.frame_cutoff <= len(traj):
                        n_frames = int(self.frac * len(traj))
                        rep_idx = rep_sample(traj, self.metric, self.N_atoms, 
                                             n_bins=n_frames, n_samples=n_frames)
                        self.pathways[traj_idx] = traj[rep_idx]
                    else:
                        self.pathways[traj_idx] = traj
            else:
                self.pathways[traj_idx] = traj
        return self.pathways

    def gen_msdmatrix(self):
        """
        Generates the MSD matrix for the trajectories using the merge scheme.
        
        Returns
        -------
        distances : array-like of shape (n_samples, n_samples)
            The MSD pairwise distances between the trajectories 
            using the merge scheme.
        """
        ntrajs = len(self.pathways)
        distances = []
        for i in range(ntrajs):
            distances.append([])
            for j in range(ntrajs):
                if i == j:
                    distances[-1].append(0)
                else:
                    combined = np.concatenate((self.pathways[i], self.pathways[j]), axis = 0)
                    if self.merge_scheme == 'intra':
                        d = extended_comparison(combined, metric=self.metric, N_atoms=self.N_atoms)
                    elif self.merge_scheme == 'inter':
                        d = (extended_comparison(combined, metric=self.metric, N_atoms=self.N_atoms) * len(combined)**2 - (extended_comparison(self.pathways[i], metric=self.metric, N_atoms=self.N_atoms)*\
                            len(self.pathways[i])**2 + extended_comparison(self.pathways[j], metric=self.metric, N_atoms=self.N_atoms)*len(self.pathways[j])**2))/(len(self.pathways[i]) * len(self.pathways[j]))
                    elif self.merge_scheme == 'semi_sum':
                        d = extended_comparison(combined, metric=self.metric, N_atoms=self.N_atoms) - 0.5 * \
                            (extended_comparison(self.pathways[i], metric=self.metric, N_atoms=self.N_atoms) + \
                                extended_comparison(self.pathways[j], metric=self.metric, N_atoms=self.N_atoms))
                    elif self.merge_scheme == 'max':
                        d = extended_comparison(combined, metric=self.metric, N_atoms=self.N_atoms) - max(
                            extended_comparison(self.pathways[i], metric=self.metric, N_atoms=self.N_atoms), 
                            extended_comparison(self.pathways[j], metric=self.metric, N_atoms=self.N_atoms))
                    elif self.merge_scheme == 'min':
                        d = extended_comparison(combined, metric=self.metric, N_atoms=self.N_atoms) - min(
                            extended_comparison(self.pathways[i], metric=self.metric, N_atoms=self.N_atoms), 
                            extended_comparison(self.pathways[j], metric=self.metric, N_atoms=self.N_atoms))
                    elif self.merge_scheme == 'haus':
                        d = max(directed_hausdorff(self.pathways[j], self.pathways[i])[0], 
                                directed_hausdorff(self.pathways[i], self.pathways[j])[0])
                    distances[-1].append(d)
        
        distances = refine_dis_matrix(distances)
        return distances
    
    def run(self):
        """
        Performs the hierarchical agglomerative clustering on the trajectories.
        
        Returns
        -------
        link_matrix : ndarray
            The hierarchical clustering encoded as a linkage matrix.
        clusters : ndarray
            An array of length n. T[i] is the flat cluster number to 
            which original observation i belongs.
        """
        self.pathways = self.process_trajs()
        distances = self.gen_msdmatrix()
        linmat = squareform(distances, force='no', checks=True)
        self.link_matrix = linkage(linmat, method=self.link)
        self.clusters = fcluster(self.link_matrix, t=self.t, 
                                 criterion=self.criterion)
        return self.link_matrix, self.clusters
    
    def group_consecutive_indices(self, indices):
        """
        Group consecutive indices into ranges for ``labels`` method
        
        Parameters
        ----------
        indices : list
            List of indices to group
        
        Returns
        -------
        str
            Grouped indices as a string        
        """
        indices = sorted(indices)
        result = []
        start = indices[0]
        end = indices[0]

        for i in range(1, len(indices)):
            if indices[i] == end + 1:
                end = indices[i]
            else:
                if start == end:
                    result.append(f"{start}")
                else:
                    result.append(f"{start}-{end}")
                start = indices[i]
                end = indices[i]
        if start == end:
            result.append(f"{start}")
        else:
            result.append(f"{start}-{end}")
        return ", ".join(result)

    def labels(self):
        """
        Generate custom labels for the dendrogram plot
        
        Returns
        -------
        custom_labels : list
            List of custom labels for the clusters
        """
        label_indices = []
        for cluster in np.unique(self.clusters):
            label_indices.append(np.where(self.clusters == cluster)[0])
        custom_labels = [f"{self.group_consecutive_indices(cluster)}" for cluster in label_indices]
        return custom_labels

    def plot(self):
        """
        Generates the dendrogram plot of the clustering results.
        
        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The dendrogram plot
        """
        self.custom_labels = self.labels()
        ax = dendrogram(self.link_matrix, no_labels=True)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
        plt.rcParams['font.size'] = 12
        for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(1.25)
        legend_handles = [Line2D([0], [0], color=colors[(i + 1) % len(colors)], lw=3, label=label)
                          for i, label in enumerate(self.custom_labels)]
        plt.legend(handles=legend_handles, loc='upper right', fontsize=10, title='Clusters')
        return ax