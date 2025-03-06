import copy
import warnings

import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

from mdance.tools.bts import align_traj
from mdance.tools.bts import extended_comparison
from mdance.tools.bts import refine_dis_matrix


class HELM:
    """Hierarchical Extended Linkage Method (HELM) is a class that
    performs hierarchical clustering of clusters. It is uses the 
    *n*-ary similarity framework to merge clusters based on the
    HELM merge schemes.
    
    Parameters
    ----------
    cluster_dict : dict
        dictionary of clusters following the format in the Notes section.
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int, default=1
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1`` 
        for non-MD systems.
    merge_scheme : str, default='inter'
        The merge scheme to use when merging two clusters. 
        Options are ``intra``, ``inter``, and ``half``.
    n_clusters : int
        Number of clusters to terminate the clustering process.
    eps : float
        epsilon MSD value to terminate the clustering process.
    trim_start : bool, default=False
        If True, the initial clusters are trimmed based on the ``trim_val`` or ``trim_k``.
    trim_val : float, default=None
        If ``trim_start`` is True, then this value is used to trim the initial clusters.
    trim_k : int, default=None
        If ``trim_start`` is True, then this value is used to trim the initial clusters.
    align_method : str, default=None
        If ``uni``, the clusters are aligned using the uniform alignment method.
        If ``kron``, the clusters are aligned using the Kronecker alignment method.
    input_top : str, default=None
        The topology file of the MD system.
    input_traj : str, default=None
        The trajectory file of the MD system.
    save_pairwise_sim : bool, default=False
        If True, the pairwise similarity matrix is saved.
    link : str, default=None
        The linkage algorithm to use. See the `Linkage Methods`_ for full descriptions.
    
    Notes
    -----
    The ``cluster_dict`` dictionary should be in the following format:
    ``Clusters = {N1: clustersN1, N2: clustersN2, ...}``
    Nk : int
        Number of clusters in the *k*-th iteration
    clustersNk : list of lists
        Contains the info about clusters in *k*th iteration ``clustersNk = [C1k, C2k, ...]``
    Cik : list of lists 
        Contains information about *i*th cluster in *k*th iteration 
        ``Cik = [Indicesik, (c_sumik, sq_sumik), Nik]``
    Indicesik : list
        Cluster indices of merged clusters. 
        For example, ``[0, 1]`` means cluster 0 and 1 are merged
    c_sumik : array-like of (n_features,)
        A feature array of the column-wsie sum of the data.
    sq_sumik: array-like of (n_features,)
        A feature array of the column-wise sum of the squared data.
    Nik : int 
        Number of elements in the cluster

    .. _Linkage Methods: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    def __init__(self, cluster_dict, metric, N_atoms, merge_scheme='inter',
                 n_clusters=None, eps=None, trim_start=False, align_method=None, 
                 min_samples=0.01, **kwargs):
        self.cluster_dict = cluster_dict
        self.metric = metric
        self.N_atoms = N_atoms
        self.merge_scheme = merge_scheme
        self.link = kwargs.get('link', None)
        self.n_clusters = n_clusters
        self.eps = eps
        self.trim_start = trim_start
        self.trim_val = kwargs.get('trim_val', None)
        self.trim_k = kwargs.get('trim_k', None)
        self.align_method = align_method
        self.min_samples = min_samples
        self.input_top = kwargs.get('input_top', None)
        self.input_traj = kwargs.get('input_traj', None)
        self.save_pairwise_sim = kwargs.get('save_pairwise_sim', False)
        self.cluster_dists = None
        self.total_incoming = sorted(self.cluster_dict)[0]
        self.total_sum = sum(cluster[2] for cluster in self.cluster_dict[self.total_incoming])
        self._check_end_condition()
        self._check_trim()
        self._check_merge_scheme()
        self._check_align_method()
        self._check_min_samples()
    

    def _check_merge_scheme(self):
        """Checks if ``merge_scheme`` is valid"""
        if self.merge_scheme not in ['intra', 'inter', 'half']:
            raise ValueError(f"Invalid merging scheme: Options are 'intra', 'inter', and 'half'.")
    

    def _check_end_condition(self):
        """Ensure that either ``n_clusters`` or ``eps`` is provided, but not both"""
        if (self.n_clusters is None and self.eps is None) or (self.n_clusters is not None and self.eps is not None):
            raise ValueError("You must provide either n_clusters or eps, but not both.")
    

    def _check_trim(self):
        """Ensure that either ``trim_val`` or ``trim_k`` is provided, but not both"""
        if self.trim_start and not (self.trim_val or self.trim_k):
            raise ValueError("If trim_start is True, then either trim_val or trim_k must be provided.")
        if self.trim_val and self.trim_k:
            raise ValueError("You can only provide either trim_val or trim_k, but not both.")
    
    
    def _check_align_method(self):
        """Checks if ``align_method`` is valid"""
        if self.align_method not in ['uni', 'kron', None]:
            raise ValueError(f"Invalid align method: Options are 'uni' and 'kron' or None.")


    def _check_min_samples(self):
        """Checks if ``min_samples`` is valid"""
        if self.min_samples < 0:
            raise ValueError("min_samples must be greater than 0.")
        
        elif 0 < self.min_samples < 1:
            self.min_samples = int(self.min_samples * self.total_sum)
        
        elif self.min_samples >= 1:
            self.min_samples = int(self.min_samples)
        return self.min_samples
    

    def __call__(self):
        return self.run()


    def run(self):
        """Performs HELM clustering of initial clusters.
        
        Returns
        -------
        dict
            dictionary of clusters
        
        Notes
        -----
        ``N`` is the number of clusters in the *k*-th iteration.
        """
        if not self.n_clusters:
            self.n_clusters = 1

        # if self.link exists and is 'ward'
        if self.link:
            self.link_matrix = self.gen_link_matrix()
            return self.link_matrix_to_cluster_dict()
            
        # Optionally trim the initial clusters step
        if self.trim_start:
            self.cluster_dict = self.trim_clusters()
        
        # Perform the clustering.
        N = sorted(self.cluster_dict)[0]
        while N > 1:
            previous_clusters = self.cluster_dict[N]
            new_clusters = self.gen_new_cluster(previous_clusters)
            self.cluster_dict[N - 1] = new_clusters

            # Termination conditions
            if N == self.n_clusters + 1 or not new_clusters:
                break
            
            # Update the iteration number
            N -= 1
        
        return self.cluster_dict
    

    def trim_clusters(self):
        """Trims the initial clusters based on the ``trim_val`` or ``trim_k``.

        Returns
        -------
        dict
            dictionary of clusters
        """
        # Calculate the similarity of each cluster before trim.
        cluster_msds = []
        for i, cluster in enumerate(self.cluster_dict[self.total_incoming]):
            c_sum = cluster[1][0]
            sq_sum = cluster[1][1]
            Nik = cluster[2]
            if Nik < self.min_samples:
                continue

            sim = extended_comparison((c_sum, sq_sum), data_type='condensed', 
                                      metric=self.metric, N=Nik, 
                                      N_atoms=self.N_atoms)
            cluster_msds.append((i, sim))
        cluster_msds = sorted(cluster_msds, key=lambda x: x[1])
        
        # Trim the clusters based on the ``trim_k`` or ``trim_val``
        if self.trim_k:
            self.trim_incoming = len(cluster_msds) - self.trim_k
            new_cluster_dict = {self.trim_incoming: []}
            if self.trim_k >= len(cluster_msds) - 1:
                raise ValueError("trim_k is too large!")
            elif self.trim_k >= len(cluster_msds) / 2:
                warnings.warn("trim_k is more than 50% of the clusters. This may lead to poor clustering.")
            for i, _ in cluster_msds[:-self.trim_k]:
                new_cluster_dict[self.trim_incoming].append(
                    self.cluster_dict[self.total_incoming][i]
                    )
       
        elif self.trim_val:
            self.trim_incoming = len([i for i, sim in cluster_msds if sim < self.trim_val])
            new_cluster_dict = {self.trim_incoming: []}
            for i, sim in cluster_msds:
                if sim < self.trim_val:
                    new_cluster_dict[self.trim_incoming].append(
                        self.cluster_dict[self.total_incoming][i]
                        )
        
        return new_cluster_dict

    
    def gen_new_cluster(self, previous_clusters):
        """Generates new cluster by merging two most similar clusters.
        
        Parameters
        ----------
        previous_clusters : list of lists
            Contains the info about clusters in *k*th iteration 
            ``clustersNk = [C1k, C2k, ...]``
        
        Returns
        -------
        list of lists
            Contains the info about clusters in *k*th iteration 
            ``clustersNk = [C1k, C2k, ...]``
        """
        c1 = len(previous_clusters) + 1
        c2 = c1
        merge_dist = 10000000 #3.08
        c_sumik = -1 #3.08
        sq_sumik = -1
        Nik = -1
        aligned_combined_clusters = np.array([])
        if self.cluster_dists is None:
            self.gen_cluster_dists(previous_clusters)
        else:
            dists_to_new_cluster = np.ones(len(previous_clusters)-1) * np.inf
            for i, _ in enumerate(previous_clusters[:-1]):
                helm_sim = self.calc(previous_clusters, i, j=-1)
                dists_to_new_cluster[i] = helm_sim
            
            # Add the new cluster to the distance matrix.
            self.cluster_dists = np.hstack((self.cluster_dists, dists_to_new_cluster[:, np.newaxis]))
            self.cluster_dists = np.vstack((self.cluster_dists, np.ones(len(previous_clusters))*np.inf))
        
        # Find the two most similar clusters.
        c1, c2 = np.unravel_index(np.argmin(self.cluster_dists, axis=None), self.cluster_dists.shape)
        merge_dist = self.cluster_dists[c1, c2]
        
        # Merge the two most similar clusters
        if self.align_method == 'kron':
            concat_clusters = np.concatenate((previous_clusters[c1][3], previous_clusters[c2][3]))
            aligned_clusters = align_traj(concat_clusters, self.N_atoms, align_method=self.align_method)
            c_sum = np.sum(aligned_clusters, axis=0)
            sq_sum = np.sum(aligned_clusters ** 2, axis=0)
        else:
            c_sum = previous_clusters[c1][1][0] + previous_clusters[c2][1][0]
            sq_sum = previous_clusters[c1][1][1] + previous_clusters[c2][1][1]
        
        c_sumik = c_sum
        sq_sumik = sq_sum
        Nik = previous_clusters[c1][2] + previous_clusters[c2][2]
        if self.align_method:
            aligned_combined_clusters = aligned_clusters

        # Save the new clusters after merging
        new_clusters = []
        # print("merging clusters: ", c1, c2, "with distance: ", merge_dist)
        for i, cluster in enumerate(previous_clusters):
            if i == c1 or i == c2:
                pass
            else:
                new_clusters.append(cluster)
        Indicesik = previous_clusters[c1][0] + previous_clusters[c2][0]
        
        # Two different ways of saving the new cluster
        if self.align_method:
            new_clusters.append([Indicesik, (c_sumik, sq_sumik), Nik, aligned_combined_clusters])
        else:
            new_clusters.append([Indicesik, (c_sumik, sq_sumik), Nik])
        
        # remove distances of the merged clusters
        self.cluster_dists = np.delete(self.cluster_dists, [c1, c2], axis=0)
        self.cluster_dists = np.delete(self.cluster_dists, [c1, c2], axis=1)
        
        if not self.eps or merge_dist < self.eps:
            return new_clusters
        return False


    def calc(self, previous_clusters, i, j):
        """Calculates the similarity between two clusters
        
        Parameters
        ----------
        previous_clusters : list of lists
            Contains the info about clusters in *k*th iteration ``clustersNk = [C1k, C2k, ...]``
        i : int
            Index of the first cluster
        j : int
            Index of the second cluster
        
        Returns
        -------
        float
            similarity between two clusters using the HELM merge schemes
        """    
        c_sum_a = previous_clusters[i][1][0]
        c_sum_b = previous_clusters[j][1][0]
        sq_sum_a = previous_clusters[i][1][1]
        sq_sum_b = previous_clusters[j][1][1]
        n_a = previous_clusters[i][2]
        n_b = previous_clusters[j][2]
        sim_a = extended_comparison([c_sum_a, sq_sum_a], data_type='condensed',
                                    metric=self.metric, N=n_a, N_atoms=self.N_atoms)
        sim_b = extended_comparison([c_sum_b, sq_sum_b], data_type='condensed',
                                    metric=self.metric, N=n_b, N_atoms=self.N_atoms)
        
        if self.align_method == 'kron':
            concat_clusters = np.concatenate((previous_clusters[i][3], previous_clusters[j][3]))
            aligned_clusters = align_traj(concat_clusters, self.N_atoms, align_method=self.align_method)
            c_sum = np.sum(aligned_clusters, axis=0)
            sq_sum = np.sum(aligned_clusters ** 2, axis=0)
        else:
            c_sum = c_sum_a + c_sum_b
            sq_sum = sq_sum_a + sq_sum_b

        n = previous_clusters[i][2] + n_b
        sim = extended_comparison([c_sum, sq_sum], data_type='condensed',
                                  metric=self.metric, N=n, N_atoms=self.N_atoms)
        
        # Different merging schemes for determining which clusters to merge
        if self.merge_scheme == 'intra':
            helm_sim = sim
        elif self.merge_scheme == 'inter':
            helm_sim = ((sim * n ** 2) - (sim_a * n_a ** 2) - (sim_b * n_b ** 2)) / (n_a * n_b)
        elif self.merge_scheme == 'half':
            helm_sim = sim - ((sim_a + sim_b) / 2)
        
        return helm_sim


    def gen_cluster_dists(self, previous_clusters):
        """Generates pairwise similarity matrix for the initial clusters
        
        Parameters
        ----------
        previous_clusters : list of lists
            Contains the info about clusters in *k*th iteration ``clustersNk = [C1k, C2k, ...]``
        
        Returns
        -------
        array-like
            pairwise similarity matrix
        """
        self.cluster_dists = np.ones((len(previous_clusters), len(previous_clusters)))*np.inf
        for i in range(len(previous_clusters)):
            for j in range(i + 1, len(previous_clusters)):
                helm_sim = self.calc(previous_clusters, i, j=j)
                self.cluster_dists[i, j] = helm_sim


    def initial_pairwise_matrix(self):
        """
        Generates pairwise similarity matrix for the initial clusters
        
        Parameters
        ----------
        previous_clusters : list of lists
            Contains the info about clusters in *k*th iteration ``clustersNk = [C1k, C2k, ...]``
        
        Returns
        -------
        array-like
            pairwise similarity matrix
        """
        # Optionally trim the initial clusters step
        if self.trim_start:
            self.cluster_dict = self.trim_clusters()
        
        N = sorted(self.cluster_dict)[0]
        previous_clusters = self.cluster_dict[N]

        distances = []
        for i in range(N):
            distances.append([])
            for j in range(N):
                if i == j:
                    distances[-1].append(0)
                else:
                    helm_sim = self.calc(previous_clusters, i, j)
                    distances[-1].append(helm_sim)

        distances = refine_dis_matrix(distances)
        return distances


    def gen_link_matrix(self):
        """Generates the linkage matrix only for ward linkage
        
        Returns
        -------
        array-like
            linkage matrix
        """
        distances = self.initial_pairwise_matrix()
        linmat = squareform(distances, force='no', checks=True)
        self.link_matrix = linkage(linmat, method=self.link)
        return self.link_matrix
    

    def link_matrix_to_cluster_dict(self):
        nani_sizes = []
        for k in self.cluster_dict:
            for clust in self.cluster_dict[k]:
                nani_sizes.append(clust[2])
        
        # Merge distances
        merge_distances = self.link_matrix[:,2].copy()
        
        # Cluster IDs
        k = len(self.link_matrix) + 1
        cluster_list = []
        for ind in range(k):
            cluster_list.append([ind])
        clust_inds = {k: copy.deepcopy(cluster_list)}
        
        for i, z_level in enumerate(self.link_matrix):
            level_clusters = []
            c1 = int(z_level[0])
            c2 = int(z_level[1])
            cluster_list.append(cluster_list[c1] + cluster_list[c2])
            new_k = k - i - 1
            for clust in clust_inds[k - i]:
                if clust == cluster_list[c1]:
                    pass
                elif clust == cluster_list[c2]:
                    pass
                else:
                    level_clusters.append(clust)
            level_clusters.append(cluster_list[-1])
            clust_inds[new_k] = level_clusters
        
        Clusters = {}
        for level in clust_inds:
            Clusters[level] = []
            for clust in clust_inds[level]:
                n_mols = 0
                for i in clust:
                    n_mols += nani_sizes[i]
                Clusters[level].append([clust, (0, 0), n_mols])
        return Clusters, merge_distances


def z_matrix(cluster_dict):
    """Converts the cluster dictionary to a linkage matrix for plotting dendrogram.
    
    Parameters
    ----------
    cluster_dict : dict
        dictionary of clusters following the format in the Notes section.
    
    Returns
    -------
    array-like
        linkage matrix
    """
    current_clusters = []
    for N in sorted(cluster_dict, reverse=True):
        current_clusters.append([])
        if len(current_clusters) == 1:
            for cluster in cluster_dict[N]:
                current_clusters[-1].append(cluster[0])
        else:
            new = []
            for cluster in cluster_dict[N]:
                if cluster[0] in current_clusters[-2]:
                    current_clusters[-1].append(cluster[0])
                else:
                    new = cluster[0]
            current_clusters[-1].append(new)
    indices_clusters = current_clusters[0]
    
    links = []
    for i in range(len(current_clusters)):
        if i == len(current_clusters) - 1:
            pass
        else:
            union = current_clusters[i+1][-1]
            set_union = set(union)
            for j in range(len(indices_clusters)):
                for k in range(len(indices_clusters)):
                    a = set(indices_clusters[j])
                    b = set(indices_clusters[k])
                    c = a.union(b)
                    if c == set_union:
                        d = set([j, k])
                    else:
                        pass
            links.append(list(d))
            indices_clusters.append(union)
    
    extra_data = []
    for i, c in enumerate(current_clusters[1:]):
        new_indices = c[-1]
        distance = float(i + 1)
        extra_data.append([distance, len(new_indices)]) 
    
    Z = []
    for i in range(len(links)):
        Z.append(links[i] + extra_data[i])
    return np.array(Z)


def compute_scores(list_list, data):
    """Computes Calinski-Harabasz and Davies-Bouldin scores of clusters
    using random labeling.
    
    Returns
    -------
    list
        list of tuples of Calinski-Harabasz and Davies-Bouldin scores of clusters
    """
    cluster_indices = []
    label = []
    count = 0
    for cluster in list_list:
        cluster_indices.append(cluster[0])
        Nik = cluster[2]
        label.extend([count] * Nik)
        count += 1
    label = np.array(label)
    
    if len(np.unique(label)) == 1:
        return None, None
    else:
        ch_score = calinski_harabasz_score(data, label)
        db_score = davies_bouldin_score(data, label)
        return ch_score, db_score