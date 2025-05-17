import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

from mdance.tools.bts import diversity_selection, calculate_comp_sim


class KmeansNANI:
    """*k*-means NANI clustering alogorithm (*N*-Ary Natural Initialization).
    
    Valid Values for ``init_types``: (*k* means number of clusters)
    | ``strat_all``: A number of bins are computed based on specified percentage of the data. Stratified sampling is then applied, and the first *k* points from the stratified data are selected as the initial centers.
    | ``strat_reduced``: Identifies high-density regions using complementary similarity, selecting a specified percentage of points. Applies stratified sampling to this subset using a number of bins based on the subset size, and selects the first *k* points as initial centers.
    | ``comp_sim``: Identifies high-density regions using complementary similarity, selecting a percentage% of the data. From this subset, diversity selection (with ``comp_sim`` as the sampling method) is used to choose the first *k* points as the initial centers.
    | ``div_select``: Applies diversity selection (using ``comp_sim`` as the sampling method) on specified percentage% of points. First *k* points are the initial centers.
    | ``k-means++`` selects the initial centers based on the greedy *k*-means++ algorithm.
    | ``random`` selects the initial centers randomly.
    | ``vanilla_kmeans++`` selects the initial centers based on the vanilla *k*-means++ algorithm.
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        A feature array.
    n_clusters : int
        Number of clusters.
    metric : str
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int
        Number of atoms in the Molecular Dynamics (MD) system. ``N_atom=1`` 
        for non-MD systems.
    init_type : str, default='comp_sim'
        Type of initiator selection for initiating *k*-means. It must be an 
        options allowed by :class:`mdance.cluster.nani.KmeansNANI`.
    percentage : int, default=10
        Percentage of the dataset to be used for the initial selection of the 
        initial centers. (**kwargs)
    
    Attributes
    ----------
    labels : array-like of shape (n_samples,)
        An array of the labels of each point.
    centers : array-like of shape (n_clusters, n_features)
        An array of the cluster centers.
    n_iter : int
        Number of iterations until coverage.
    cluster_dict : dict
        Dictionary of the clusters and their corresponding indices.
    """
    def __init__(self, data, n_clusters, metric, N_atoms, init_type='strat_all', 
                 **kwargs):
        self.data = data
        self.n_clusters = n_clusters
        self.metric = metric
        self.N_atoms = N_atoms
        self.init_type = init_type
        self._check_init_type()
        if self.init_type in ['comp_sim', 'div_select', 'strat_reduced', 'strat_all']:
            self.percentage = kwargs.get('percentage', 10)
            self._check_percentage()
    
    
    def _check_init_type(self):
        """Checks the ``init_type`` attribute.

        Raises
        ------
        ValueError
            If ``init_type`` is not one of the following: ``comp_sim``, ``div_select``, 
            ``k-means++``, ``random``, ``vanilla_kmeans++``.
        """
        if self.init_type not in ['comp_sim', 'div_select', 'k-means++', 
                                  'random', 'vanilla_kmeans++', 'strat_all',
                                  'strat_reduced']:
            raise ValueError('init_type must be one of the following: comp_sim, \
                             div_select, k-means++, random, vanilla_kmeans++, strat_all, \
                             strat_reduced.')
    
    
    def _check_percentage(self):
        """Checks the ``percentage`` attribute.
        
        Raises
        ------
        TypeError
            If percentage is not an integer.
        ValueError
            If percentage is not between 0 and 100.
        """
        if not isinstance(self.percentage, int):
            raise TypeError('percentage must be an integer [0, 100].')
        if not 0 <= self.percentage <= 100:
            raise ValueError('percentage must be an integer [0, 100].')
    
    
    def initiate_kmeans(self):
        """Initializes the *k*-means algorithm with the selected initiators.
        
        Raises
        ------
        ValueError
            If the number of initiators is less than the number of clusters.
        
        Returns
        -------
        numpy.ndarray
            The initial centers for *k*-means of shape (n_clusters, n_features).
        """
        if self.init_type in ['strat_reduced', 'comp_sim']:
            n_total = len(self.data)
            n_max = int(n_total * self.percentage / 100)
            comp_sim = calculate_comp_sim(self.data, self.metric, self.N_atoms)
            sorted_indices = np.argsort(comp_sim)
            top_comp_sim_indices = sorted_indices[-n_max:]
            top_cc_data = self.data[top_comp_sim_indices]

            if self.init_type == 'strat_reduced':
                sampling_method, start_method = 'strat', 'medoid'
            else:
                sampling_method, start_method = 'comp_sim', 'medoid'
            initiator_idxs = diversity_selection(top_cc_data, 100, self.metric, self.N_atoms, 
                                                 sampling_method, start_method)
            initiators = top_cc_data[initiator_idxs]
            
        elif self.init_type == 'strat_all':
            initiator_idxs = diversity_selection(self.data, self.percentage, self.metric, 
                                                 self.N_atoms, 'strat', 'medoid')
            initiators = self.data[initiator_idxs]
        
        elif self.init_type == 'div_select':
            initiator_idxs = diversity_selection(self.data, self.percentage, self.metric, 
                                                 self.N_atoms, 'comp_sim', 'medoid')
            initiators = self.data[initiator_idxs]
        
        elif self.init_type == 'vanilla_kmeans++':
            initiators, indices = kmeans_plusplus(self.data, self.n_clusters, random_state=None, 
                                                  n_local_trials=1)
        
        if len(initiators) < self.n_clusters:
            raise ValueError('The number of initiators is less than the number of clusters. Try increasing the percentage.')
        
        return initiators[:self.n_clusters]
    
    
    def kmeans_clustering(self, initiators):
        """Executes the *k*-means algorithm with the selected initiators.

        Parameters
        ----------
        initiators : {numpy.ndarray, 'k-means++', 'random'}
            Method for selecting initial centers.
            ``k-means++`` selects initial centers in a smart way to speed up convergence.
            ``random`` selects initial centers randomly.
            numpy.ndarray selects initial centers based on the input array.

        Returns
        -------
        tuple
            Labels, centers and number of iterations.
        """
        if self.init_type in ['k-means++', 'random']:
            initiators = self.init_type
        n_init = 1
        kmeans = KMeans(self.n_clusters, init=initiators, n_init=n_init, 
                        random_state=None)
        kmeans.fit(self.data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        n_iter = kmeans.n_iter_
        return labels, centers, n_iter


    def create_cluster_dict(self, labels):
        """Creates a dictionary with the labels as keys and the indices of the 
        data as values.
        
        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            Cluster labels.
        
        Returns
        -------
        dict
            Dictionary with the labels as keys and the indices of the data as values.
        """
        dict_labels = {}
        for i in range(self.n_clusters):
            dict_labels[i] = np.where(labels == i)[0]
        return dict_labels
    
    
    def compute_scores(self, labels):
        """Computes the Davies-Bouldin and Calinski-Harabasz scores.
        
        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            Cluster labels.
        
        Returns
        -------
        tuple
            Davies-Bouldin and Calinski-Harabasz scores.
        """
        ch_score = calinski_harabasz_score(self.data, labels)
        db_score = davies_bouldin_score(self.data, labels)
        return ch_score, db_score


    def write_centroids(self, centers, n_iter):
        """Writes the centroids to a file.

        Parameters
        ----------
        centers : array-like of shape (n_clusters, n_features)
            Centroids of the clusters.
        n_iter : int
            Number of iterations until converage.
        """
        header = f'Number of clusters: {self.n_clusters}, Number of iterations: {n_iter}\n\nCentroids\n'
        np.savetxt('centroids.txt', centers, delimiter=',', header=header)
    
    
    def execute_kmeans_all(self):
        """Function to complete all steps of NANI for all different ``init_type`` options.

        Returns
        -------
        tuple
            Labels, centers and number of iterations.
        """
        if self.init_type == 'k-means++' or self.init_type == 'random':
            labels, centers, n_iter = self.kmeans_clustering(initiators=self.init_type)
        else:
            initiators = self.initiate_kmeans()
            labels, centers, n_iter = self.kmeans_clustering(initiators)
        return labels, centers, n_iter


def compute_scores(data, labels):
    """Computes the Calinski-Harabasz and Davies-Bouldin scores.
    
    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Cluster labels.
    
    Returns
    -------
    tuple
        Calinski-Harabasz and Davies-Bouldin scores (in that order).
    """
    ch_score = calinski_harabasz_score(data, labels)
    db_score = davies_bouldin_score(data, labels)
    return ch_score, db_score