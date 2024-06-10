import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from mdance.tools.bts import diversity_selection, calculate_comp_sim

class KmeansNANI:
    """K-means algorithm with the N-Ary Natural Initialization (NANI).
    
    Attributes
    ----------
    data : array-like of shape (n_samples, n_features)
        Input dataset.
    n_clusters : int
        Number of clusters.
    metric : {'MSD', 'RR', 'JT', etc}
        Metric used for extended comparisons. 
        See `tools.bts.extended_comparison` for all available metrics.
    N_atoms : int
        Number of atoms.
    percentage : int
        Percentage of the dataset to be used for the initial selection of the 
        initial centers. Default is 10.
    init_type : {'div_select', 'comp_sim', 'k-means++', 'random'}
        Type of initiator selection. 
    labels : array-like of shape (n_samples,)
        Labels of each point.
    centers : array-like of shape (n_clusters, n_features)
        Cluster centers.
    n_iter : int
        Number of iterations run.
    cluster_dict : dict
        Dictionary of the clusters and their corresponding indices.
    
    Properties
    ----------
    kmeans : tuple
        Tuple of the labels, centers and number of iterations.
    kmeans_info : tuple
        Tuple of the labels, centers, number of iterations, scores and cluster dictionary.

    Methods
    -------
    __init__(data, n_clusters, metric, N_atoms, percentage=10, init_type='comp_sim'):
        Initializes the class.
    _check_init_type(self)
        Checks the 'init_type' attribute.
    _check_percentage(self)
        Checks the 'percentage' attribute.
    initiate_kmeans(self)
        Initializes the k-means algorithm with the selected initiators.
    kmeans_clustering(self, initiators)
        Gets the k-means algorithm.
    get_kmeans_info(self)
        Gets the k-means information.
    create_cluster_dict(self, labels)
        Creates a dictionary of the clusters and their corresponding indices.
    write_centroids(self, centroids, n_iter)
        Writes the centroids of the k-means algorithm to a file.
    """
    def __init__(self, data, n_clusters, metric, N_atoms, init_type='comp_sim', **kwargs):
        """Initializes the KmeansNANI class.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input dataset.
        n_clusters : int
            Number of clusters.
        metric : {'MSD', 'RR', 'JT', etc}
            Metric used for extended comparisons. 
            See `tools.bts.extended_comparison` for all available metrics.
        N_atoms : int
            Number of atoms.
        percentage : int
            Percentage of the dataset to be used for the initial selection of the 
            initial centers. Default is 10.
        init_type : {'comp_sim', 'div_select', 'k-means++', 'random', 'vanilla_kmeans++'}
            'comp_sim' selects the inital centers based on the diversity in the densest region of the data.
            'div_select' selects the initial centers based on the highest diversity of all data.
            'k-means++' selects the initial centers based on the greedy k-means++ algorithm.
            'random' selects the initial centers randomly.
            'vanilla_kmeans++' selects the initial centers based on the vanilla k-means++ algorithm
        """
        self.data = data
        self.n_clusters = n_clusters
        self.metric = metric
        self.N_atoms = N_atoms
        self.init_type = init_type
        self._check_init_type()
        if self.init_type == 'comp_sim' or self.init_type == 'div_select':
            self.percentage = kwargs.get('percentage', 10)
            self._check_percentage()
    
    def _check_init_type(self):
        """Checks the 'init_type' attribute.

        Raises
        ------
        ValueError
            If 'init_type' is not one of the following: 
            'comp_sim', 'div_select', 'k-means++', 'random', 'vanilla_kmeans++'.
        """
        if self.init_type not in ['comp_sim', 'div_select', 'k-means++', 'random', 'vanilla_kmeans++']:
            raise ValueError('init_type must be one of the following: comp_sim, div_select, k-means++, random, vanilla_kmeans++.')
        
    def _check_percentage(self):
        """Checks the 'percentage' attribute.
        
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
        """Initializes the k-means algorithm with the selected initiating method
        (comp_sim, div_select, k-means++, random, vanilla_kmeans++).

        Returns
        -------
        numpy.ndarray
            The initial centers for k-means of shape (n_clusters, n_features).
        """
        if self.init_type == 'comp_sim':
            n_total = len(self.data)
            n_max = int(n_total * self.percentage / 100)
            comp_sim = calculate_comp_sim(self.data, self.metric, self.N_atoms)
            sorted_comp_sim = sorted(comp_sim, key=lambda item: item[1], reverse=True)
            top_comp_sim_indices = [int(i[0]) for i in sorted_comp_sim][:n_max]
            top_cc_data = self.data[top_comp_sim_indices]
            initiators_indices = diversity_selection(top_cc_data, 100, self.metric, 
                                                     'medoid', self.N_atoms)
            initiators = top_cc_data[initiators_indices]
            if len(initiators) < self.n_clusters:
                raise ValueError('The number of initiators is less than the number of clusters. Try increasing the percentage.')
        elif self.init_type == 'div_select':
            initiators_indices = diversity_selection(self.data, self.percentage, self.metric, 
                                                     'medoid', self.N_atoms)
            initiators = self.data[initiators_indices]
        elif self.init_type == 'vanilla_kmeans++':
            initiators, indices = kmeans_plusplus(self.data, self.n_clusters, 
                                                  random_state=None, n_local_trials=1)
        return initiators
    
    def kmeans_clustering(self, initiators):
        """Executes the k-means algorithm with the selected initiators.

        Parameters
        ----------
        initiators : {numpy.ndarray, 'k-means++', 'random'}
            Method for selecting initial centers.
            'k-means++' selects initial centers in a smart way to speed up convergence.
            'random' selects initial centers randomly.
            If 'initiators' is a numpy.ndarray, it must be of shape (n_clusters, n_features) 

        Returns
        -------
        tuple
            Labels, centers and number of iterations of the k-means algorithm.
        """
        if self.init_type in ['k-means++', 'random']:
            initiators = self.init_type
        elif isinstance(initiators, np.ndarray):
            initiators = initiators[:self.n_clusters]
        n_init = 1
        kmeans = KMeans(self.n_clusters, init=initiators, n_init=n_init, random_state=None)
        kmeans.fit(self.data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        n_iter = kmeans.n_iter_
        return labels, centers, n_iter

    def create_cluster_dict(self, labels):
        """Creates a dictionary with the labels as keys and the indices of the data as values.
        
        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            Labels of the k-means algorithm.
        
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
            Labels of the k-means algorithm.
        
        Returns
        -------
        tuple
            Davies-Bouldin and Calinski-Harabasz scores.
        """
        ch_score = calinski_harabasz_score(self.data, labels)
        db_score = davies_bouldin_score(self.data, labels)
        return ch_score, db_score

    def write_centroids(self, centers, n_iter):
        """Writes the centroids of the k-means algorithm to a file.

        Parameters
        ----------
        centers : array-like of shape (n_clusters, n_features)
            Centroids of the k-means algorithm.
        n_iter : int
            Number of iterations of the k-means algorithm.
        """
        header = f'Number of clusters: {self.n_clusters}, Number of iterations: {n_iter}\n\nCentroids\n'
        np.savetxt('centroids.txt', centers, delimiter=',', header=header)
    
    def execute_kmeans_all(self):
        """Function to complete all steps of KMeans for all different 'init_type' options.

        Returns
        -------
        tuple
            Labels, centers and number of iterations of the k-means algorithm.
        """
        if self.init_type in ['comp_sim', 'div_select', 'vanilla_kmeans++']:
            initiators = self.initiate_kmeans()
            labels, centers, n_iter = self.kmeans_clustering(initiators)
        elif self.init_type == 'k-means++' or self.init_type == 'random':
            labels, centers, n_iter = self.kmeans_clustering(initiators=self.init_type)
        return labels, centers, n_iter
    
def compute_scores(data, labels):
    """Computes the Davies-Bouldin and Calinski-Harabasz scores.
    
    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Labels of the k-means algorithm.
    
    Returns
    -------
    tuple
        Davies-Bouldin and Calinski-Harabasz scores.
    """
    ch_score = calinski_harabasz_score(data, labels)
    db_score = davies_bouldin_score(data, labels)
    return ch_score, db_score