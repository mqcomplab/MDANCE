import warnings

import numpy as np
from sklearn.cluster import kmeans_plusplus, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from mdance.tools.bts import align_traj
from mdance.tools.bts import calculate_comp_sim
from mdance.tools.bts import calculate_medoid
from mdance.tools.bts import diversity_selection
from mdance.tools.bts import extended_comparison


class ExtendedQuality:
    """
    Extended quality clustering algorithm is an extension of the radial 
    threshold algorithm. It grows clusters from seeds and can rejects low density 
    clusters.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        A feature array.
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int
        Number of atoms in the system used for normalization. 
        ``N_atoms=1`` for non-Molecular Dynamics datasets.
    threshold : float
        The distance between the seed of the subcluster and a new sample 
        should be lesser than the threshold. 
    n_seeds : {float, int}
        Number of seeds to be used per iteration. Default is 1.
        float: Real number between (0, 1). Indicates the % of the total data.
        int: Number of seeds.
    seed_method : {'comp_sim', 'greedy', 'medoid', 'mini_batch_kmeans', 'vanilla'}
        Method used to select the initial seeds.
    check_sim : bool, default False
        If True, validates the proposed cluster against a similarity threshold 
        to ensure it meets acceptable criteria.
    reject_lowd : bool, default False
        If True, will reject low density clusters if they are below the minimum cluster size.
    align_method: {'uni', 'kron', None}, optional
        Alignment method used for the data. Default is None, which means no alignment.
        'uni' is a uniform alignment method.
        'kron' is a Kronecker alignment method.
    percentage : int, default=10
        Percentage of the dataset to be used for the initial selection 
        of the initial centers. (**kwargs)
    sim_threshold : float
        The largest similarity value that is acceptable for the proposed 
        cluster. (**kwargs)
    min_samples : {float, int}, default=10
        Minimum number of data points required in a cluster. (**kwargs)
        float: Real number between (0, 1). Indicates the % of the total data.
        int: Number of data points.
    """
    def __init__(self, data, threshold, metric, N_atoms, seed_method='greedy',
                 n_seeds=1, check_sim=False, reject_lowd=True, **kwargs):
        self.data = data
        self.threshold = threshold
        self.metric = metric
        self.N_atoms = N_atoms
        self.seed_method = seed_method
        self.n_seeds = n_seeds
        self.check_sim = check_sim
        self.reject_lowd = reject_lowd
        
        self.n_objects = len(self.data)
        
        # Alignment method criteria
        self.align_method = kwargs.get('align_method', None)
        self.align_method = self._check_align_method()
        
        # Percentage criteria
        if self.seed_method == 'comp_sim':
            self.percentage = kwargs.get('percentage', 10)
            self.percentage = self._check_percentage()
        
        # Number of seeds criteria
        self.n_seeds = self._check_n_seeds()

        # sim_threshold criteria
        if self.check_sim:
            self.sim_threshold = kwargs.get('sim_threshold')
            self.sim_threshold = self._check_sim_threshold()

        # Minimum cluster size criteria
        if self.reject_lowd:
            self.min_samples = kwargs.get('min_samples', 10)
            self.min_samples = self._check_min_samples()
    

    def _check_align_method(self):
        """
        Checks the ``align_method`` attribute.
        
        Raises
        ------
        ValueError
            If align_method is not one of the following: 
            [``uni``, ``kron``, ``None``].
        
        Returns
        -------
        str
            Alignment method.
        """
        if self.align_method not in ['uni', 'kron', None]:
            raise ValueError(f"Invalid alignment method. Must be one of the following: ['uni', 'kron', None].")
        return self.align_method
    

    def _check_percentage(self):
        """
        Checks the ``percentage`` attribute.
        
        Raises
        ------
        ValueError
            If ``percentage`` is not an integer or not ``[0, 100]``.
        
        Returns
        -------
        int
            Percentage of the dataset to be used for the initial selection
        """
        if not 0 <= self.percentage <= 100 or not isinstance(self.percentage, int):
            raise ValueError("percentage must be an integer [0, 100].")
        return self.percentage
    

    def _check_n_seeds(self):
        """
        Checks the ``n_seeds`` attribute.
        
        Raises
        ------
        ValueError
            If ``n_seeds`` is not an integer or not less than the number of objects.
        
        Returns
        -------
        int
            Number of seeds to be used.
        """
        if 0 < self.n_seeds < 1:
            self.n_seeds = int(self.n_objects * self.n_seeds)
        elif isinstance(self.n_seeds, int):
            if self.n_seeds >= self.n_objects:
                raise ValueError(f"n_seeds, {self.n_seeds} must be less than the n_objects, {self.n_objects}.")
            self.n_seeds = self.n_seeds
        return self.n_seeds
    

    def _check_sim_threshold(self):
        """
        Checks the ``sim_threshold`` attribute.
        
        Raises
        ------
        ValueError
            If ``sim_threshold`` is not specified when ``check_sim`` is True.
        
        Returns
        -------
        float
            Threshold for the similarity of the proposed cluster.
        """
        if not self.sim_threshold:
            raise ValueError("sim_threshold must be specified if check_msd is True.")
        self.sim_threshold = self.sim_threshold
        return self.sim_threshold
    

    def _check_min_samples(self):
        """
        Checks the ``min_samples`` attribute.
        
        Raises
        ------
        ValueError
            If ``min_samples`` is not specified when ``reject_lowd`` is True.
        ValueError
            If ``min_samples`` is not an integer or not less than the number of objects.
        
        Returns
        -------
        int
            Minimum number of data points in a cluster.
        """
        if not self.min_samples:
            raise ValueError("min_samples must be specified if reject_lowd is True.")
        elif 0 < self.min_samples < 1:
            self.min_samples = int(self.n_objects * self.min_samples)
        elif isinstance(self.min_samples, int):
            if self.min_samples > self.n_objects:
                raise ValueError(f"min_samples, {self.min_samples} must be less than the n_objects, {self.n_objects}.")
            self.min_samples = self.min_samples
        return self.min_samples
    

    def run(self):
        """
        Run the ExtendedQuality algorithm.

        Returns
        -------
        dict
            Key: iteration number, value: numpy.ndarray of the cluster members.
        """
        cluster_dict = self.grow_clusters()
        non_empty_clusters = {k: v for k, v in cluster_dict.items() if v}
        sorted_clusters = {k: v for k, v in sorted(non_empty_clusters.items(), 
                                                   key=lambda item: len(item[1]), 
                                                   reverse=True)}
        renumbered_clusters = {}
        for i, v in enumerate(sorted_clusters.values()):
            renumbered_clusters[i] = v
        if not renumbered_clusters:
            warnings.warn(f"threshold {self.threshold}. All clusters are empty. sim_threshold may be too low.")
        return renumbered_clusters
    

    def comp_sim_seeds(self):
        """
        Selects the inital centers based on the diversity in the high density 
        region of the data using the *n*-ary similarity.
        
        Returns
        -------
        numpy.ndarray
            (n_seeds, n_features) array of the initial seeds.
        
        Notes
        -----
        A complementary similarity is calculated for each point in the dataset.
        Next, the top n% of the points are selected for diversity selection.
        The first ``n_seeds`` number of points are selected as the seeds.
        """
        n_max = int(self.percentage * self.n_objects / 100)
        comp_sim = calculate_comp_sim(self.data, self.metric, self.N_atoms)
        sorted_indices = np.argsort(comp_sim)  
        top_comp_sim_indices = sorted_indices[-n_max:]
        top_cc_data = self.data[top_comp_sim_indices]
        medoids_indices = diversity_selection(top_cc_data, 100, self.metric, 
                                              self.N_atoms, 'medoid')
        seeds = top_cc_data[medoids_indices]
        return seeds
    

    def greedy_seeds(self):
        """
        Select the initial centers using the greedy *k*-means++ algorithm. 
        (Arthur and Vassilvitskii, 2007).
        
        Returns
        -------
        numpy.ndarray
            (n_seeds, n_features) array of the initial seeds.
        """
        centers, indices = kmeans_plusplus(self.data, n_clusters=self.n_seeds, 
                                           random_state=42)
        seeds = self.data[indices]
        return seeds


    def find_medoids(self):
        """
        Finds the seeds by selecting the medoids using the complementary similarity.

        Returns
        -------
        numpy.ndarray
            (n_seeds, n_features) array of the initial seeds.
            
        Notes
        -----
        A complementary similarity is calculated for each point in the dataset.
        Then, the first ``n_seeds`` number of points are selected as the seeds.
        """
        comp_sim = calculate_comp_sim(self.data, self.metric, self.N_atoms)
        sorted_indices = np.argsort(comp_sim)  
        medoid_indices = sorted_indices[-self.n_seeds:]
        seeds = self.data[medoid_indices]
        return seeds
    

    def mini_batch_kmeans_seeds(self):
        """
        Select the initial centers using the mini-batch *k*-means algorithm.
        
        Returns
        -------
        numpy.ndarray
            (n_seeds, n_features) array of the initial seeds.
        """
        mbk = MiniBatchKMeans(n_clusters=self.n_seeds, random_state=42)
        mbk.fit(self.data)
        seeds = mbk.cluster_centers_
        return seeds
    

    def vanilla_seeds(self):
        """
        Select the initial centers using the vanilla *k*-means++ algorithm.
        
        Returns
        -------
        numpy.ndarray
            (n_seeds, n_features) array of the initial seeds.
        """
        centers, indices = kmeans_plusplus(self.data, n_clusters=self.n_seeds, 
                                           random_state=42, n_local_trials=1)
        seeds = self.data[indices]
        return seeds


    def _choose_seed_method(self):
        """
        Chooses the seed method based on the ``seed_method`` attribute.
        
        Raises
        ------
        ValueError
            If ``seed_method`` is not one of the following: 
            [``comp_sim``, ``greedy``, ``medoid``, ``mini_batch_kmeans``, ``vanilla``].
        
        Returns
        -------
        numpy.ndarray
            (n_seeds, n_features) array of the initial seeds.
        """
        if self.seed_method == 'comp_sim':
            seeds = self.comp_sim_seeds()
        elif self.seed_method == 'greedy':
            seeds = self.greedy_seeds()
        elif self.seed_method == 'medoid':
            seeds = self.find_medoids()
        elif self.seed_method == 'mini_batch_kmeans':
            seeds = self.mini_batch_kmeans_seeds()
        elif self.seed_method == 'vanilla':
            seeds = self.vanilla_seeds()
        else:
            raise ValueError(f"Invalid seed method. Must be one of the following: ['comp_sim', 'greedy', 'medoid', 'mini_batch_kmeans', 'vanilla'].")
        return seeds


    def grow_clusters(self):
        """
        The heart of the ``ExtendedQuality`` algorithm. 
        
        Returns
        -------
        dict
            Key: iteration number, value: numpy.ndarray of the cluster members.
        
        Notes
        -----
        1. Initial seeds are selected using the method in ``seed_method`` attribute.
        2. Each seed proposes a cluster by adding available objects within the radial threshold.
        3. The winner seed cluster is the most dense cluster. If there are multiple, 
            the one with the lowest similarity is chosen.
        4. Objects in the winner seed cluster are removed from the data.
        5. If ``check_sim`` is True, clusters above the similarity threshold are rejected.
        6. if ``reject_lowd`` is True, clusters below the ``min_samples`` are rejected.
        7. Repeat steps 1-6 until there are less than 2 objects left in the data because
            it is not possible to determine seeds with 2 or less objects.
        """
        seed_clusters = {}
        iteration = 0
        while len(self.data) > 2 and len(self.data) > self.n_seeds:
            # Select the initial seeds
            all_seeds = self._choose_seed_method()
            
            # Propose clusters for each seed
            candidate_cluster = []
            for index, seed in enumerate(all_seeds):
                current_cluster = []
                for i, candidate in enumerate(self.data):
                    value = extended_comparison(np.array([seed, candidate]), data_type='full', 
                                                metric=self.metric, N_atoms=self.N_atoms)
                    if value <= self.threshold:
                        current_cluster.append(candidate)
                candidate_cluster.append(current_cluster)
            
            # Find the winner cluster
            max_indices = [i for i, cluster in enumerate(candidate_cluster) \
                if len(cluster) == max(len(c) for c in candidate_cluster)]
            max_clusters = [candidate_cluster[i] for i in max_indices]
            if len(max_indices) == 1:
                winner = max_clusters[0]
                seed_clusters[iteration] = winner
            else:
                sim_threshold = 9999
                index = len(max_clusters) + 1
                for i, c in enumerate(max_clusters):
                    sim = extended_comparison(np.array(c), data_type='full', 
                                              metric=self.metric, N_atoms=self.N_atoms)
                    if sim < sim_threshold:
                        sim_threshold = sim
                        index = i
                if sim_threshold == 9999:
                    index = 0
                winner = max_clusters[index]
                seed_clusters[iteration] = winner
            
            # Remove winner cluster from the original dataset
            for i in winner:
                self.data = self.data[~np.all(self.data == i, axis=1)]
            
            # Check similarity and reject low density clusters
            if not self.check_sim or self._check_sim(winner):
                seed_clusters = self._low_density_termination(seed_clusters, winner, iteration)
                self.data = align_traj(self.data, self.N_atoms, align_method=self.align_method)
                iteration += 1
            else:
                seed_clusters[iteration] = []
                break
        return seed_clusters
    

    def _check_sim(self, current_cluster):
        """
        Checks the similarity of the proposed cluster.
        
        Parameters
        ----------
        current_cluster : list
            List of points in the current cluster.
        
        Returns
        -------
        bool
            True if the similarity value is less than the ``sim_threshold``, False otherwise.
        """
        sim = extended_comparison(np.array(current_cluster), data_type='full',
                                  metric=self.metric, N_atoms=self.N_atoms)
        if sim <= self.sim_threshold:
            return True
        else:
            return False


    def _low_density_termination(self, seed_clusters, winner, iteration):
        """
        Removes low density clusters if it is below the minimum cluster size.
        
        Parameters
        ----------
        seed_clusters : dict
            Winner clusters from all previous iterations are stored in this dictionary.
        winner : list
            Winner cluster from current iteration that is to be checked for rejection.
        iteration : int
            Iteration number.
        
        Returns
        -------
        dict
            Updated seed_clusters dictionary.
            Key: iteration number, value: numpy.ndarray of the cluster members.
        """
        if self.reject_lowd:
            if len(winner) >= self.min_samples: 
                seed_clusters[iteration] = winner   
            else:
                seed_clusters[iteration] = []
        else:
            seed_clusters[iteration] = winner
        return seed_clusters
    

    def calculate_populations(self, clusters):
        """
        Calculate the populations of the clusters.

        Returns
        -------
        dict
            Key: cluster number, value: cluster population.
        """
        each_top10_frac = []
        top10_total_items = 0
        for i, v in enumerate(clusters.values()):
            if i < 10:
                each_top10_frac.append('%.6f' % (len(v) / self.n_objects))
                top10_total_items += len(v)
        top10_fraction = '%.6f' % (top10_total_items / self.n_objects)
        pop_list = [self.threshold, len(clusters), top10_fraction] + each_top10_frac
        return pop_list 
    

    def calculate_best_frames(self, clusters, n_structures=10, sorted_by='frame'):
        """
        Extract the best n structures for each cluster.

        Parameters
        ----------
        clusters : dict
            Dictionary of the clusters and their corresponding indices.
        n_structures : int, default=10
            Number of structures to be extracted for each cluster.
        sorted_by : {'frame', 'similarity'}, default='frame'
            Sort the best structures by frame number or similarity value.
        
        Returns
        -------
        numpy.ndarray
            Array of the best frames.
        """
        best_frames = []
        for v in clusters.values():
            medoid_index = calculate_medoid(np.array(v), metric=self.metric, N_atoms=self.N_atoms)
            medoid = v[medoid_index]
            msd_to_medoid = []
            for i, frame in enumerate(v):
                msd_to_medoid.append((i, extended_comparison(np.array([frame, medoid]), data_type='full', 
                                                             metric=self.metric, N_atoms=self.N_atoms)))
            msd_to_medoid = np.array(msd_to_medoid)
            if sorted_by == 'frame':
                sorted_indices = np.argsort(msd_to_medoid[:, 0])
            elif sorted_by == 'similarity':
                sorted_indices = np.argsort(msd_to_medoid[:, 1])
            best_n_structures = [v[idx] for idx in sorted_indices[:n_structures]]
            best_frames.append(best_n_structures)
        return np.array(best_frames)
    

    def find_best_frames_indices(self, best_frames, sieve):
        """
        Find the indices of the best frames.

        Parameters
        ----------
        best_frames : numpy.ndarray
            Array of the best frames.
        sieve : int
            The sieve value used to select the frames.
        
        Returns
        -------
        numpy.ndarray
            Array of the best frames indices.
        """
        best_frames_indices = []
        for i, frame in enumerate(self.data):
            i = i * sieve
            for k, v in enumerate(best_frames):
                if any((frame == x).all() for x in v):
                    best_frames_indices.append((i, k))
                    break
        best_frames_indices = np.array(best_frames_indices)
        best_frames_indices = best_frames_indices[best_frames_indices[:, 1].argsort()]
        return best_frames_indices
    

    def labels(self, clusters, sieve):
        """
        Assigns labels to the clusters.

        Parameters
        ----------
        clusters : dict
            Dictionary of the clusters and their corresponding indices.
        
        Returns
        -------
        numpy.ndarray
            Array of the cluster labels.
        """
        frame_vs_cluster = []
        for i, frame in enumerate(self.data):
            i = i * sieve
            for k, v in clusters.items():
                if any((frame == x).all() for x in v):
                    frame_vs_cluster.append((i, k))
                    break
        frame_vs_cluster = np.array(frame_vs_cluster)
        return frame_vs_cluster


def compute_scores(results):
    """
    Compute the Calinski-Harabasz (CH) and Davies-Bouldin (DB) scores for the clusters.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input dataset.
    results : dict
        Dictionary of the clusters and their corresponding indices.
        
    Returns
    -------
    tuple
        A tuple of the CH and DB scores in that order.
        
    Notes
    -----
    Labels are assigned based on number of clusters. If there is only one cluster, 
        the CH and DB scores cannot be calculated and None is returned.
        
    Example
    ----
    >>> import numpy as np
    >>> from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    >>> from mdance.cluster.equal import compute_scores
    >>> data = np.array([[1, 2], [1, 4], [1, 0],
    ...                  [4, 2], [4, 4], [4, 0]])  
    >>> results = {0: [0, 1, 2], 1: [3, 4, 5]}
    >>> ch, db = compute_scores(data, results)
    >>> print(ch, db)
    3.375 0.8888888888888888
    """
    data = np.array([frame for cluster in results.values() for frame in cluster])
    labels = []
    count = 0
    for v in results.values():
        labels.extend([count] * len(v))
        count += 1
    labels = np.array(labels)
    if len(np.unique(labels)) == 1 or data.shape[0] == 0:
        return None, None
    else:
        ch_score = calinski_harabasz_score(data, labels)
        db_score = davies_bouldin_score(data, labels)
        return ch_score, db_score