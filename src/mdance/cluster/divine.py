import numpy as np
import warnings
from mdance.cluster.nani import KmeansNANI
from sklearn.cluster import KMeans
from mdance.tools.bts import extended_comparison, calculate_medoid, calculate_outlier
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

class Divine:
    """Divisive hierarchical clustering algorithm for molecular dynamics analysis.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input feature array.
    split : {'MSD', 'radius', 'weighted_MSD'}, default='weighted_MSD'
        Strategy to evaluate and choose which cluster to split.
    anchors : {'nani', 'outlier_pair', 'splinter_split'}, default='nani'
        Method used to determine anchor points for cluster splitting.
    init_type : {'strat_all', 'strat_reduced', 'comp_sim', 'div_select', 'k-means++', 'random'}, default='strat_all'
        Initialization strategy used when `anchors='nani'`.
    end : {'k', 'points'}, default='k'
        Stopping criterion: 'k' for fixed cluster count, 'points' for singleton clusters.
    k : int or None
        Desired number of final clusters when `end='k'`.
    refine : bool, default=True
        Whether to refine splits using an additional k-means clustering.
    N_atoms : int, default=1
        Number of atoms per data point. Used to normalize distance metrics.
    threshold : float, default=0
        Minimum relative size of a valid subcluster, in [0, 1].
    percentage : int, default=10
        Percentage of data used for initiator selection in NANI.
    """

    def __init__(self, data, split='weighted_MSD', anchors='nani', init_type='strat_all', end='k', k=None, refine=True, N_atoms=1, threshold=0, percentage=10):

        self.data = data
        self.split = split
        self.anchors = anchors
        self.end = end
        self.k = k
        self.metric = 'MSD'
        self.N_atoms = N_atoms
        self.refine = refine
        self.init_type = init_type
        self.percentage = percentage
        self.threshold = threshold

    def _check_split(self):
        """Check if split is one of 'MSD', 'radius', or 'weighted_MSD'."""
        if self.split not in ['MSD', 'radius', 'weighted_MSD']:
            raise ValueError(f"split must be either 'MSD', 'radius', or 'weighted_MSD'. Got {self.split}")

    def _check_anchors(self):
        """Check if anchors is either 'nani', 'outlier_pair', or 'splinter_split'.        """
        if self.anchors not in ['nani', 'outlier_pair','splinter_split']:
            raise ValueError(f"anchors must be either 'nani', 'outlier_pair', 'splinter_split'. Got {self.anchors}")
        if self.anchors == 'nani':
            if self.init_type not in ['strat_all', 'strat_reduced', 'comp_sim','div_select', 'k-means++', 'random']:
                raise ValueError(f"init_type must be either 'strat_all', 'strat_reduced', 'comp_sim','div_select', 'k-means++', 'random'. Got {self.init_type}")
            
    def _check_end(self):
        """Check if end is either 'points' or 'k'."""
        if self.end not in ['points', 'k']:
            raise ValueError(f"end must be either 'points' or 'k'. Got {self.end}")

    def _check_k(self):
        """Ensure that k is provided if end is set to 'k'."""
        if self.end == 'k' and self.k is None:
            raise ValueError("k must be provided if end is 'k'")
        if self.k is not None and self.k > len(self.data):
            raise ValueError(f"Requested k = {self.k} exceeds number of data points ({len(self.data)}).")

    def _check_threshold(self):
        """Check if threshold is a numeric value between 0 and 1."""
        if not isinstance(self.threshold, (int, float)):
            raise TypeError("threshold must be a numeric type (int or float).")
        if not (0 <= self.threshold <= 1):
            raise ValueError("threshold must be between 0 and 1.")

    def _check_params(self):
        self._check_split()
        self._check_anchors()
        self._check_end()
        self._check_k()
        self._check_threshold()

    def run(self):
        """Execute the divisive clustering algorithm.

        Returns
        -------
        clusters : list of ndarray
            A list of arrays, each containing the indices for a cluster.
        labels : ndarray of shape (n_samples,)
            Cluster labels assigned to each sample.
        scores : list of tuple
            Each tuple contains (number of clusters, CH score, DB score).
        msds : list of tuple
            Each tuple contains (iteration, cluster_index, cluster_size, MSD value).
        """

        if self.data is None or self.data.size == 0:
            return [], np.array([]), [], []
        self._check_params()

        return self.divisive_algorithm()

    def divisive_algorithm(self):
        """Main loop for recursively splitting clusters.

        Returns
        -------
        clusters : list of ndarray
            Final list of clusters.
        labels : ndarray
            Cluster label for each data point.
        scores : list of tuple
            CH and DB scores per iteration.
        msds : list of tuple
            MSD scores and sizes per cluster per iteration.
        """

        n_total = len(self.data)
        clusters = [np.arange(n_total)]
        scores = []
        msds = []
        labels = self._create_label_array(clusters, n_total)
        min_frames = max(1, int(self.threshold * n_total))
        failed_splits = set()
        iteration = 1

        while True:
            # Stopping conditions
            if self.end == 'k':
                if len(clusters) >= self.k:
                    break
            elif self.end == 'points':
                if all(len(c) == 1 for c in clusters):
                    break

            # Determine which cluster to split
            cluster_to_split = self.select_cluster_to_split(clusters, failed_splits)
            if cluster_to_split < 0:
                warnings.warn(
                    f"No more cluster splits possible that would yield valid subclusters (min_frames={min_frames}). Consider loosening the threshold (currently {self.threshold}) Current number of clusters: {len(clusters)}.")
                break

            # Split the selected cluster into sub-clusters
            new_subclusters = self.split_selected_cluster(clusters, cluster_to_split)

            # Filter new subclusters with min_frames
            valid_subclusters = [subcluster for subcluster in new_subclusters if len(subcluster) >= min_frames]
            
            if len(valid_subclusters) < 2:
                warnings.warn(f"Cluster {cluster_to_split} could not be split meaningfully — skipping.")
                failed_splits.add(cluster_to_split)
                continue

            # Delete the cluster to split and add the new subclusters
            del clusters[cluster_to_split]
            clusters.extend(valid_subclusters)

            # Update the label array
            labels = self._create_label_array(clusters, n_total)

            failed_splits.clear()  # Reset failed splits for the next iteration

            # Compute MSD and size of each cluster 
            for i, cluster in enumerate(clusters):
                cluster_data = self.data[cluster]
                msd_value = extended_comparison(cluster_data, data_type='full', metric=self.metric, N_atoms=self.N_atoms)
                msds.append((iteration, i, len(cluster), msd_value))

            # Compute clustering scores 
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                ch_score, db_score = self.compute_scores(labels)
                scores.append((len(clusters), ch_score, db_score))
            else:
                scores.append((len(clusters), 0, 0))

            iteration += 1

        return clusters, labels, scores, msds

    def select_cluster_to_split(self, clusters, failed_splits=None):
        """Select the cluster with the highest score for a criteria.

        Parameters
        ----------
        clusters : list of ndarray
            Current list of clusters.
        failed_splits : set of int, optional
            Indices of clusters previously deemed unsplittable.

        Returns
        -------
        int
            Index of the cluster to split, or -1 if no suitable split found.
        """

        if failed_splits is None:
            failed_splits = set()
        top_cluster = -1
        best_score = -1

        for i, indices in enumerate(clusters):
            if i in failed_splits or len(indices) < 2:
                continue

            subdata = self.data[indices]

            if self.split == 'MSD':
                val = extended_comparison(subdata, data_type='full', metric=self.metric, N_atoms=self.N_atoms)

            elif self.split == 'radius':
                medoid_idx = calculate_medoid(subdata, self.metric, self.N_atoms)
                medoid = subdata[medoid_idx]
                dists = np.sum((subdata - medoid) ** 2, axis=1) / self.N_atoms
                val = np.max(dists)

            elif self.split == 'weighted_MSD':
                msd_val = extended_comparison(subdata, data_type='full', metric=self.metric, N_atoms=self.N_atoms)
                val = msd_val * len(indices)

            else:
                raise ValueError(f"Unsupported split type: {self.split}")

            if val > best_score:
                best_score = val
                top_cluster = i

        return top_cluster

    def split_selected_cluster(self, clusters, cluster_to_split):
        """Split the specified cluster into two subclusters.

        Parameters
        ----------
        clusters : list of ndarray
            Current list of clusters.
        cluster_to_split : int
            Index of the cluster to split.

        Returns
        -------
        list of ndarray
            Two new subclusters.
        """
        cluster_indices = clusters[cluster_to_split]
        if len(cluster_indices) < 2:
            warnings.warn(f"{self.anchors} cannot split a cluster with less than 2 points")
            return [cluster_indices]
        
        subdata = self.data[cluster_indices]

        if self.anchors == 'nani':
            nani = KmeansNANI(data=subdata, n_clusters=2, metric=self.metric,
                              N_atoms=self.N_atoms, init_type=self.init_type, percentage=self.percentage)
            initiators = nani.initiate_kmeans()
            labels, centers, _ = nani.kmeans_clustering(initiators)

            clusterA = cluster_indices[labels == 0]
            clusterB = cluster_indices[labels == 1]
            return [clusterA, clusterB]

        if self.anchors == 'outlier_pair':
            idx_outlier = calculate_outlier(subdata, metric=self.metric, N_atoms=self.N_atoms)
            anchorA = subdata[idx_outlier]

            dists = np.sum((subdata - anchorA) ** 2, axis=1) / self.N_atoms
            idx_furthest = np.argmax(dists)
            anchorB = subdata[idx_furthest]

            data_c = self.data[cluster_indices]
            dA = np.sum((data_c - anchorA) ** 2, axis=1) / self.N_atoms
            dB = np.sum((data_c - anchorB) ** 2, axis=1) / self.N_atoms
            initial_mask = dA < dB

            if self.refine:
                groupA = subdata[initial_mask]
                groupB = subdata[~initial_mask]

                medoidA = groupA[0] if len(groupA) <= 2 else groupA[calculate_medoid(groupA, self.metric, self.N_atoms)]
                medoidB = groupB[0] if len(groupB) <= 2 else groupB[calculate_medoid(groupB, self.metric, self.N_atoms)]

                init_centers = np.vstack([medoidA, medoidB]).astype(np.float64)

                kmeans = KMeans(n_clusters=2, init=init_centers, n_init=1, max_iter=300, algorithm="elkan")
                kmeans.fit(subdata)
                final_labels = kmeans.labels_

                if len(np.unique(final_labels)) < 2:
                    warnings.warn("K-Means refinement failed to find two distinct clusters; returning original clusters.")
                    
                    clusterA = cluster_indices[initial_mask]
                    clusterB = cluster_indices[~initial_mask]
                    return [clusterA, clusterB]

                clusterA = cluster_indices[final_labels == 0]
                clusterB = cluster_indices[final_labels == 1]

            else:
                clusterA = cluster_indices[initial_mask]
                clusterB = cluster_indices[~initial_mask]
                return [clusterA, clusterB]

            return [clusterA, clusterB]
        
        if self.anchors == 'splinter_split':
            if len(subdata) < 2:
                return [cluster_indices]

            splinter_idx = calculate_outlier(subdata, self.metric, self.N_atoms)
            splinter_point = subdata[splinter_idx]

            medoid_idx = calculate_medoid(subdata, self.metric, self.N_atoms)
            medoid_point = subdata[medoid_idx]

            splinter_group = [splinter_idx]
            main_group = []

            for i in range(len(subdata)):
                if i == splinter_idx:
                    continue

                dS = np.sum((subdata[i] - splinter_point) ** 2) / self.N_atoms
                dM = np.sum((subdata[i] - medoid_point) ** 2) / self.N_atoms

                if dS < dM:
                    splinter_group.append(i)
                else:
                    main_group.append(i)

            if self.refine:
                groupA = subdata[main_group]
                groupB = subdata[splinter_group]

                medoid_0 = groupA[0] if len(groupA) <= 2 else groupA[calculate_medoid(groupA, self.metric, self.N_atoms)]
                medoid_1 = groupB[0] if len(groupB) <= 2 else groupB[calculate_medoid(groupB, self.metric, self.N_atoms)]

                init_centers = np.vstack([medoid_0, medoid_1]).astype(np.float64)

                # Step 5: KMeans refinement
                kmeans = KMeans(n_clusters=2, init=init_centers, n_init=1, max_iter=300, algorithm="elkan")
                kmeans.fit(subdata)
                final_labels = kmeans.labels_

                if len(np.unique(final_labels)) < 2:
                    warnings.warn("K-Means refinement failed to find two distinct clusters; returning original clusters.")
                    
                    clusterA = cluster_indices[main_group]
                    clusterB = cluster_indices[splinter_group]
                    return [clusterA, clusterB]

                clusterA = cluster_indices[final_labels == 0]
                clusterB = cluster_indices[final_labels == 1]

            else:
                clusterA = cluster_indices[main_group]
                clusterB = cluster_indices[splinter_group]
                return [clusterA, clusterB]

            return [clusterA, clusterB]

        else:
            raise ValueError(f"anchor must be 'nani', 'outlier_pair' or 'splinter_split'. Got {self.anchors}")

    def _create_label_array(self, list_of_clusters, n_total):
        """Create label array from cluster indices.

        Parameters
        ----------
        list_of_clusters : list of ndarray
            Cluster membership lists.
        n_total : int
            Total number of data points.

        Returns
        -------
        ndarray
            Array of labels with shape (n_total,).
        """

        labels = np.full(n_total, -1, dtype=int)
        for idx, inds in enumerate(list_of_clusters):
            if inds.size > 0:
                labels[inds] = idx

        return labels

    def compute_scores(self, labels):
        """Compute clustering quality scores.

        Parameters
        ----------
        labels : ndarray
            Cluster labels for each data point.

        Returns
        -------
        tuple of float
            Calinski-Harabasz score and Davies-Bouldin score.
        """

        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1 or len(unique_labels) >= len(self.data):
            ch_score = 0
            db_score = 0
        else:
            ch_score = calinski_harabasz_score(self.data, labels)
            db_score = davies_bouldin_score(self.data, labels)
        return ch_score, db_score