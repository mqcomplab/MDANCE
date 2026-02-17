from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform, cdist
from mdance.tools.bts import refine_dis_matrix, calculate_medoid
from mdance.cluster.nani import KmeansNANI


class PRISM:
    """
    Pathway Representation via Intrinsic Structural Medoids (PRISM).

    PRISM is a state-aware pathway clustering framework that summarizes each
    pathway (trajectory) by a compact set of representative conformations
    ("structural medoids") and then clusters pathways using hierarchical
    agglomerative clustering (HAC) over a pathway-pathway dissimilarity matrix.

    Overview
    --------
    Given an ensemble of pathways, PRISM proceeds as:

    1  Representative-set construction (medoid sets):
       Each pathway is mapped to a finite set of representative structures
       (medoids) obtained from k-means NANI clustering in the chosen feature
       space. Three construction strategies are supported:

       - Option 1 (global medoid sharing):
         Pool all frames across all pathways, cluster into `k` clusters, compute
         one medoid per cluster, then assign to each pathway the subset of
         global medoids whose clusters contain at least one frame from that
         pathway.

       - Option 2 (independent per-pathway medoids):
         Cluster each pathway independently into `k` clusters and compute its
         medoids. Representatives are pathway-specific (no enforced sharing).

       - Option 3 (two-stage refinement):
         First perform Option 2 to obtain local medoids per pathway, pool all
         local medoids, cluster into `k_final` clusters, compute refined global
         medoids, then assign each pathway the refined medoids whose clusters
         contain at least one of its local medoids.

    2  Pathway dissimilarity (weighted average Hausdorff):
       For two pathways represented by sets A and B (their medoid sets), PRISM
       computes a symmetric dissimilarity based on the mean nearest-neighbor
       mismatch in both directions (robust to outliers compared to classical
       Hausdorff). The implementation supports several weighting/normalization
       schemes.

    3  Hierarchical clustering:
       The resulting dissimilarity matrix is converted to condensed form and
       clustered with SciPy linkage; flat clusters can be extracted via
       `fcluster` using a user-provided criterion.

    Parameters
    ----------
    trajs : list[tuple[Hashable, np.ndarray]]
        Sequence of (trajectory_id, frames) pairs. Each `frames` array must be
        shape (n_frames, n_features) in a common feature space.
    metric : str
        Frame-level metric identifier used by the underlying MDANCE routines
        (e.g., for medoid computation / NANI clustering).
    t : float | int
        Threshold parameter forwarded to `scipy.cluster.hierarchy.fcluster`.
        Interpretation depends on `criterion` (e.g., `maxclust` uses `t` as the
        requested maximum number of clusters).
    criterion : str
        Criterion forwarded to `scipy.cluster.hierarchy.fcluster` (e.g.,
        'maxclust', 'distance', 'inconsistent', ...).
    link : str, default='ward'
        Linkage method forwarded to `scipy.cluster.hierarchy.linkage`.
        Common choices: 'ward', 'average', 'complete', ...
    N_atoms : int, optional
        Number of atoms for MD-based metrics (default 1 for non-MD feature
        vectors).
    option : int, optional
        Representative-set construction option in {1, 2, 3}. See Overview.
    k : int, optional
        Number of clusters used when building representative medoids (global or
        per-pathway stage, depending on `option`).
    k_final : int, optional
        Second-stage cluster count used only for `option==3`.
    percentage : int, optional
        Percentage parameter forwarded to k-means NANI initialization/execution
        (MDANCE-specific).
    weight_scheme : str, optional
        Scheme controlling the weighting/normalization used by the weighted
        average Hausdorff calculation. Supported values are:
        {'unnormalized', 'weighted_avg', 'sym_avg', 'product_avg'}.

    .. _Linkage Methods: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    
    .. _fcluster: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
        
    """

    def __init__(self, trajs, metric, t, criterion, link='ward', **kwargs):
        self.trajs = trajs
        self.metric = metric
        
        self.t = t
        self.criterion = criterion
        self.link = link

        self.pathways = {}
        self.N_atoms = int(kwargs.get('N_atoms', 1))
        self.option = int(kwargs.get('option', 1))
        self.k = int(kwargs.get('k', 10))
        self.k_final = int(kwargs.get('k_final', 25))
        self.percentage = int(kwargs.get('percentage', 100))
        self.weight_scheme = kwargs.get('weight_scheme', 'weighted_avg')

        self._validate_inputs()
     
           
    def _validate_inputs(self):
        """Validate init args and raw trajs before processing."""
        if self.trajs is None or not isinstance(self.trajs, (list, tuple)) or len(self.trajs) == 0:
            raise ValueError("`trajs` must be a non-empty list of (unique_id, frames_array).")

        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("`metric` must be a non-empty string (e.g., 'MSD').")

        if self.option not in (1, 2, 3):
            raise ValueError(f"`option` must be 1, 2, or 3. Got {self.option}.")

        if self.k <= 0:
            raise ValueError(f"`k` must be > 0. Got {self.k}.")

        if self.option == 3 and self.k_final <= 0:
            raise ValueError(f"`k_final` must be > 0 when option==3. Got {self.k_final}.")

        if self.weight_scheme not in {"unnormalized", "weighted_avg", "sym_avg", "product_avg"}:
            raise ValueError(
                "Invalid `weight_scheme`. Must be one of: "
                "{'unnormalized','weighted_avg','sym_avg','product_avg'}; "
                f"got {self.weight_scheme!r}."
            )

    def process_trajs(self):
        """
        Build the internal pathway dictionary.        
        
        Converts the input list of (trajectory_id, frames) into `self.pathways`,
        mapping each trajectory identifier to its frame array.

        Returns
        -------
        pathways : dict
            Dictionary containing the sampled trajectories
        """

        for traj_idx, traj in self.trajs:
            try:
                traj_idx = int(traj_idx)
            except Exception:
                raise ValueError(
                    f"Trajectory ID {traj_idx!r} is not numeric. "
                    "Ensure trajectory filenames map to numeric IDs."
                )

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
        distances = []

        order = sorted(self.pathways.keys())
        traj_list = [self.pathways[k] for k in order]
        index_pos = {k: i for i, k in enumerate(order)}

        medoids_per_traj = self._build_ave_haus_medoids(traj_list)
        
        for _ in self.pathways.keys():
            distances.append([None] * len(self.pathways))
        
        keys = list(self.pathways.keys())

        for k in sorted(self.pathways):
            arr = self.pathways[k]
            print(f"{k}: len={len(arr)}, shape={getattr(arr, 'shape', ('?'))}")

        for i_idx, i in enumerate(keys):
            distances[i_idx][i_idx] = 0
            for j_idx, j in enumerate(keys):
                if j_idx <= i_idx:
                    continue

                else:
                    Ai = medoids_per_traj[index_pos[i]]
                    Bj = medoids_per_traj[index_pos[j]]
                    d = np.inf if (len(Ai) == 0 or len(Bj) == 0) else self._average_hausdorff(Ai, Bj, self.weight_scheme)
                    distances[i_idx][j_idx] = d
                    distances[j_idx][i_idx] = d

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

    def labels(self, condensed=True):
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
        label_traj_idx = [np.array(list(self.pathways.keys()))[indices] for indices in label_indices]

        if condensed:
            custom_labels = [f"{self.group_consecutive_indices(cluster)}" for cluster in label_traj_idx]
        else:
            custom_labels = label_traj_idx
            
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

    def _average_hausdorff(self, A, B, scheme=None):
        """
        Weighted average Hausdorff dissimilarity between two representative sets.

        Given two finite representative sets A and B compute a symmetric dissimilarity 
        based on mean nearest-neighbor distances in both directions:

            d(A->B) = sum_{a in A} min_{b in B} ||a - b||
            d(B->A) = sum_{b in B} min_{a in A} ||b - a||

        The final dissimilarity is a weighted combination:

            d(A, B) = w_A * d(A->B) + w_B * d(B->A)

        Parameters
        ----------
        A, B : np.ndarray
            2D arrays of shape (n_points, n_features) representing the medoid sets.
        scheme : str, optional
            Normalization scheme:
            * 'unnormalized' : w_A = 1, w_B = 1
            * 'weighted_avg' : w_A = 1/(2|A|), w_B = 1/(2|B|)
            * 'sym_avg'      : w_A = w_B = 1/(|A| + |B|)
            * 'product_avg'  : w_A = w_B = 1/(|A||B|)

        Returns
        -------
        float
            Weighted average Hausdorff dissimilarity.

        """
        A, B = np.asarray(A, float), np.asarray(B, float)

        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
            raise ValueError("A and B must be 2D arrays with same number of columns")
        
        nA, nB = len(A), len(B)
        if nA == 0 or nB == 0:
            return np.inf
        
        D2 = cdist(A, B, metric="sqeuclidean")
        dAB = np.sqrt(np.min(D2, axis=1)).sum()
        dBA = np.sqrt(np.min(D2, axis=0)).sum()

        if scheme == "unnormalized":
            wA, wB = 1.0, 1.0
        elif scheme == "weighted_avg":
            wA, wB = 1.0/(2*nA), 1.0/(2*nB)
        elif scheme == "sym_avg":
            wA = wB = 1.0/(nA + nB)
        elif scheme == "product_avg":
            wA = wB = 1.0/(nA*nB)

        else:
            raise ValueError("scheme must be one of: unnormalized, weighted_avg, sym_avg, product_avg")
        return wA * dAB + wB * dBA

    def _cluster_medoids_nonempty(self, data, labels, k):
        """Return medoids for non-empty clusters only (list of 1D points)."""
        meds = []
        for i in range(k):
            pts = data[labels == i]
            if len(pts) == 0:
                continue
            if len(pts) == 1:
                meds.append(pts[0])
            else:
                m_idx = calculate_medoid(pts, self.metric)
                meds.append(pts[m_idx])
        return meds

    def _build_ave_haus_medoids(self, traj_list):
        """
        Construct representative medoid sets for each pathway (PRISM Options 1–3).

        Parameters
        ----------
        traj_list : list[np.ndarray]
            List of pathway frame arrays, each of shape (n_frames_i, n_features).

        Returns
        -------
        list[np.ndarray]
            A list `reps` where `reps[i]` is the representative set for the i-th
            pathway in `traj_list`, as a 2D array of shape (n_medoids_i, n_features).

        """

        if len(traj_list) < 2:
            return [np.empty((0, traj_list[0].shape[1]))]
        feat_dims = {t.shape[1] for t in traj_list}
        if len(feat_dims) != 1:
            raise ValueError("All trajectories must have the same number of features (columns).")

        k = self.k
        pct = self.percentage
        opt = self.option
        k_final = self.k_final
        feat_dim = traj_list[0].shape[1]

        # Option 1: cluster all frames; assign cluster medoids to trajectories that contributed frames
        if opt == 1:
            combined = np.vstack(traj_list)
            kmeans = KmeansNANI(data=combined, n_clusters=k, metric=self.metric, N_atoms=self.N_atoms, init_type='quota', percentage=pct)
            labels, _, _ = kmeans.execute_kmeans_all()

            cluster_meds = self._cluster_medoids_nonempty(combined, labels, k)

            # stacked trajectory boundaries
            lens = [len(t) for t in traj_list]
            starts = np.cumsum([0] + lens[:-1])
            ends = np.cumsum(lens)

            per_traj_meds = [[] for _ in traj_list]
            kept_ids = []
            for i in range(k):
                if np.any(labels == i):
                    kept_ids.append(i)

            #assign medoids to trajectories that contributed frames to that cluster, using trajectory boundaries created above
            for cmed, ci in zip(cluster_meds, kept_ids):
                for t_id, (s, e) in enumerate(zip(starts, ends)):
                    if np.any(labels[s:e] == ci):
                        per_traj_meds[t_id].append(cmed)

            return [np.asarray(m, dtype=float) if len(m) else np.empty((0, feat_dim)) for m in per_traj_meds]

        # Option 2: Per trajectory clustering; medoids are per trajectory only
        if opt == 2:
            per_traj_meds = []

            for T in traj_list:
                arr = np.asarray(T, dtype=float)
                n = arr.shape[0]
                if n <= k:
                    per_traj_meds.append(arr)
                    continue

                kmeans = KmeansNANI(data=T, n_clusters=k, metric=self.metric,N_atoms=self.N_atoms, init_type='quota', percentage=pct)
                labels, _, _ = kmeans.execute_kmeans_all()
                meds = self._cluster_medoids_nonempty(T, labels, k)
                per_traj_meds.append(np.asarray(meds, dtype=float) if len(meds) else np.empty((0, feat_dim)))
            return per_traj_meds

        # Option 3: Per trajectory clustering, then re-cluster all medoids and assign final medoids to trajectories that contributed medoids
        if opt == 3:
            per_traj_stage1 = []
            counts = []

            for T in traj_list:
                arr = np.asarray(T, dtype=float)
                n = arr.shape[0]
                if n <= k:
                    per_traj_stage1.append(arr)
                    counts.append(n)
                    continue

                kmeans = KmeansNANI(data=T, n_clusters=k, metric=self.metric,N_atoms=self.N_atoms, init_type='quota', percentage=pct)
                labels, _, _ = kmeans.execute_kmeans_all()
                meds = self._cluster_medoids_nonempty(T, labels, k)
                arr = np.asarray(meds, dtype=float) if len(meds) else np.empty((0, feat_dim))
                per_traj_stage1.append(arr)
                counts.append(len(arr))

            if sum(counts) == 0:
                return [np.empty((0, feat_dim)) for _ in traj_list]

            combined_meds = np.vstack([m for m in per_traj_stage1 if len(m)])
            starts = np.cumsum([0] + counts[:-1])
            ends = np.cumsum(counts)

            # Clustering of all medoids
            kmeans2 = KmeansNANI(data=combined_meds, n_clusters=self.k_final, metric=self.metric,N_atoms=self.N_atoms, init_type='quota', percentage=100)
            labels2, _, _ = kmeans2.execute_kmeans_all()

            # For each final cluster, compute cluster medoid and assign to trajectories that contributed any medoid to that cluster
            final_per_traj = [[] for _ in traj_list]
            unique_final = np.unique(labels2)
            for ci in unique_final:
                idx = np.where(labels2 == ci)[0]
                if idx.size == 0:
                    continue
                pts = combined_meds[idx]
                if len(pts) == 1:
                    cmed = pts[0]
                else:
                    cm_idx = calculate_medoid(pts, self.metric)
                    cmed = pts[cm_idx]

                # Did trajectory t contribute any medoid to this final cluster?
                for t_id, (s, e) in enumerate(zip(starts, ends)):
                    if e - s == 0:
                        continue
                    if np.any((idx >= s) & (idx < e)):
                        final_per_traj[t_id].append(cmed)

            return [np.asarray(m, dtype=float) if len(m) else np.empty((0, feat_dim)) for m in final_per_traj]

        raise ValueError("Option must be 1, 2, or 3")
