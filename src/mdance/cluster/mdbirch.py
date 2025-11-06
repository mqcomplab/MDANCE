# mdBIRCH is an open-source clustering module based on n-ary comparisons
#
# Please, cite the BitBIRCH paper: https://doi.org/10.1039/D5DD00030K
#
# mdBIRCH is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# mdBIRCH is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# mdBIRCH License: GPL-3.0 https://www.gnu.org/licenses/gpl-3.0.en.html
#
# mdBIRCH authors: Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
#                  Lexin Chen <le.chen@ufl.edu>
#                  Jherome Brylle Woody Santos <ja.santos@ufl.edu>
#
### Part of the tree-management code was derived from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
### Authors: Manoj Kumar <manojkumarsivaraj334@gmail.com>
###          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
###          Joel Nothman <joel.nothman@gmail.com>
### License: BSD 3 clause
# Parts of the BitBIRCH algorithm were previously released under the LGPL-3.0 license by: 
# Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# Vicky (Vic) Jung <jungvicky@ufl.edu>
# Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# Kate Huddleston <kdavis2@chem.ufl.edu>

import numpy as np
from scipy import sparse

def set_merge(merge_criterion, features=None):
    """
    Sets merge_accept function for merge_subcluster, based on user specified merge_criteria. 

    Radius: merge subcluster if the post-merge average distance from centroid to points in cluster is less than threshold

    Parameters:
    -----------
    merge_criterion : str
                        merge criterion to use. Currently only 'radius' is supported
    features : int
                        number of features in the data. 

    Returns:
    --------
    merge_accept : function 
                        function that determines if cluster is accepted to merge based on the specified criteria
    """

    if merge_criterion == 'radius':
        D = features if features is not None else 1
        def merge_accept(threshold, new_ls, new_ss, new_n):
            radius_sq = np.sum(new_ss)/new_n - np.dot(new_ls, new_ls)/new_n**2
            return radius_sq <= threshold**2 / 4 * D
    else:
        raise ValueError(f"Unsupported merge criterion: '{merge_criterion}'. Currently only 'radius' is supported.")

    globals()['merge_accept'] = merge_accept

def msd_condensed(c_sum, sq_sum, N):
    """Condensed version of Mean Square Deviation (MSD) calculation 
    for *n*-ary objects. 

    Parameters
    ----------
    c_sum : array-like of shape (n_features,)
        A feature array of the column-wsie sum of the data. 
    sq_sum : array-like of shape (n_features,)
        A feature array of the column-wise sum of the squared data. 
    N : int
        Number of data points.
    
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
    return msd

def max_separation(X):
    """Finds two objects in X that are very separated
    This is an approximation (not guaranteed to find
    the two absolutely most separated objects), but it is
    a very robust O(N) implementation. Quality of clustering
    does not diminish in the end.
    
    Algorithm:
    a) Find centroid of X
    b) mol1 is the molecule most distant from the centroid
    c) mol2 is the molecule most distant from mol1
    
    Returns
    -------
    (mol1, mol2) : (int, int)
                   indices of mol1 and mol2
    1 - sims_mol1 : np.ndarray
                   Distances to mol1
    1 - sims_mol2: np.ndarray
                   Distances to mol2
    These are needed for node1_dist and node2_dist in _split_node
    """
    # Get the centroid of the set
    n_samples = len(X)
    linear_sum = np.sum(X, axis = 0)
    centroid = calc_centroid(linear_sum, n_samples)

    # Get the similarity of each molecule to the centroid
    matrix_row_norm = np.sum(X**2, axis = 1)
    a_centroid = np.dot(X, centroid)
    sims_med = np.dot(centroid, centroid) + matrix_row_norm - 2 * a_centroid

    # Get the least similar molecule to the centroid
    mol1 = np.argmax(sims_med)

    # Get the similarity of each molecule to mol1
    a_mol1 = np.dot(X, X[mol1])
    sims_mol1 = np.dot(X[mol1], X[mol1]) + matrix_row_norm - 2 * a_mol1

    # Get the least similar molecule to mol1
    mol2 = np.argmax(sims_mol1)

    # Get the similarity of each molecule to mol2
    a_mol2 = np.dot(X, X[mol2])
    sims_mol2 = np.dot(X[mol2], X[mol2]) + matrix_row_norm - 2 * a_mol2
    
    return (mol1, mol2), sims_mol1, sims_mol2

def calc_centroid(linear_sum, n_samples):
    """Calculates centroid
    
    Parameters
    ----------
    
    linear_sum : np.ndarray
                 Sum of the elements column-wise
    n_samples : int
                Number of samples
                
    Returns
    -------
    centroid : np.ndarray
               Centroid fingerprints of the given set
    """
    return linear_sum/n_samples

def _iterate_sparse_X(X):
    """This little hack returns a densified row when iterating over a sparse
    matrix, instead of constructing a sparse matrix for every row that is
    expensive.
    """
    n_samples, n_features = X.shape
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr

    for i in range(n_samples):
        row = np.zeros(n_features)
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        row[nonzero_indices] = X_data[startptr:endptr]
        yield row

def _split_node(node, threshold, branching_factor):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. Two empty nodes and two empty subclusters are initialized.
    2. The pair of distant subclusters are found.
    3. The properties of the empty subclusters and nodes are updated
       according to the nearest distance between the subclusters to the
       pair of distant subclusters.
    4. The two nodes are set as children to the two subclusters.
    """
    new_subcluster1 = _BFSubcluster()
    new_subcluster2 = _BFSubcluster()
    new_node1 = _BFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    new_node2 = _BFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2  
    
    # O(N) implementation of max separation
    farthest_idx, node1_dist, node2_dist = max_separation(node.centroids_)    
    # Notice that max_separation is returning similarities and not distances
    node1_closer = node1_dist < node2_dist
    # Make sure node1 is closest to itself even if all distances are equal.
    # This can only happen when all node.centroids_ are duplicates leading to all
    # distances between centroids being zero.
    node1_closer[farthest_idx[0]] = True

    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2


class _BFNode:
    """Each node in a BFTree is called a BFNode.

    The BFNode can have a maximum of branching_factor
    number of BFSubclusters.

    Parameters
    ----------
    threshold : float
        Threshold needed for a new subcluster to enter a BFSubcluster.

    branching_factor : int
        Maximum number of BF subclusters in each node.

    is_leaf : bool
        We need to know if the BFNode is a leaf or not, in order to
        retrieve the final subclusters.

    n_features : int
        The number of features.

    Attributes
    ----------
    subclusters_ : list
        List of subclusters for a particular BFNode.

    prev_leaf_ : _BFNode
        Useful only if is_leaf is True.

    next_leaf_ : _BFNode
        next_leaf. Useful only if is_leaf is True.
        the final subclusters.

    init_centroids_ : ndarray of shape (branching_factor + 1, n_features)
        Manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .

    centroids_ : ndarray of shape (branching_factor + 1, n_features)
        View of ``init_centroids_``.

    """

    def __init__(self, *, threshold, branching_factor, is_leaf, n_features, dtype):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features

        # The list of subclusters, centroids and squared norms
        # to manipulate throughout.
        self.subclusters_ = []
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features), dtype=dtype)
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        
        # Keep centroids as views. In this way
        # if we change init_centroids, it is sufficient
        self.centroids_ = self.init_centroids_[: n_samples + 1, :]
        
    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """
        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.centroids_[ind] = new_subcluster1.centroid_
        self.append_subcluster(new_subcluster2)

    def insert_bf_subcluster(self, subcluster):
        """Insert a new subcluster into the node."""
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        threshold = self.threshold
        branching_factor = self.branching_factor
        # We need to find the closest subcluster among all the
        # subclusters so that we can insert our new subcluster.
        a = np.dot(self.centroids_, subcluster.centroid_)
        sim_matrix = np.dot(subcluster.centroid_, subcluster.centroid_) + np.sum(self.centroids_**2, axis = 1) - 2 * np.dot(self.centroids_, subcluster.centroid_)
        closest_index = np.argmin(sim_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        # If the subcluster has a child, we need a recursive strategy.
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_bf_subcluster(subcluster)

            if not split_child:
                # If it is determined that the child need not be split, we
                # can just update the closest_subcluster
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                self.centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                return False

            # things not too good. we need to redistribute the subclusters in
            # our child node, and add a new subcluster in the parent
            # subcluster to accommodate the new child.
            else:
                new_subcluster1, new_subcluster2 = _split_node(
                    closest_subcluster.child_,
                    threshold,
                    branching_factor
                )
                self.update_split_subclusters(
                    closest_subcluster, new_subcluster1, new_subcluster2
                )

                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        # good to go!
        else:
            merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
            if merged:
                self.centroids_[closest_index] = closest_subcluster.centroid_
                self.init_centroids_[closest_index] = closest_subcluster.centroid_
                return False

            # not close to any other subclusters, and we still
            # have space, so add.
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False

            # We do not have enough space nor is it closer to an
            # other subcluster. We need to split.
            else:
                self.append_subcluster(subcluster)
                return True


class _BFSubcluster:
    """Each subcluster in a BFNode is called a BFSubcluster.

    A BFSubcluster can have a BFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray of shape (n_features,), default=None
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    centroid_ : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``BFNode.centroids_`` is called.
    
    mol_indices : list, default=[]
        List of indices of molecules included in the given cluster.

    child_ : _BFNode
        Child Node of the subcluster. Once a given _BFNode is set as the child
        of the _BFNode, it is set to ``self.child_``.
    """

    def __init__(self, *, linear_sum = None, mol_indices = []):
        if linear_sum is None:
            self.n_samples_ = 0
            self.centroid_ = self.linear_sum_ = 0
            self.mol_indices = []
            self.sq_sum = 0
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.mol_indices = mol_indices
            self.sq_sum = self.centroid_**2
        
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.mol_indices += subcluster.mol_indices
        self.centroid_ = calc_centroid(self.linear_sum_, self.n_samples_)
        self.sq_sum += subcluster.sq_sum

    def merge_subcluster(self, nominee_cluster, threshold):
        """Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_ss = self.sq_sum + nominee_cluster.sq_sum
        
        if merge_accept(threshold, new_ls, new_ss, new_n):
            new_centroid = calc_centroid(new_ls, new_n)
            (
                self.n_samples_,
                self.linear_sum_,
                self.centroid_,
                self.mol_indices,
                self.sq_sum
            ) = (new_n, new_ls, new_centroid, self.mol_indices + nominee_cluster.mol_indices, new_ss)
            return True
        return False
    

class mdBirch():
    """Implements the mdBIRCH clustering algorithm for online clustering of molecular trajectories.
    
    mdBIRCH combines a BIRCH CF-tree with RMSD-calibrated merge decisions to provide
    simple and physically interpretable clustering. The algorithm performs online clustering
    in a single pass, maintaining a clear guarantee that the average distance to the
    centroid of each cluster remains within the chosen tolerance throughout the process.
    
    For each new frame insertion, the algorithm evaluates the candidate cluster after
    hypothetical inclusion of the new frame. If the merge satisfies the bound implied
    by the threshold, the frame is accepted and the clustering features are updated.
    This approach is memory-bounded, fast, and enables efficient assignment of frames
    in an incremental operation with clusters that are easy to explain.
    
    Parameters
    ----------
    threshold : float, default=1.0
        The RMSD tolerance threshold for cluster membership. This serves as an intuitive
        control for granularity: increasing threshold reduces the number of clusters,
        concentrates coverage in the largest states, and broadens within-cluster RMSD
        distributions. The threshold represents the actual physical tolerance and provides
        interpretable choices for clustering resolution.

    branching_factor : int, default=50
        Maximum number of subclusters in each node. When a new sample would cause
        a node to exceed this limit, the node is split into two nodes with subclusters
        redistributed between them. The parent subcluster is removed and two new
        subclusters are added as parents of the split nodes.

    Attributes
    ----------
    root_ : _BFNode
        Root of the BF Tree.

    dummy_leaf_ : _BFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.

    index_tracker : int
        Counter tracking the current frame/molecule index.

    first_call : bool
        Flag indicating if this is the first call to fit.

    Notes
    -----
    The tree data structure consists of nodes with each node containing a number of
    subclusters. The maximum number of subclusters in a node is determined by the
    branching factor. Each subcluster maintains a linear sum, squared sum, molecule
    indices, and the number of samples in that subcluster. Additionally, each
    subcluster can have a child node if it is not a member of a leaf node.

    For a new point entering the root, it is merged with the closest subcluster
    and the linear sum, molecule indices, and sample count are updated. This process
    continues recursively until the properties of the leaf node are updated.

    """


    def __init__(
        self,
        *,
        threshold=1,
        branching_factor=50,
    ):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.index_tracker = 0
        self.first_call = True

    def fit(self, X):
        """
        Build a BF Tree for the input data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self
            Fitted estimator.
        """

        # TODO: Add input verification

        return self._fit(X)

    def _fit(self, X):
        threshold = self.threshold
        branching_factor = self.branching_factor

        n_features = X.shape[1]
        d_type = X.dtype

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        if self.first_call:
            # The first root is the leaf. Manipulate this object throughout.
            self.root_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
    
            # To enable getting back subclusters.
            self.dummy_leaf_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        # Cannot vectorize. Enough to convince to use cython.
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        for sample in iter_func(X):
            #set_bits = np.sum(sample)
            subcluster = _BFSubcluster(linear_sum=sample.copy(), mol_indices = [self.index_tracker])
            split = self.root_.insert_bf_subcluster(subcluster)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor
                )
                del self.root_
                self.root_ = _BFNode(
                    threshold=threshold,
                    branching_factor=branching_factor,
                    is_leaf=False,
                    n_features=n_features,
                    dtype=d_type,
                )
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)
            self.index_tracker += 1

        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids
        self._n_features_out = self.subcluster_centers_.shape[0]
        
        self.first_call = False
        return self

    def _get_leaves(self):
        """
        Retrieve the leaves of the BF Node.

        Returns
        -------
        leaves : list of shape (n_leaves,)
            List of the leaf nodes.
        """
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves
    
    def get_centroids(self):
        """Method to return a list of Numpy arrays containing the centroids' fingerprints"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        centroids = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                centroids.append(subcluster.centroid_)

        return centroids
    
    def get_cluster_mol_ids(self):
        """Method to return the indices of molecules in each cluster"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        clusters_mol_id = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                clusters_mol_id.append(subcluster.mol_indices)

        return clusters_mol_id

