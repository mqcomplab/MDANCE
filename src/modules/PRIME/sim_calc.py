from src.tools.esim_modules import SimilarityIndex, calc_medoid, calc_outlier
import numpy as np
import re
import json
import glob

class FrameSimilarity:
    """A class to calculate the similarity between clusters.
    
    Attributes:
        c0 (numpy.ndarray): The dominant cluster.
        input_files (list): The list of cluster files.
        summary_file (str): The path to the summary file.
        n_clusters (int): The number of clusters to analyze.
        weighted_by_frames (bool): Whether to weight the similarity values by the number of frames in the cluster.
        n_ary (str): The n_ary similarity metric to use.
        weight (str): The weight to use for the similarity metric.
    
    Methods:
        calculate_pairwise: Calculates the similarity between the dominant cluster and all other clusters.
        calculate_union: Calculates the similarity between the dominant cluster and the union of all other clusters.
        _perform_calculation: Auxiliary function to calculate the similarity between the dominant cluster and a single cluster.
        calculate_medoid: Calculates the similarity between the dominant cluster and the cluster with the lowest average distance to the dominant cluster.
        calculate_outliers: Calculates the similarity between the dominant cluster and the cluster with the highest average distance to the dominant cluster.
    """
    
    def __init__(self, cluster_folder=None, summary_file=None, trim_frac=None, n_clusters=None, 
                 weighted_by_frames=True, n_ary='RR', weight='nw'):
        """Initializes instances for the FrameSimilarity class.
        
        Args:
            cluster_folder (str): The path to the folder containing the normalized 
                cluster files.
            summary_file (str): The path to the summary file containing the number 
                of frames for each cluster (CPPTRAJ clustering output).
            trim_frac (float): The fraction of outliers to trim from the top cluster.
            n_clusters (int): The number of clusters to analyze, None for all clusters.
            weighted_by_frames (bool): Whether to weight similarity values by the 
                number of frames.
            n_ary (str): The similarity metric to use for comparing clusters. 
            weight (str): The weighting scheme to use for comparing clusters.

        Returns:
            None.
            
        Notes:
            Options for `n_ary` and `weight` under `esim.py`.
        """
        self.c0 = np.load(f"{cluster_folder}/normed_clusttraj.c0.npy")
        if trim_frac:
            self.c0 = _trim_outliers(self.c0, trim_frac=trim_frac, n_ary=n_ary, weight=weight)
        self.input_files = sorted(glob.glob(f"{cluster_folder}/normed_clusttraj.c*"), key=lambda x: int(re.findall("\d+", x)[0]))[1:]
        self.summary_file = summary_file
        self.n_clusters = n_clusters
        self.weighted_by_frames = weighted_by_frames
        self.n_ary = n_ary
        self.weight = weight
        self.sims = {}
    
    def calculate_pairwise(self):
        """Calculates pairwise similarity between each cluster and all other clusters.

        Notes:
            For each cluster file, loads the data and calculates the similarity score 
                with the top (c0) cluster.
            The similarity score is calculated as the average of pairwise similarity 
                values between each frame in the cluster and the top c0 cluster.
            The esim index used is defined by the `n_ary` parameter.
        
        Returns:
            If `frame_weighted_sim` returns `False`, 
                nw_dict (dict): unweighted average similarity values.
            If `frame_weighted_sim` returns `True`, 
                w_dict (dict): calls `weight_dict` function to weight similarity values.
        """
        for each, file in enumerate(self.input_files):
            ck = np.load(file)
            self.sims[each] = {}
            for i, x in enumerate(self.c0):
                total = 0
                for j, y in enumerate(ck):
                    c_total = np.sum(np.array([x, y]), axis=0)
                    pair_sim = SimilarityIndex(c_total, 2, n_ary=self.n_ary, weight=self.weight, 
                                               c_threshold=None, w_factor="fraction")()
                    total += pair_sim
                avg = total / len(ck)
                if f"f{i}" not in self.sims[each]:
                    self.sims[each][f"f{i}"] = []
                self.sims[each][f"f{i}"] = avg

        nw_dict = _format_dict(self.sims)
        if not self.weighted_by_frames:
            return nw_dict
        elif self.weighted_by_frames:
            return weight_dict(file_path=None, summary_file=self.summary_file, dict=nw_dict, n_clusters=self.n_clusters)

    def calculate_union(self):
        """ Calculates the extended similarity between the union of frame in c0 and cluster k.

        Notes:
            For each cluster file, loads the data and calculates the extended similarity.
            The similarity score is calculated as the union similarity between 
                all frames in the cluster and the top c0 cluster.
            The esim index used is defined by the `n_ary` parameter.
        
        Returns:
            If `frame_weighted_sim` returns `False`, 
                nw_dict (dict): unweighted average similarity values.
            If `frame_weighted_sim` returns `True`, 
                w_dict (dict): calls `weight_dict` function to weight similarity values.
        """
        for each, file in enumerate(self.input_files):
            ck = np.load(file)
            self.sims[each] = {}
            for i, x in enumerate(self.c0):
                c_total = np.sum(ck, axis=0) + x
                n_fingerprints = len(ck) + 1
                index = SimilarityIndex(c_total, n_fingerprints, n_ary=self.n_ary, weight=self.weight,
                                        c_threshold=None, w_factor="fraction")()
                if f"f{i}" not in self.sims[each]:
                    self.sims[each][f"f{i}"] = []
                self.sims[each][f"f{i}"] = index
        
        nw_dict = _format_dict(self.sims)
        if not self.weighted_by_frames:
            return nw_dict
        elif self.weighted_by_frames:
            return weight_dict(file_path=None, summary_file=self.summary_file, dict=nw_dict, n_clusters=self.n_clusters)

    def _perform_calculation(self, index_func):
        """Auxillary function for `calculate_medoid` and `calculate_outlier`.

        Args:
            index_func (func): `calculate_medoid` or `calculate_outlier`.

        Returns:
            dict: A dictionary containing the average similarity between each pair of clusters.
        """
        for each, file in enumerate(self.input_files):
            ck = np.load(file)
            index = index_func(ck, n_ary=self.n_ary, weight=self.weight)
            medoid = ck[index]
            self.sims[each] = {}
            for i, x in enumerate(self.c0):
                c_total = medoid + x
                if f"f{i}" not in self.sims[each]:
                    self.sims[each][f"f{i}"] = []
                pair_sim = SimilarityIndex(c_total, 2, n_ary=self.n_ary, weight=self.weight,
                                           c_threshold=None, w_factor="fraction")()
                self.sims[each][f"f{i}"] = pair_sim
        nw_dict = _format_dict(self.sims)
        return nw_dict
    
    def calculate_medoid(self):
        """Calculates the pairwise similarity between every frame in c0 and the medoid of each cluster.

        Notes:
            Calculate the medoid of each cluster using the `calculate_medoid` function from `esim`.
            The pairwise similarity value between each frame in c0 and the medoid of each cluster is calculated 
            using similarity indices.
            Calls the `_perform_calculation` aux function.
        
        Returns:
            If `frame_weighted_sim` returns `False`, 
                nw_dict (dict): unweighted average similarity values.
            If `frame_weighted_sim` returns `True`, 
                w_dict (dict): calls `weight_dict` function to weight similarity values.
        """
        nw_dict = self._perform_calculation(calc_medoid)
        if not self.weighted_by_frames:
            return nw_dict
        elif self.weighted_by_frames:
            return weight_dict(file_path=None, summary_file=self.summary_file, dict=nw_dict, n_clusters=self.n_clusters)
        
    def calculate_outlier(self):
        """Calculates the pairwise similarity between every frame in c0 and the outlier of each cluster.

        Notes:
            Calculate the outlier of each cluster using the `calculate_outlier` function from `esim`.
            The pairwise similarity value between each frame in c0 and the outlier of each cluster is calculated 
            using similarity indices.
            Calls the `_perform_calculation` auxillary function.
        
        Returns:
            If `frame_weighted_sim` returns `False`, 
                nw_dict (dict): unweighted average similarity values.
            If `frame_weighted_sim` returns `True`, 
                w_dict (dict): calls `weight_dict` function to weight similarity values.
        """
        nw_dict = self._perform_calculation(calc_outlier)
        if not self.weighted_by_frames:
            return nw_dict
        elif self.weighted_by_frames:
            return weight_dict(file_path=None, summary_file=self.summary_file, dict=nw_dict, n_clusters=self.n_clusters)

def _trim_outliers(total_data, trim_frac=0.1, n_ary='RR', weight='nw', removal='nan'):
    """Trims a desired percentage of outliers (most dissimilar) from the dataset 
    by calculating largest complement similarity.

    Args:
        total_data (numpy.ndarray): A 2D array, containing the data to be trimmed.
        trim_frac (float): The fraction of outliers to be removed. Must be between 0 and 1. Defaults to 0.1.
        n_ary (str): The similarity metric to be used. Must be either 'RR' or 'SM'. Defaults to 'RR'.
        weight (str): The weight function to be used. Must be either 'nw' or 'fraction'. Defaults to 'nw'.

    Returns:
        numpy.ndarray: A 2D array, with a fraction trimed, corresponding 
            to the rows with the highest complement similarity scores.
    """
    n_fingerprints = len(total_data)
    c_total = np.sum(total_data, axis = 0)
    comp_sims = []
    for i, pixel in enumerate(total_data):
        c_total_i = c_total - total_data[i]
        sim_index = SimilarityIndex(c_total_i, n_fingerprints - 1, n_ary=n_ary, weight=weight,
                                    c_threshold=None, w_factor="fraction")()
        comp_sims.append(sim_index)
    comp_sims = np.array(comp_sims)
    cutoff = int(np.floor(n_fingerprints * float(trim_frac)))
    highest_indices = np.argpartition(-comp_sims, cutoff)[:cutoff]
    if removal == 'nan':
        total_data[highest_indices] = np.nan
    elif removal == 'delete':
        total_data = np.delete(total_data, highest_indices, axis=0)
    return total_data

def weight_dict(file_path=None, summary_file=None, dict=None, n_clusters=None):
    """Calculates frame-weighted similarity values by the number of frames in each cluster.

    Args:
        file_path (str): Path to the json file containing the unweighted similarity values between each pair of clusters.
        summary_file (str): Path to the summary file containing the number of frames in each cluster (CPPTRAJ output).
        dict (dict): A dictionary containing the unweighted similarity values between each pair of clusters.
        n_clusters (int): The number of clusters to analyze. Default is `None`, analyze all clusters from summary file.

    Returns:
        dict: frame-weighted similarity values between each pair of clusters.
    """
    if file_path:
        with open(file_path, 'r') as file:
            dict = json.load(file)
    elif dict:
        dict = dict
    for key in dict:
        dict[key].pop()
    
    num = np.loadtxt(summary_file, unpack=True, usecols=(1), skiprows=(1))
    if n_clusters:
        num = num[0:n_clusters]
    w_sum = np.sum(num, axis=0)
    weights = num / w_sum
    weights = weights[1:]

    w_dict = {}
    for key in dict:
        old_list = dict[key]
        w_dict[key] = [old_list[i] * v for i, v in enumerate(weights)]
    for k in w_dict:
        average = sum(w_dict[k]) / len(w_dict[k])
        w_dict[k].append(average)

    return w_dict

def _format_dict(dict):
    """Sorts dict to have frame # as the key and attaches the average value to the end of each key.

    Args:
        dict (dict): A dictionary containing the similarity values.

    Returns:
        dict: Sorted by the keys with the average value attached to the end of each key.
    """
    nw_dict = {}
    for i in sorted(dict):
        for k, v in dict[i].items():
            if k not in nw_dict:
                nw_dict[k] = [None] * len(dict)
            nw_dict[k][i] = v
    for k in nw_dict:
        average = sum(nw_dict[k]) / len(nw_dict[k])
        nw_dict[k].append(average)
    
    return nw_dict