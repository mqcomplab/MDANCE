import glob
import json
import numpy as np
import re

from mdance.tools.esim import SimilarityIndex, calc_medoid, calc_outlier


class FrameSimilarity:
    """A class to calculate the similarity between clusters.
        
    Parameters
    ----------
    cluster_folder : str
        The path to the folder containing the normalized cluster files.
    summary_file : str
        The path to the summary file containing the number of frames for each cluster.
    trim_frac : float
        The fraction of outliers to trim from the top cluster.
    n_clusters : int
        The number of clusters to analyze.
    weighted_by_frames : bool
        Whether to weight the similarity values by the number of frames in the cluster.
    n_ary : {'RR', 'SM'}
        The n_ary similarity metric to use.
    weight : {'nw', 'w', 'fraction'}
        The weight to use for the similarity metric.
    """
    def __init__(self, cluster_folder=None, summary_file=None, trim_frac=None, n_clusters=None, 
                 weighted_by_frames=True, n_ary='RR', weight='nw'):
        self.c0 = np.load(f"{cluster_folder}/normed_clusttraj.c0.npy")
        if trim_frac:
            self.c0 = _trim_outliers(self.c0, trim_frac=trim_frac, n_ary=n_ary, weight=weight)
        self.input_files = sorted(glob.glob(f"{cluster_folder}/normed_clusttraj.c*"), 
                                  key=lambda x: int(re.findall("\d+", x)[0]))[1:]
        self.summary_file = summary_file
        self.n_clusters = n_clusters
        self.weighted_by_frames = weighted_by_frames
        self.n_ary = n_ary
        self.weight = weight
        self.sims = {}
    
    def calculate_pairwise(self):
        """Calculates pairwise similarity between each cluster and all other 
        clusters. The similarity score is calculated as the average of pairwise 
        similarity values between each frame in the cluster and the top c0 cluster.  
        
        Returns
        -------
        dict
            A dictionary containing the average similarity between each pair of clusters.
            ``weighted_by_frames=True`` will return the frame-weighted similarity values.
            ``weighted_by_frames=False`` will return the unweighted similarity values.
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
            return weight_dict(file_path=None, summary_file=self.summary_file, 
                               dict=nw_dict, n_clusters=self.n_clusters)

    def calculate_union(self):
        """Calculates the extended similarity between the union of frame 
        in c0 and cluster k. The similarity score is calculated as the union 
        similarity between all frames in the cluster and the top c0 cluster.
        
        Returns
        -------
        dict
            A dictionary containing the average similarity between each pair of clusters.
            ``weighted_by_frames=True`` will return the frame-weighted similarity values.
            ``weighted_by_frames=False`` will return the unweighted similarity values.
        """
        for each, file in enumerate(self.input_files):
            ck = np.load(file)
            self.sims[each] = {}
            for i, x in enumerate(self.c0):
                c_total = np.sum(ck, axis=0) + x
                n_fingerprints = len(ck) + 1
                index = SimilarityIndex(c_total, n_fingerprints, n_ary=self.n_ary, 
                                        weight=self.weight,
                                        c_threshold=None, w_factor="fraction")()
                if f"f{i}" not in self.sims[each]:
                    self.sims[each][f"f{i}"] = []
                self.sims[each][f"f{i}"] = index
        
        nw_dict = _format_dict(self.sims)
        if not self.weighted_by_frames:
            return nw_dict
        elif self.weighted_by_frames:
            return weight_dict(file_path=None, summary_file=self.summary_file, 
                               dict=nw_dict, n_clusters=self.n_clusters)

    def _perform_calculation(self, index_func):
        """Auxillary function for ``calculate_medoid`` and ``calculate_outlier``.
        
        Parameters
        ----------
        index_func : function
            The function to calculate the medoid or outlier of each cluster.
        
        Returns
        -------
        dict
            A dictionary containing the average similarity between each pair of clusters.
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
        """Calculates the pairwise similarity between every frame in c0 and the 
        medoid of each cluster. The pairwise similarity value between each frame 
        in c0 and the medoid of each cluster is calculated using similarity indices.
        
        Returns
        -------
        dict
            A dictionary containing the average similarity between each pair of clusters.
            ``weighted_by_frames=True`` will return the frame-weighted similarity values.
            ``weighted_by_frames=False`` will return the unweighted similarity values.
        """
        nw_dict = self._perform_calculation(calc_medoid)
        if not self.weighted_by_frames:
            return nw_dict
        elif self.weighted_by_frames:
            return weight_dict(file_path=None, summary_file=self.summary_file, dict=nw_dict, n_clusters=self.n_clusters)
        
    def calculate_outlier(self):
        """Calculates the pairwise similarity between every frame in c0 and the 
        outlier of each cluster. The pairwise similarity value between each frame 
        in c0 and the outlier of each cluster is calculated using similarity indices.
        
        Returns
        -------
        dict
            A dictionary containing the average similarity between each pair of clusters.
            ``weighted_by_frames=True`` will return the frame-weighted similarity values.
            ``weighted_by_frames=False`` will return the unweighted similarity values.
        """
        nw_dict = self._perform_calculation(calc_outlier)
        if not self.weighted_by_frames:
            return nw_dict
        elif self.weighted_by_frames:
            return weight_dict(file_path=None, summary_file=self.summary_file, dict=nw_dict, n_clusters=self.n_clusters)


def _trim_outliers(total_data, trim_frac=0.1, n_ary='RR', weight='nw', removal='nan'):
    """Trims a desired percentage of outliers (most dissimilar) from the dataset 
    by calculating largest complement similarity.

    Parameters
    ----------
    total_data : numpy.ndarray
        The input data to be trimmed.
    trim_frac : float
        The fraction of outliers to trim. Defaults to 0.1.
    n_ary : {'RR', 'SM'}
        The similarity metric to use.
    weight : {'nw', 'w', 'fraction'}
        The weight to use for the similarity metric.
    removal : {'nan', 'delete'}
        The value to replace the trimmed data with. Defaults to 'nan'.
    
    Returns
    -------
    numpy.ndarray
        The trimmed data.
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
    
    Parameters
    ----------
    file_path : str
        The path to the json file containing the unweighted similarity values.
    summary_file : str
        The path to the summary file containing the number of frames for each cluster.
    dict : dict
        A dictionary containing the unweighted similarity values.
    n_clusters : int
        The number of clusters to analyze.
    
    Returns
    -------
    dict
        A dictionary containing the frame-weighted similarity values between each pair of clusters.
    """
    if file_path:
        with open(file_path, 'r') as file:
            dict = json.load(file)
    elif dict:
        dict = dict
    for key in dict:
        dict[key].pop()
    
    num = np.loadtxt(summary_file, unpack=True, usecols=(1), skiprows=(1), delimiter=',')
    if n_clusters:
        num = num[0:n_clusters]
    weights = num
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
    
    Parameters
    ----------
    dict : dict
        A dictionary containing the similarity values.
    
    Returns
    -------
    dict
        A dictionary sorted by the keys with the average value attached to the end of each key.
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