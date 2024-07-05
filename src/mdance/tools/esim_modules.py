""" Miranda Quintana Group - University of Florida
Extended Similarity Indices Modules

Notes
-----
Please, cite the original papers on the n-ary indices:
https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3
https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00504-4
"""
import numpy as np
from math import ceil

def calculate_counters(c_total, n_objects, c_threshold=None, w_factor="fraction"):
    """Calculate 1-similarity, 0-similarity, and dissimilarity counters

    Parameters
    ---------
    c_total : array-like of shape (n_objects, n_features)
        Vector containing the sums of each column of the fingerprint matrix.
    
    n_objects : int
        Number of objects to be compared.

    c_threshold : {None, 'dissimilar', int}
        Coincidence threshold.
        None : Default, c_threshold = n_objects % 2
        'dissimilar' : c_threshold = ceil(n_objects / 2)
        int : Integer number < n_objects
        float : Real number in the (0 , 1) interval. Indicates the % of the total data that will serve as threshold.

    w_factor : {"fraction", "power_n"}
        Type of weight function that will be used.
        'fraction' : similarity = d[k]/n
                     dissimilarity = 1 - (d[k] - n_objects % 2)/n_objects
        'power_n' : similarity = n**-(n_objects - d[k])
                    dissimilarity = n**-(d[k] - n_objects % 2)
        other values : similarity = dissimilarity = 1

    Returns
    -------
    counters : dict
        Dictionary with the weighted and non-weighted counters.
    """
    # Assign c_threshold
    if not c_threshold:
        c_threshold = n_objects % 2
    if c_threshold == 'dissimilar':
        c_threshold = ceil(n_objects / 2)
    if c_threshold == 'min':
        c_threshold = n_objects % 2
    if isinstance(c_threshold, int):
        if c_threshold >= n_objects:
            raise ValueError("c_threshold cannot be equal or greater than n_objects.")
        c_threshold = c_threshold
    if 0 < c_threshold < 1:
        c_threshold *= n_objects
    
    # Set w_factor
    if w_factor:
        if "power" in w_factor:
            power = int(w_factor.split("_")[-1])
            def f_s(d):
                return power**-float(n_objects - d)
    
            def f_d(d):
                return power**-float(d - n_objects % 2)
        elif w_factor == "fraction":
            def f_s(d):
                return d/n_objects
    
            def f_d(d):
                return 1 - (d - n_objects % 2)/n_objects
        else:
            def f_s(d):
                return 1
    
            def f_d(d):
                return 1
    else:
        def f_s(d):
            return 1
    
        def f_d(d):
            return 1
    
    # Calculate a, d, b + c
    
    a_indices = 2 * c_total - n_objects > c_threshold
    d_indices = n_objects - 2 * c_total > c_threshold
    dis_indices = np.abs(2 * c_total - n_objects) <= c_threshold
    
    a = np.sum(a_indices)
    d = np.sum(d_indices)
    total_dis = np.sum(dis_indices)
    
    a_w_array = f_s(2 * c_total[a_indices] - n_objects)
    d_w_array = f_s(abs(2 * c_total[d_indices] - n_objects))
    total_w_dis_array = f_d(abs(2 * c_total[dis_indices] - n_objects))
    
    w_a = np.sum(a_w_array)
    w_d = np.sum(d_w_array)
    total_w_dis = np.sum(total_w_dis_array)
        
    total_sim = a + d
    total_w_sim = w_a + w_d
    p = total_sim + total_dis
    w_p = total_w_sim + total_w_dis
    
    counters = {"a": a, "w_a": w_a, "d": d, "w_d": w_d,
                "total_sim": total_sim, "total_w_sim": total_w_sim,
                "total_dis": total_dis, "total_w_dis": total_w_dis,
                "p": p, "w_p": w_p}
    return counters
    
def gen_sim_dict(c_total, n_objects, c_threshold=None, w_factor="fraction"):
    """Generate a dictionary with the similarity indices
    
    Parameters
    ----------
    c_total : array-like of shape (n_objects, n_features)
        Vector containing the sums of each column of the fingerprint matrix.
    n_objects : int
        Number of objects to be compared.
    c_threshold : {None, 'dissimilar', int}
        Coincidence threshold.
    w_factor : {"fraction", "power_n"}
        Type of weight function that will be used.
    
    Returns
    -------
    dict
        Dictionary with the similarity indices.
    
    Notes
    -----
    Available indices:
    BUB: Baroni-Urbani-Buser, Fai: Faith, Gle: Gleason, Ja: Jaccard,
    JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
    SM: Sokal-Michener, SSn: Sokal-Sneath n
    """
    
    counters = calculate_counters(c_total, n_objects, c_threshold=c_threshold, 
                                  w_factor=w_factor)
    bub_nw = ((counters['w_a'] * counters['w_d']) ** 0.5 + counters['w_a'])/\
             ((counters['a'] * counters['d']) ** 0.5 + counters['a'] + counters['total_dis'])
    fai_nw = (counters['w_a'] + 0.5 * counters['w_d'])/\
             (counters['p'])
    gle_nw = (2 * counters['w_a'])/\
             (2 * counters['a'] + counters['total_dis'])
    ja_nw = (3 * counters['w_a'])/\
            (3 * counters['a'] + counters['total_dis'])
    jt_nw = (counters['w_a'])/\
            (counters['a'] + counters['total_dis'])
    rt_nw = (counters['total_w_sim'])/\
            (counters['p'] + counters['total_dis'])
    rr_nw = (counters['w_a'])/\
            (counters['p'])
    sm_nw = (counters['total_w_sim'])/\
            (counters['p'])
    ss1_nw = (counters['w_a'])/\
             (counters['a'] + 2 * counters['total_dis'])
    ss2_nw = (2 * counters['total_w_sim'])/\
             (counters['p'] + counters['total_sim'])

    Indices = {'BUB':bub_nw, 'Fai':fai_nw, 'Gle':gle_nw, 'Ja':ja_nw, 'JT':jt_nw, 
               'RT':rt_nw, 'RR':rr_nw, 'SM':sm_nw, 'SS1':ss1_nw, 'SS2':ss2_nw}
    return Indices

class SimilarityIndex:
    def __init__(self, data, n_objects = None, c_threshold = None, n_ary = 'RR', 
                 w_factor = 'fraction', weight = 'nw', return_dict = False):
        # Indices
        # AC: Austin-Colwell, BUB: Baroni-Urbani-Buser, CTn: Consoni-Todschini n
        # Fai: Faith, Gle: Gleason, Ja: Jaccard, Ja0: Jaccard 0-variant
        # JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
        # SM: Sokal-Michener, SSn: Sokal-Sneath n

        self.data = data
        self.n_objects = n_objects
        self.n_ary = n_ary
        self.w_factor = w_factor
        self.c_threshold = c_threshold
        self.weight = weight
        self.counters = calculate_counters(self.data, self.n_objects, self.c_threshold, self.w_factor)
        self.return_dict = return_dict
        
        if self.return_dict == True:
         self.index_functions = \
         {'nw': {'AC': self.ac_nw, 'BUB': self.bub_nw, 'CT1': self.ct1_nw, 
            'CT2': self.ct2_nw, 'CT3': self.ct3_nw, 'CT4': self.ct4_nw, 
            'Fai': self.fai_nw, 'Gle': self.gle_nw, 'Ja': self.ja_nw,
            'Ja0': self.ja0_nw, 'JT': self.jt_nw, 'RT': self.rt_nw, 'RR': self.rr_nw,
            'SM': self.sm_nw, 'SS1': self.ss1_nw, 'SS2': self.ss2_nw},
         'w': {'AC': self.ac_w, 'BUB': self.bub_w, 'CT1': self.ct1_w, 'CT2': self.ct2_w, 'CT3': self.ct3_w,
            'CT4': self.ct4_w, 'Fai': self.fai_w, 'Gle': self.gle_w, 'Ja': self.ja_w,
            'Ja0': self.ja0_w, 'JT': self.jt_w, 'RT': self.rt_w, 'RR': self.rr_w,
            'SM': self.sm_w, 'SS1': self.ss1_w, 'SS2': self.ss2_w},
         'nw_nw': {'RR': self.rr_nw_nw, 'SM': self.sm_nw_nw}}
   
    def __call__(self):
        """The default method to be called when the class is instantiated.
        
        Returns:
        --------
        If `return_dict` attribute is True, returns a dictionary of similarity scores.
        If `return_dict` attribute is False, returns the similarity score obtained by
        applying the specified n-ary comparison method (`n_ary`) and weight function
        (`weight`) to the dataset.
        """

        if self.return_dict:
            return self.gen_dict()
        else:
            return getattr(self, f"{self.n_ary.lower()}_{self.weight}")() 

    def gen_sim_dict(self): 
        """
        Generates a dictionary of all similarity indices.
        """
        return {outer_key: {inner_key: inner_func() for inner_key, inner_func in inner_dict.items()}
            for outer_key, inner_dict in self.index_functions.items()}


    # Calculate for individual Index Functions
    # Weighted Indices
    def ac_w(self):
        ac_w = (2/np.pi) * np.arcsin(np.sqrt(self.counters['total_w_sim']/
                                            self.counters['w_p']))
        return ac_w

    def bub_w(self):
        bub_w = ((self.counters['w_a'] * self.counters['w_d'])**0.5 + self.counters['w_a'])/\
                ((self.counters['w_a'] * self.counters['w_d'])**0.5 + self.counters['w_a'] + self.counters['total_w_dis'])
        return bub_w

    def ct1_w(self):
        ct1_w = (log(1 + self.counters['w_a'] + self.counters['w_d']))/\
                (log(1 + self.counters['w_p']))
        return ct1_w

    def ct2_w(self):
        ct2_w = (log(1 + self.counters['w_p']) - log(1 + self.counters['total_w_dis']))/\
                (log(1 + self.counters['w_p']))
        return ct2_w
    
    def ct3_w(self):
        ct3_w = (log(1 + self.counters['w_a']))/\
                (log(1 + self.counters['w_p']))
        return ct3_w

    def ct4_w(self):
        ct4_w = (log(1 + self.counters['w_a']))/\
                (log(1 + self.counters['w_a'] + self.counters['total_w_dis']))
        return ct4_w

    def fai_w(self):
        fai_w = (self.counters['w_a'] + 0.5 * self.counters['w_d'])/\
                (self.counters['w_p'])
        return fai_w

    def gle_w(self):
        gle_w = (2 * self.counters['w_a'])/\
                (2 * self.counters['w_a'] + self.counters['total_w_dis'])
        return gle_w

    def ja_w(self):
        ja_w = (3 * self.counters['w_a'])/\
               (3 * self.counters['w_a'] + self.counters['total_w_dis'])
        return ja_w

    def ja0_w(self):
        ja0_w = (3 * self.counters['total_w_sim'])/\
                (3 * self.counters['total_w_sim'] + self.counters['total_w_dis'])
        return ja0_w

    def jt_w(self):
        jt_w = (self.counters['w_a'])/\
               (self.counters['w_a'] + self.counters['total_w_dis'])
        return jt_w

    def rt_w(self):
        rt_w = (self.counters['total_w_sim'])/\
               (self.counters['w_p'] + self.counters['total_w_dis'])
        return rt_w

    def rr_w(self):
        rr_w = (self.counters['w_a'])/\
               (self.counters['w_p'])
        return rr_w

    def sm_w(self):
        sm_w = (self.counters['total_w_sim'])/\
               (self.counters['w_p'])
        return sm_w

    def ss1_w(self):
        ss1_w = (self.counters['w_a'])/\
                (self.counters['w_a'] + 2 * self.counters['total_w_dis'])
        return ss1_w

    def ss2_w(self):
        ss2_w = (2 * self.counters['total_w_sim'])/\
                (self.counters['w_p'] + self.counters['total_w_sim'])
        return ss2_w

    # Non-Weighted Indices
    def ac_nw(self):
        ac_nw = (2/np.pi) * np.arcsin(np.sqrt(self.counters['total_w_sim']/
                                            self.counters['p']))
        return ac_nw
    
    def bub_nw(self):
        bub_nw = ((self.counters['w_a'] * self.counters['w_d'])**0.5 + self.counters['w_a'])/\
                ((self.counters['a'] * self.counters['d'])**0.5 + self.counters['a'] + self.counters['total_dis'])
        return bub_nw

    def ct1_nw(self):
        ct1_nw = (log(1 + self.counters['w_a'] + self.counters['w_d']))/\
                (log(1 + self.counters['p']))
        return ct1_nw

    def ct2_nw(self):
        ct2_nw = (log(1 + self.counters['w_p']) - log(1 + self.counters['total_w_dis']))/\
                (log(1 + self.counters['p']))
        return ct2_nw

    def ct3_nw(self):
        ct3_nw = (log(1 + self.counters['w_a']))/\
                (log(1 + self.counters['p']))
        return ct3_nw

    def ct4_nw(self):
        ct4_nw = (log(1 + self.counters['w_a']))/\
                (log(1 + self.counters['a'] + self.counters['total_dis']))
        return ct4_nw

    def fai_nw(self):
        fai_nw = (self.counters['w_a'] + 0.5 * self.counters['w_d'])/\
                (self.counters['p'])
        return fai_nw

    def gle_nw(self):
        gle_nw = (2 * self.counters['w_a'])/\
                (2 * self.counters['a'] + self.counters['total_dis'])
        return gle_nw

    def ja_nw(self):
        ja_nw = (3 * self.counters['w_a'])/\
                (3 * self.counters['a'] + self.counters['total_dis'])
        return ja_nw

    def ja0_nw(self):
        ja0_nw = (3 * self.counters['total_w_sim'])/\
                (3 * self.counters['total_sim'] + self.counters['total_dis'])
        return ja0_nw

    def jt_nw(self):
        jt_nw = (self.counters['w_a'])/\
                (self.counters['a'] + self.counters['total_dis'])
        return jt_nw

    def rt_nw(self):
        rt_nw = (self.counters['total_w_sim'])/\
                (self.counters['p'] + self.counters['total_dis'])
        return rt_nw

    def rr_nw(self):
        rr_nw = (self.counters['w_a'])/\
                (self.counters['p'])
        return rr_nw

    def sm_nw(self):
        sm_nw = (self.counters['total_w_sim'])/\
                (self.counters['p'])
        return sm_nw

    def ss1_nw(self):
        ss1_nw = (self.counters['w_a'])/\
                (self.counters['a'] + 2 * self.counters['total_dis'])
        return ss1_nw

    def ss2_nw(self):
        ss2_nw = (2 * self.counters['total_w_sim'])/\
                (self.counters['p'] + self.counters['total_sim'])
        return ss2_nw
    
    # Non-Weighted Indices for Denominator and Numerator
    def rr_nw_nw(self):
        rr_nw_nw = (self.counters['a'])/\
                (self.counters['p'])
        return rr_nw_nw
    
    def sm_nw_nw(self):
        sm_nw_nw = (self.counters['total_sim'])/\
                (self.counters['p'])
        return sm_nw_nw

def calc_medoid(data, n_ary = 'RR', w_factor = 'fraction', weight = 'nw', c_total = None):
    """Calculate the medoid of a set
    
    Arguments 
    --------    
    data : np.array
        np.array of all the binary objects

    n_ary : str
        string with the initials of the desired similarity index to calculate the medoid from. 
        See gen_sim_dict description for keys
    
    weight : str, default = 'nw'
        string with the initials of the desired weighting factor to calculate the medoid from.
    
    w_factor : str, default = 'fraction'
        desired weighing factors for the counters

    c_total: np.array, default = None
        np.array with the columnwise sums.
    """
        
    if c_total and len(data[0]) != c_total: raise ValueError("Dimensions of objects and columnwise sum differ")
    elif not c_total: c_total = np.sum(data, axis = 0)
    n_objects = len(data)
    index = n_objects + 1
    min_sim = 1.01
    comp_sums = c_total - data
    for i, obj in enumerate(comp_sums):
        SI = SimilarityIndex(obj, n_objects = n_objects - 1, n_ary = n_ary, w_factor = w_factor, weight = weight)
        sim_index = SI()
        if sim_index < min_sim:
            min_sim = sim_index
            index = i
        else:
            pass
    return index

def calc_outlier(data, n_ary = 'RR', w_factor = 'fraction', weight = 'nw', c_total = None):
    """Calculate the medoid of a set
    Arguments 
    --------    
    data : np.array
        np.array of all the binary objects

    n_ary : str
        string with the initials of the desired similarity index to calculate the medoid from. 
        See gen_sim_dict description for keys
    
    weight : str, default = 'nw'
        string with the initials of the desired weighting factor to calculate the medoid from.
    
    w_factor : str, default = 'fraction'
        desired weighing factors for the counters

    c_total: np.array, default = None
        np.array with the columnwise sums.
    """

    if c_total and len(data[0]) != c_total: raise ValueError("Dimensions of objects and columnwise sum differ")
    elif not c_total: c_total = np.sum(data, axis = 0)
    n_objects = len(data)
    index = n_objects + 1
    max_sim = -0.01
    comp_sums = c_total - data 
    for i, obj in enumerate(comp_sums):
        SI = SimilarityIndex(obj, n_objects = n_objects - 1, n_ary = n_ary, w_factor = w_factor, weight = weight)
        sim_index = SI()
        if sim_index > max_sim:
            max_sim = sim_index
            index = i
        else:
            pass
    return index

def calc_comp_sim(data, c_threshold = None, n_ary = 'RR', w_factor = 'fraction', weight = 'nw', c_total = None):
    """Calculate the complementary similarity for all elements"""

    if c_total and len(data[0]) != c_total: raise ValueError("Dimensions of objects and columnwise sum differ")
    elif not c_total: c_total = np.sum(data, axis = 0)

    n_objects = len(data)   
    comp_sums = c_total - data
    total = []
    for i, obj in enumerate(comp_sums):
        SI = SimilarityIndex(obj, n_objects = n_objects - 1, n_ary = n_ary, w_factor = w_factor, weight = weight)
        sim_index = SI()
        total.append((i, sim_index))
    return total