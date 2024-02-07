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

if __name__ == "__main__":
    arr = np.random.rand(100, 100)
    d = gen_sim_dict(np.sum(arr, axis=0), 100)
    print(d["RR"])