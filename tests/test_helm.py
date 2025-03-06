import numpy as np
import pytest

from mdance import data
from mdance.tools.bts import extended_comparison
from mdance.cluster.helm import HELM
from mdance.cluster.helm import compute_scores
from mdance.cluster.helm import z_matrix


@pytest.fixture(scope='module')
def traj_numpy():
    """Load the trajectory data."""
    return np.load(data.sim_traj_numpy)


@pytest.fixture(scope='module')
def cluster_labels():
    """Load the cluster labels."""
    frames, labels = np.loadtxt(data.labels_60, dtype=int, unpack=True, 
                                delimiter=',')
    return frames, labels


@pytest.fixture(scope='module')
def input_cluster(traj_numpy, cluster_labels):
    """Build cluster dictionary to pass to HELM."""
    frames, labels = cluster_labels
    unique_labels = np.sort(np.unique(labels))
    input_clusters = []
    for each_unique in unique_labels:
        input_clusters.append(traj_numpy[frames[labels == each_unique]])
    print(f'Number of clusters: {len(input_clusters)}')
    N0 = len(input_clusters)
    return input_clusters, N0


def helm_input(input_cluster):
    """Build cluster dictionary to pass to HELM."""
    input_clusters, N0 = input_cluster
    cluster_dict = {N0: []}
    for i in range(N0):
        Indicesik = [i]
        Nik = len(input_clusters[i])
        c_sumik = np.sum(input_clusters[i], axis=0)
        sq_sumik = np.sum(input_clusters[i]**2, axis=0)
        cluster_dict[N0].append([Indicesik, (c_sumik, sq_sumik), Nik])
    return cluster_dict


def test_helm_eps(input_cluster):
    """Test the ``eps`` parameter."""
    input_clusters, N0 = input_cluster
    eps_dict = helm_input(input_cluster)

    clusters = HELM(cluster_dict=eps_dict, metric='MSD', N_atoms=50, 
                    merge_scheme='inter', eps=15)()
    
    # Compute CH and DB scores
    scores = []
    for key, value in enumerate(clusters.values()):
        input_cluster_indices = []
        if not value:
            continue
        for i in value:
            input_cluster_indices += i[0]
        arr = np.concatenate([input_clusters[i] for i in input_cluster_indices])
        ch, db = compute_scores(value, arr)
        scores.append([N0 - key, ch, db])
    expected_scores = [
        [60, 291.2198060306322, 1.7370614645545726], 
        [59, 295.7352641684398, 1.7122884537735075], 
        [58, 296.11490509768595, 1.7245038612367665], 
        [57, 297.8213492701506, 1.7246552370601154], 
        [56, 299.0810730592307, 1.738637643465005], 
        [55, 300.34386300863565, 1.750692719498292], 
        [54, 302.7012989347063, 1.755325543510106]
        ]
    print(scores)
    assert np.allclose(scores, expected_scores, atol=1e-5)


def test_helm_pops(input_cluster):
    """Test the ``pops`` parameter."""
    n_clus_dict = helm_input(input_cluster)

    clusters = HELM(cluster_dict=n_clus_dict, metric='MSD', N_atoms=50, 
                    merge_scheme='inter', n_clusters=10)()
    
    # Check the population of each cluster
    pops = []
    merged_clusters = []
    for subcluster in clusters[10]:
        merged_clusters.append(subcluster[0])
        pops.append(subcluster[2] / 6001)
    pops = np.sort(pops)[::-1]
    expected_populations = [
        0.32511248,
        0.21213131,
        0.11898017,
        0.11714714,
        0.09948342,
        0.07648725,
        0.02216297, 
        0.01366439,
        0.00799867,
        0.00683219
        ]
    
    expected_merged_clusters = [
        [33], 
        [49], 
        [53], 
        [0, 11], 
        [47, 2, 23, 36, 40, 38, 58], 
        [1, 19, 25, 8, 15, 31, 9, 41, 18, 5, 55], 
        [50, 51, 56], 
        [35, 52, 57, 22, 45, 32, 37], 
        [12, 44, 28, 54, 30, 48, 59, 39, 20, 42], 
        [13, 29, 7, 10, 3, 6, 16, 4, 17, 43, 24, 14, 27, 46, 21, 26, 34]
        ]
    
    assert np.allclose(pops, expected_populations, atol=1e-5)
    assert merged_clusters == expected_merged_clusters


def test_helm_n_clus(input_cluster):
    """Test the ``n_clusters`` parameter."""
    input_clusters, N0 = input_cluster
    n_clus_dict = helm_input(input_cluster)

    clusters = HELM(cluster_dict=n_clus_dict, metric='MSD', N_atoms=50, 
                    merge_scheme='inter', n_clusters=37)()
    
    # Compute CH and DB scores
    scores = []
    for key, value in enumerate(clusters.values()):
        input_cluster_indices = []
        for i in value:
            input_cluster_indices += i[0]
        arr = np.concatenate([input_clusters[i] for i in input_cluster_indices])
        ch, db = compute_scores(value, arr)
        scores.append([N0 - key, ch, db])
    
    expected_scores = [
        [60, 291.2198060306322, 1.7370614645545726], 
        [59, 295.7352641684398, 1.7122884537735075], 
        [58, 296.11490509768595, 1.7245038612367665], 
        [57, 297.8213492701506, 1.7246552370601154], 
        [56, 299.0810730592307, 1.738637643465005], 
        [55, 300.34386300863565, 1.750692719498292], 
        [54, 302.7012989347063, 1.755325543510106], 
        [53, 304.51797241739484, 1.7672459242576242], 
        [52, 306.6006824661377, 1.7695122974000195], 
        [51, 308.9021982976074, 1.7607190308607732], 
        [50, 307.1901532394585, 1.7721711289472055], 
        [49, 297.43176362926556, 1.7888202076551794], 
        [48, 299.34757173879535, 1.7833018738518591], 
        [47, 301.52432336513584, 1.7936056164868994], 
        [46, 306.08279103249237, 1.8034172355518991], 
        [45, 310.3208501789975, 1.809472686067903], 
        [44, 310.0361137593372, 1.825336725435764], 
        [43, 313.8978629372179, 1.8344594141834631], 
        [42, 315.0527200319327, 1.828991467839653], 
        [41, 314.5428710018461, 1.818951502602474], 
        [40, 314.94309021259465, 1.8091147842592137], 
        [39, 318.77039278363225, 1.8100164274989112], 
        [38, 323.7748207939142, 1.8174753570754028], 
        [37, 329.3068227654963, 1.8437066236912167]
        ]
    
    assert np.allclose(scores, expected_scores, atol=1e-5)


def test_trim_k(input_cluster):
    """Test the ``trim_k`` parameter."""
    input_clusters, N0 = input_cluster
    helm_dict = helm_input(input_cluster)
    clusters = HELM(cluster_dict=helm_dict, metric='MSD', N_atoms=50, 
                    merge_scheme='inter', n_clusters=1, trim_start=True, 
                    trim_k=1, trim_val=None, min_samples=0.025)()
    expected_n_clusters = 8
    assert len(clusters[8]) == expected_n_clusters
    
    total_pop = 6001
    pops = []
    msds = []
    for cluster in clusters[8]:
        c_sum = cluster[1][0]
        sq_sum = cluster[1][1]
        Nik = cluster[2]
        msd = extended_comparison((c_sum, sq_sum), 'condensed', N=Nik, N_atoms=50)
        msds.append(msd)
        pops.append(Nik / total_pop)
    assert all([pop > 0.025 for pop in pops])
    
    expected_msds = [
        0.7159290983066504, 2.339201422302611, 3.362941769846286, 
        3.53136323974767, 5.2539237168022215, 5.458759986365163, 
        5.9705582725822515, 6.09399997860286]
    
    assert np.allclose(msds, expected_msds, atol=1e-5)
    N0 = len(clusters)
    # Compute CH and DB scores
    scores = []
    for key, value in enumerate(clusters.values()):
        input_cluster_indices = []
        for i in value:
            input_cluster_indices += i[0]
        arr = np.concatenate([input_clusters[i] for i in input_cluster_indices])
        ch, db = compute_scores(value, arr)
        scores.append([N0 - key, ch, db])
    
    expected_scores = [[8, 1027.0159808301096, 1.3503102468493706], 
                       [7, 1104.807519769014, 1.205066197610796], 
                       [6, 1172.6483112177802, 0.8824355830201499], 
                       [5, 1227.2333015488437, 0.9712858749211579], 
                       [4, 1332.727994435348, 0.9596798547189016], 
                       [3, 1381.3259570829998, 1.1368566266419298], 
                       [2, 1482.6296232781137, 0.9381045938050468], 
                       [1, None, None]]
    
    assert np.allclose(scores[:-1], expected_scores[:-1], atol=1e-5)
    assert scores[-1] == [1, None, None]
    

def test_trim_k_2(input_cluster):
    """Test the ``trim_k`` parameter."""
    helm_dict = helm_input(input_cluster)
    clusters = HELM(cluster_dict=helm_dict, metric='MSD', N_atoms=50, 
                    merge_scheme='inter', n_clusters=1, trim_start=True, 
                    trim_k=50, min_samples=0)()
    expected_n_clusters = 10
    assert len(clusters[10]) == expected_n_clusters
    
    msds = []
    for cluster in clusters[10]:
        c_sum = cluster[1][0]
        sq_sum = cluster[1][1]
        Nik = cluster[2]
        msd = extended_comparison((c_sum, sq_sum), 'condensed', N=Nik, N_atoms=50)
        msds.append(msd)
    
    expected_msds = [0.533111401482992, 0.7159290983066504, 
                     0.84978424877854, 1.5715154323945075, 
                     2.2918233940183708, 2.339201422302611, 
                     2.394802731761288, 2.516757040247173, 
                     2.5290370776959334, 2.788671261009775]
    
    assert np.allclose(msds, expected_msds, atol=1e-5)


def test_trim_val(input_cluster):
    """Test the ``trim_val`` parameter."""
    helm_dict = helm_input(input_cluster)

    clusters = HELM(cluster_dict=helm_dict, metric='MSD', N_atoms=50, 
                    merge_scheme='inter', n_clusters=1, trim_start=True, 
                    trim_val=5, min_samples=0.025)()
    expected_n_clusters = 4
    assert len(clusters[4]) == expected_n_clusters
    
    msds = []
    for cluster in clusters[4]:
        c_sum = cluster[1][0]
        sq_sum = cluster[1][1]
        Nik = cluster[2]
        msd = extended_comparison((c_sum, sq_sum), 'condensed', N=Nik, N_atoms=50)
        msds.append(msd)
    
    expected_msds = [
        0.7159290983066504, 
        2.339201422302611, 
        3.362941769846286, 
        3.53136323974767]
    
    assert np.allclose(msds, expected_msds, atol=1e-5)
    assert all([msd < 5 for msd in msds])
    

def test_z_matrix(input_cluster):
    """Test the ``z_matrix()`` function."""
    n_clus_dict = helm_input(input_cluster)

    clusters = HELM(cluster_dict=n_clus_dict, metric='MSD', N_atoms=50, 
                    merge_scheme='inter', n_clusters=37)()
    Z = z_matrix(clusters)
    
    expected_z_matrix = [
        [2, 23, 1.0, 2], 
        [60, 47, 2.0, 3], 
        [15, 31, 3.0, 2], 
        [45, 22, 4.0, 2], 
        [5, 55, 5.0, 2], 
        [34, 26, 6.0, 2], 
        [41, 9, 7.0, 2], 
        [27, 14, 8.0, 2], 
        [17, 4, 9.0, 2], 
        [8, 62, 10.0, 3], 
        [0, 11, 11.0, 2], 
        [3, 6, 12.0, 2], 
        [25, 19, 13.0, 2], 
        [57, 63, 14.0, 3], 
        [35, 52, 15.0, 2], 
        [64, 18, 16.0, 3], 
        [42, 20, 17.0, 2], 
        [16, 68, 18.0, 3], 
        [66, 75, 19.0, 5], 
        [72, 1, 20.0, 3], 
        [65, 21, 21.0, 3], 
        [40, 36, 22.0, 2], 
        [56, 51, 23.0, 2]
        ]
    
    assert np.allclose(Z, expected_z_matrix, atol=1e-5)


@pytest.fixture(scope='module')
def toy_clusters():
    """Build a toy cluster dictionary."""
    cluster_dict = {
        4: [
            [[0], (1, 2), 3], 
            [[1], (0.5, 1), 2], 
            [[2], (0.1, 2.3), 3], 
            [[3], (1.4, 2.1), 3]
            ]}
    return cluster_dict


@pytest.fixture(scope='module')
def toy_helm_mod(toy_clusters):
    """Build a toy cluster dictionary."""
    return HELM(cluster_dict=toy_clusters, metric='MSD', N_atoms=1, 
                merge_scheme='intra', link='ward', n_clusters=1, trim_start=False)


def test_ward_clust(toy_helm_mod):
    """Test the ``ward_clust()`` function."""
    helm = toy_helm_mod()
    matrix = toy_helm_mod.gen_link_matrix()
    merged = matrix[:,2]
    expected_dict = {
        4: [
            [[0], (0, 0), 3], 
            [[1], (0, 0), 2], 
            [[2], (0, 0), 3], 
            [[3], (0, 0), 3]
            ], 
        
        3: [
            [[0], (0, 0), 3], 
            [[2], (0, 0), 3], 
            [[1, 3], (0, 0), 5]
            ], 
        
        2: [
            [[2], (0, 0), 3], 
            [[0, 1, 3], (0, 0), 8]
            ], 
        
        1: [
            [[2, 0, 1, 3], (0, 0), 11]
            ]
        } 
    
    assert helm == expected_dict
    expected_merge_dist = [0.9512, 1.05940876, 1.46968743]
    assert np.allclose(merged, expected_merge_dist, atol=1e-5)


def test_initial_pairwise_matrix(toy_helm_mod):
    """Test the ``initial_pairwise_matrix()`` function."""
    pairwise_matrix = toy_helm_mod.initial_pairwise_matrix()
    
    expected_matrix = [
        [0.0, 1.02, 1.3661111111111108, 1.0466666666666664], 
        [1.02, 0.0, 1.2912000000000001, 0.9512], 
        [1.3661111111111108, 1.2912000000000001, 0.0, 1.3416666666666668], 
        [1.0466666666666664, 0.9512, 1.3416666666666668, 0.0]]
    
    assert np.allclose(pairwise_matrix, expected_matrix, atol=1e-5)