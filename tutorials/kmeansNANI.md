# KmeansNANI - KMeans N-Ary Natural Implementation

Table of Contents
=================
- [Overview](#overview)
- [Tutorial](#tutorial)
    - [1. Normalize the trajectory](#1-normalize-the-trajectory)
    - [2. Screening K-MeansNANI for multiple number of clusters](#2-screening-k-meansnani-for-multiple-number-of-clusters)
    - [3. Analysis of NANI screening results](#3-analysis-of-nani-screening-results)
    - [4. Assign labels to the clusters](#4-assign-labels-to-the-clusters)
    - [5. Extract frames for each cluster (Optional)](#5-extract-frames-for-each-cluster-optional)

## Overview
K-MeansNANI is an algorithm for selecting initial centroids for K-Means clustering. K-MeansNANI is an extension of the K-Means++ algorithm. K-Means++ selects the initial centroids by randomly selecting a point from the dataset and then selecting the next centroid based on the distance from the previous centroid. K-MeansNANI stratifies the data to high density region and perform diversity selection on top of the it to select the initial centroids. This is a deterministic algorithm that will always select the same initial centroids for the same dataset and improve on k-means++ by reducing the number of iterations required to converge and improve the clustering quality.

This clustering tutorial is meant for datasets for all applications (2D fingerprints, mass spectrometry imaging data, etc). Molecular Dynamics Trajectory has a different treatment. If specific step is only for Molecular Dynamics trajectory, it will be specified. Otherwise, it is applicable for all datasets.

## Tutorial
### 1. Normalize the trajectory.
<details>
<summary>Normalization for Molecular Dynamics Trajectory</summary>

Prepare a valid topology file (e.g. `.pdb`, `.prmtop`), trajectory file (e.g. `.dcd`, `.nc`), and the atom selection. This step will convert a Molecular Dynamics trajectory to a numpy ndarray.  

**Step-by-step tutorial can be found in the [scripts/inputs/preprocessing.ipynb](../scripts/inputs/preprocessing.ipynb).**
</details>

<details>
<summary>Normalization for all other datasets</summary>

[**scripts/inputs/normalize.py**](../scripts/inputs/normalize.py) will normalize the dataset. The following parameters to be specified in the script:

    # System info - EDIT THESE
    data_file = '../examples/2D/blob_disk.csv'
    array = np.genfromtxt(data_file, delimiter=',')
    output_base_name = '../examples/2D/blob_normed'

#### Inputs
##### System info
`data_file` is your input file. 
`array` is the array from the loaded dataset. This step can be changed according to the type of file format you have. However, `array` must be an array-like in the shape (number of samples, number of features).
`output_base_name` is output base name
</details>

### 2. Screening K-MeansNANI for multiple number of clusters.
[scripts/kmeansnani/screen_nani.py](../scripts/kmeansnani/screen_nani.py) will run K-MeansNANI clustering for multiple number of clusters and give the most optimal number of clusters. For the best result, we recommend running K-MeansNANI with a wide range of number of clusters. The following parameters to be specified in the script:

    # System info - EDIT THESE
    input_traj_numpy = '../../examples/md/backbone.npy'
    output_dir = 'results'                                      
    percentage = 10
    N_atoms = 50
    init_types = ['comp_sim', 'k-means++', 'random', 'vanilla_kmeans++']
    metric = 'MSD'
    start_n_clusters = 5
    end_n_clusters = 31
    sieve = 1

`input_file` is the normalized trajectory from step 1. <br>
`output_dir` is the directory to store the clustering results. <br>
`percentage` is ONLY for `init_type='comp_sim'` and `'div_select'` and is the percentage of frames to be selected to use for diversity selection. <br>
`N_atoms` is the number of atoms used in the clustering. <br>
`init_types` is the list of initialization methods to use. <br>
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/tools/bts.py) for details). <br>
`start_n_clusters` is the starting number of clusters to use. <br>
`end_n_clusters` is the ending number of clusters to use. <br>
`sieve` takes every `sieve`th frame from the trajectory for analysis. <br>

#### Execution
```bash
python screen_nani.py
```

#### Outputs
csv file containing the number of clusters and the corresponding number of iterations, Callinski-Harabasz score, Davies-Bouldin score, and mean-square deviation for each initialization method. 

### 3. Analysis of NANI screening results.
[scripts/kmeansnani/analysis.ipynb](../scripts/kmeansnani/analysis.ipynb) will analyze different initialization methods based on number of iterations, Callinski-Harabasz score, Davies-Bouldin score, and mean-square deviation.

**Step-by-step tutorial can be found in the [analysis notebook](../scripts/kmeansnani/analysis.ipynb).**

### 4. Assign labels to the clusters
[scripts/kmeansnani/assign_labels.py](../scripts/kmeansnani/assign_labels.py) will assign labels to the clusters for Kmeans clustering using the initialization methods. 
The following parameters to be specified in the script:

    # System info - EDIT THESE
    input_traj_numpy = '../../examples/md/normed_backbone.npy'
    N_atoms = 50
    sieve = 1

    # K-means params - EDIT THESE
    n_clusters = 15
    init_type = 'k-means++'                                             # Default
    percentage = None                                                   # Default
    metric = 'MSD'                                                      # Default
    n_structures = 11                                                   # Default

#### Inputs
##### System info
`input_traj_numpy` is the normalized trajectory from step 1. <br>
`N_atoms` is the number of atoms used in the clustering. <br>
`sieve` takes every `sieve`th frame from the trajectory for analysis. <br>

##### K-means params
`n_clusters` is the number of clusters to use. <br>
`init_type` is the initialization method to use. <br>
`percentage` is ONLY for `init_type='comp_sim'` and `'div_select'` and is the percentage of frames to be selected to use for diversity selection. <br>
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/tools/bts.py) for details). <br>
`n_structures` is the number of frames to extract from each cluster. 

#### Execution
```bash
python assign_labels.py
```

#### Outputs
csv file contains the indices of the best frames in each cluster. <br>
csv file contains the cluster labels for each frame. <br>
csv file contains the population of each cluster. <br>

### 5. Extract frames for each cluster (Optional)
[scripts/postprocessing.ipynb](../scripts/postprocessing.ipynb) will use the indices from last step to extract the designated frames from the original trajectory for each cluster.


### Extra Analysis
[scripts/nani/rank.py](../scripts/nani/rank.py) will rank which initialization method is the best using the metrics: number of iterations, Callinski-Harabasz score, Davies-Bouldin score, and average mean-square deviation. (1) means `comp_sim` is the best, (2) means `k-means++` is the best, (3) means `vanilla_kmeans++` is the best, and (4) means `random` is the best.
[scripts/nani/rmsd_dist.py](../scripts/nani/rmsd_dist.py) will calculate the RMSD distribution for each cluster.

**Step-by-step tutorial can be found in the [postprocessing notebook](../scripts/postprocessing.ipynb).**

<kbd> [↩️ Main](../README.md) </kbd>