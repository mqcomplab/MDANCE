# HELM - Hierarchical Extended Linkage Method

Table of Contents
=================
- [Overview](#overview)
- [Tutorial](#tutorial)
    - [1. Input Preparations](#1-input-preparations)
    - [2. Prior Clustering (Optional)](#2-prior-clustering-optional)
    - [3. HELM clustering](#3-helm-clustering)
    - [4. Get most optimal number of clusters](#4-get-most-optimal-number-of-clusters)
    - [5. Assign cluster labels to the trajectory](#5-assign-cluster-labels-to-the-trajectory)
    - [6. Extract frames for each cluster (Optional)](#6-extract-frames-for-each-cluster-optional)

## Overview
HELM is a hierarchical agglomerative clustering algorithm that uses the *n*-ary similarity to merge clusters at each level. It transformed from the traditional hierarchical clustering algorithm to be more efficient and scalable turning a $O(N^2)$ algorithm to $O(N)$. It specializes in recognizing dominant conformations within an ensemble and is often used alongside NANI to achieve a balance between efficiency and precision. 

This clustering tutorial is meant for datasets for all applications (2D fingerprints, mass spectrometry imaging data, etc). Molecular Dynamics Trajectory has a different treatment. If specific step is only for Molecular Dynamics trajectory, it will be specified. Otherwise, it is applicable for all datasets.

## Tutorial
### 1. Input Preparations
<details>
<summary>Preparation for Molecular Dynamics Trajectory</summary>

Prepare a valid topology file (e.g. `.pdb`, `.prmtop`), trajectory file (e.g. `.dcd`, `.nc`), and the atom selection. This step will convert a Molecular Dynamics trajectory to a numpy ndarray. **Make sure the trajectory is already aligned and/or centered if needed!**

**Step-by-step tutorial can be found in the [scripts/inputs/preprocessing.ipynb](../scripts/inputs/preprocessing.ipynb).**
</details>

<details>
<summary>Preparation for all other datasets (OPTIONAL)</summary>

This step is **optional**. If you are using a metric that is NOT the mean-square deviation (MSD)--default metric, you will need to normalize the dataset. Otherwise, you can skip this step.

[**scripts/inputs/normalize.py**](../scripts/inputs/normalize.py) will normalize the dataset. The following parameters to be specified in the script:

    # System info - EDIT THESE
    data_file = data.blob_disk
    array = np.genfromtxt(data_file, delimiter=',')
    output_base_name = 'output_base_name'

#### Inputs
##### System info
`data_file` is your input file with a 2D array.<br>
`array` is the array is the loaded dataset from `data_file`. This step can be changed according to the type of file format you have. However, `array` must be an array-like in the shape (number of samples, number of features).<br>
`output_base_name` is the base name for the output file. The output file will be saved as `output_base_name.npy`.<br>
</details>


### 2. Prior Clustering (Optional)
HELM requires a set of initial clusters to start with. You can start from any clustering method. An example is with the NANI clustering also among this tutorial set. All you need is to have the cluster labels similar to this format. 

    #frame,cluster
    0,0
    1,0
    2,1
    3,1
    4,2

### 3. HELM clustering.
[**scripts/helm/intra/run_helm.py**](../scripts/helm/intra/run_helm_intra.py) will run HELM clustering on the dataset. The following parameters to be specified in the script:

    # System info - EDIT THESE
    input_traj_numpy = data.sim_traj_numpy
    cluster_labels = '../labels_60.csv'
    sieve = 1
    N_atoms = 50                                    # Number of atoms in the system

    # HELM params - EDIT THESE
    metric = 'MSD'                                  # Default  
    N0 = 60                                         # How many clusters to start with
    final_target = 1                                # How many clusters to end with
    align_method = None                             # Default
    save_pairwise_sim = False                       # Default
    merging_scheme = 'inter'                        # {'inter', 'intra'}

#### Inputs
##### System info

`input_file` is the trajectory from step 1. <br>
`cluster_labels` is the cluster labels from step 2. <br>
`sieve` is reading every `sieve` frames from the trajectory. <br>
`N_atoms` is the number of atoms used in the clustering. <br>

##### HELM params
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/mdance/tools/bts.py#L96) for details). <br>
`N0` is the number of clusters to start with. <br>
`final_target` is the number of clusters to end with. <br>
`align_method` *(optional)* is the method to align the clusters. Default is None. <br>
`save_pairwise_sim` *(optional)* is a boolean variable to indicate whether to save the pairwise similarity matrix. Default is False. <br>
`merging_scheme` *(optional)* is the merging scheme to use. {`inter`, `intra`}. `inter` merges clusters with lowest interdistance. `intra` merges clusters with lowest intradistance. Default is `inter`. <br>

#### Execution
```bash
python run_helm_intra.py
```

#### Outputs
- Pickle file containing the clustering results. <br>
- CSV file containing the Calinski-Harabasz and Davies-Bouldin scores for each number of clusters. 

### 4. Get most optimal number of clusters.
Find the most optimal number of clusters from clustering scores.

**Step-by-step tutorial can be found in the [scripts/helm/intra/analysis_db.ipynb](../scripts/helm/intra/analysis_db.ipynb).**


### 5. Cluster Assignment
[**scripts/helm/intra/assign_labels.py**](../scripts/helm/intra/assign_labels_intra.py) will assign cluster labels to the trajectory. The following parameters to be specified in the script:

    # System info - EDIT THESE
    input_traj_numpy = data.sim_traj_numpy
    N_atoms = 50
    sieve = 1

    # HELM params - EDIT THESE
    n_clusters = 10
    pre_cluster_labels = '../labels_60.csv'
    pickle_file = 'inter-helm.pkl'
    metric = 'MSD'                                                      # Default
    extract_type = 'top'                                                # Default
    n_structures = 11                                                   # Default


#### Inputs
##### System info
`input_traj_numpy` is the normalized trajectory from step 1. <br>
`N_atoms` is the number of atoms used in the clustering. <br>
`sieve` is reading every `sieve` frames from the trajectory. <br>

##### HELM params
`n_clusters` is the number of clusters to assign labels to. Use most optimal number of clusters from analysis in step 4. <br>
`pre_cluster_labels` is the cluster labels from step 2. <br>
`pickle_file` is the clustering results from step 3. <br>
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/mdance/tools/bts.py#L96) for details). <br>
`extract_type` is the type of extraction method to use. {`top`, `random`}. `top` means to extract the top `n_structures` from each cluster. `random` means to extract `n_structures` random structures from each cluster. <br>
`n_structures` is the number of structures to extract from each cluster. <br>

#### Execution
```bash
python assign_labels_intra.py
```

#### Outputs
`helm_cluster_labels.csv` contains the cluster labels for each frame. <br>
`helm_best_frames_indices.csv` contains the indices of the best or random frames to extract from each cluster. <br>
`helm_summary.csv` contains the summary of the clustered population. 

### 6. Extract frames for each cluster (Optional)
**This step is *optional* and for Molecular Dynamics Trajectories only.**

The last step is to extract the designated frames from the original trajectory for each cluster.

**Step-by-step tutorial can be found in the [postprocessing notebook](../scripts/outputs/postprocessing.ipynb).**

<kbd> [↩️ Main](../README.md) </kbd>