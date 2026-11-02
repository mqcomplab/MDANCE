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

    # System parameters
    data_file = data.blob_disk                          # Input data as a CSV file
    array = np.genfromtxt(data_file, delimiter=',')
    output_base_name = 'array'                          # Prefix for the output files

#### Inputs
##### System parameters
`data_file` is your input file with a 2D array.<br>
`array` is the array is the loaded dataset from `data_file`. This step can be changed according to the type of file format you have. However, `array` must be an array-like in the shape (number of samples, number of features).<br>
`output_base_name` is the base name for the output file. The output file will be saved as `outputs/{output_base_name}.npy`.<br>
</details>


### 2. Prior Clustering (Optional)
HELM requires a set of initial clusters to start with. You can start from any clustering method. An example is with the NANI clustering also among this tutorial set. All you need is to have the cluster labels similar to this format (frame index, cluster index); `mdance.data.labels_60` is a sample with 60 clusters from NANI. 

    #frame,cluster
    0,0
    1,0
    2,1
    3,1
    4,2

### 3. HELM clustering.
[**scripts/helm/run_helm.py**](../scripts/helm/run_helm.py) will run HELM clustering on the dataset. The following parameters to be specified in the script:

    # System parameters
    input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
    cluster_labels = data.labels_60                     # Labels of the initial clusters (e.g. from NANI)
    sieve = 1                                           # Use every sieve-th frame (1 = all frames)
    N_atoms = 50                                        # Number of atoms in the system

    # HELM parameters
    metric = 'MSD'                                      # Distance metric for clustering
    merge_scheme = 'intra'                              # Merge scheme {'inter', 'intra', 'half'}
    link = None                                         # Linkage method {None, 'ward'}
    n_clusters = 1                                      # Either n_clusters or eps must be set
    eps = None                                          # Either n_clusters or eps must be set
    trim_start = False                                  # Remove the least compact initial clusters first
    trim_val = None                                     # Either trim_val or trim_k must be set
    trim_k = 5                                          # Either trim_val or trim_k must be set
    save_pairwise_sim = False                           # Save the pairwise similarity matrix
    output_dir = 'outputs'                              # Folder for output files

    # Analysis parameters
    plot_metrics = True                                 # Plot DBI and CHI, and suggest values
    x_min = 5                                           # Only suggest values >= x_min (None = all)

#### Inputs
##### System parameters

`input_traj_numpy` is the numpy array prepared from step 1. <br>
`cluster_labels` is the cluster labels from step 2. <br>
`sieve` is reading every `sieve` frames from the trajectory. <br>
`N_atoms` is the number of atoms used in the clustering. <br>

##### HELM parameters
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/mdance/tools/bts.py#L94) for details). <br>
`merge_scheme` is the merging scheme to use. {`inter`, `intra`, `half`}. `inter` merges clusters with lowest interdistance. `intra` merges clusters with lowest intradistance. `half` merges clusters with the lowest combined MSD minus the average MSD of the two clusters. <br>
`link` *(optional)* is `'ward'` to merge the clusters with Ward linkage instead of `merge_scheme`. Default is None. <br>
`n_clusters` is the number of clusters to stop at. Either `n_clusters` or `eps` must be set. <br>
`eps` is the largest distance at which two clusters can still be merged. Either `n_clusters` or `eps` must be set. <br>
`trim_start` *(optional)* is a boolean to remove the least compact initial clusters before merging, using `trim_val` (an MSD cutoff) or `trim_k` (the number of clusters to remove). Default is False. <br>
`save_pairwise_sim` *(optional)* is a boolean variable to indicate whether to save the pairwise similarity matrix. Default is False. <br>
`output_dir` is the directory to store the results. <br>

##### Analysis parameters
`plot_metrics` is a boolean to plot the clustering scores and mark the suggested number of clusters (see step 4). If False, no analysis is done. <br>
`x_min` is the smallest number of clusters that can be suggested. The whole hierarchy is still plotted. `None` considers all values. <br>

#### Execution
```bash
python run_helm.py
```

#### Outputs
Files are named after `link`, or `merge_scheme` if `link` is None (e.g. `intra`):
- `outputs/helm_intra.pkl` containing the clustering results. <br>
- `outputs/helm_intra_scores.csv` containing the Calinski-Harabasz score, Davies-Bouldin score, and average MSD of the clusters for each number of clusters. 

### 4. Get most optimal number of clusters.
Find the most optimal number of clusters from clustering scores.

When `plot_metrics = True`, `run_helm.py` plots the Davies-Bouldin index (DBI) and the Calinski-Harabasz index (CHI) of each number of clusters (e.g. `outputs/helm_intra_scores_dbi.png` and `outputs/helm_intra_scores_chi.png`) with the suggested number of clusters marked: lowest DBI or highest CHI, and the sharpest 2<sup>nd</sup> derivatives. The same analysis can be run on any scores csv with [`mdance.outputs.plot.plot_scores`](../src/mdance/outputs/plot.py).


### 5. Cluster Assignment
[**scripts/helm/assign_labels.py**](../scripts/helm/assign_labels.py) will assign cluster labels to the trajectory. The following parameters to be specified in the script:

    # System parameters
    input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
    N_atoms = 50                                        # Number of atoms in the system
    sieve = 1                                           # Use every sieve-th frame (1 = all frames)

    # HELM parameters
    n_clusters = 10                                     # Number of clusters to assign
    pre_cluster_labels = data.labels_60                 # Labels of the initial clusters (e.g. from NANI)
    pickle_file = 'outputs/helm_intra.pkl'              # HELM output from run_helm.py
    metric = 'MSD'                                      # Distance metric for clustering
    n_structures = 11                                   # Frames saved per cluster, closest to its medoid
    output_dir = 'outputs'                              # Folder for output files


#### Inputs
##### System parameters
`input_traj_numpy` is the numpy array prepared from step 1. <br>
`N_atoms` is the number of atoms used in the clustering. <br>
`sieve` is reading every `sieve` frames from the trajectory. <br>

##### HELM parameters
`n_clusters` is the number of clusters to assign labels to. Use most optimal number of clusters from analysis in step 4. <br>
`pre_cluster_labels` is the cluster labels from step 2. <br>
`pickle_file` is the clustering results from step 3. <br>
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/mdance/tools/bts.py#L94) for details). <br>
`n_structures` is the number of structures closest to the medoid to extract from each cluster. <br>
`output_dir` is the directory to store the results. <br>

#### Execution
```bash
python assign_labels.py
```

#### Outputs
All files are saved in `outputs/`, with `{n_clusters}` in the name (e.g. `n_clusters = 10`). Clusters are numbered by size, cluster 0 being the largest. <br>
`helm_labels_10.csv` contains the cluster of each frame, in frame order (frame index, cluster index). Frames not in any cluster are left out. <br>
`helm_best_frames_10.csv` contains the `n_structures` frames closest to the medoid of each cluster, closest first (frame index, cluster index). <br>
`helm_summary_10.csv` contains the population and MSD of each cluster (cluster index, fraction of total frames, number of frames, MSD); the header also gives the total number of frames. 

### 6. Extract frames for each cluster (Optional)
**This step is *optional* and for Molecular Dynamics Trajectories only.**

The last step is to extract the designated frames (e.g. `outputs/helm_best_frames_10.csv`) from the original trajectory for each cluster.

**Step-by-step tutorial can be found in the [postprocessing notebook](../scripts/outputs/postprocessing.ipynb).**

<kbd> [↩️ Main](../README.md) </kbd>