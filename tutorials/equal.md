<p align="center"><img src="../docs/img/algorithms/eQual.png" alt="" width="150"></p>

# eQual - Extended Quality Clustering

<a href="https://www.youtube.com/watch?v=cjlNN8qhpOw&list=PLPuFwYwFEee8"><img src="https://img.youtube.com/vi/cjlNN8qhpOw/maxresdefault.jpg" alt="eQual video tutorial" width="480"></a>

▶ [Watch the eQual video tutorial](https://www.youtube.com/watch?v=cjlNN8qhpOw&list=PLPuFwYwFEee8) (MDANCE Ep. 7, YouTube)

Table of Contents
=================
- [Overview](#overview)
- [Tutorial](#tutorial)
    - [1. Input Preparations](#1-input-preparations)
    - [2. eQual Screening](#2-equal-screening)
    - [3. eQual Screening Analysis](#3-equal-screening-analysis)
    - [4. Assign labels to the frames](#4-assign-labels-to-the-frames)
    - [5. Extract frames for each cluster (Optional)](#5-extract-frames-for-each-cluster-optional)

## Overview
eQual selects the seed to start the clustering. It then grows the cluster by adding the neighbors within a threshold away from the seed. This threshold is calculated using the mean-square deviation from the seed. The iteration continues until it runs out of neighbors and chosen neighbors will be removed from orginal dataset. A new iteration begins and selects medoid and its neighbors from the available dataset. If user selects multiple medoids, then the medoid that proposed the densest and most similar cluster will be chosen. The process repeats and the algorithm terminates when the dataset is empty.

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

### 2. eQual Screening
[**scripts/equal/screen_equal.py**](../scripts/equal/screen_equal.py) will screen eQual clustering for multiple thresholds and give the most optimal threshold. For the best result, we recommend screening eQual with a wide range of threshold values. <br> 
*Depending on the number of samples or features, consider sieving over wide threshold range. For large dataset, please submit this as a job instead of running on command line.* <br>
The following parameters to be specified in the script:

    # System parameters
    input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
    N_atoms = 50                                        # Number of atoms in the system
    sieve = 1                                           # Use every sieve-th frame (1 = all frames)

    # eQual parameters
    metric = 'MSD'                                      # Distance metric for clustering
    n_seeds = 3                                         # Seeds tried per iteration (int, or fraction of the data)
    check_sim = True                                    # Reject clusters less similar than sim_threshold
    reject_lowd = True                                  # Reject clusters smaller than min_samples
    sim_threshold = 16                                  # Largest similarity value a cluster may have
    min_samples = 10                                    # Minimum cluster size

    # Screening parameters
    start_threshold = 5                                 # First threshold to screen
    end_threshold = 6                                   # Last threshold to screen
    step = 0.1                                          # Step between thresholds
    save_clusters = False                               # Save the clusters of each threshold as .pkl
    output_dir = 'outputs'                              # Folder for output files

    # Analysis parameters
    plot_metrics = True                                 # Plot DBI and CHI, and suggest values
    x_min = None                                        # Only suggest values >= x_min (None = all)

#### Inputs

##### System parameters
`input_traj_numpy` is the numpy array prepared from step 1, if not it will be your loaded dataset. <br>
`N_atoms` is the number of atoms used in the clustering. **For all non-Molecular Dynamics datasets, this is 1.** <br>
`sieve` takes every sieve-th frame from the trajectory for analysis. <br>

##### eQual params
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/mdance/tools/bts.py#L94) for details). <br>
`n_seeds` is the is the number of seeds selected per iteration. If `n_medoids` is greater than 1, then multiple clusters will be proposed; the cluster with the densest and greatest similarity of members will be selected. Performance time will increase with more seeds. <br>
`check_sim` is boolean to check the similarity of the seed to the cluster. 
`reject_lowd` is boolean to reject low density clusters. `sim_threshold` needs to be specified. <br>
`sim_threshold` is the similarity threshold to reject less compact clusters. <br>
`min_samples` is the minimum cluster size to reject low density clusters. Default is 10. <br>

##### Radial threshold screening params
`start_threshold` is the starting value `r_theshold` for screening range. <br>
`end_threshold` is the ending value of `r_theshold` screening range. <br>
`step` is the increment of the `r_theshold` screening range. <br>
`save_clusters` is boolean to save the cluster dictionary of each threshold in `outputs/clusters_dict/`. Default is False. <br>
`output_dir` is the directory to store the results. <br>

##### Analysis parameters
`plot_metrics` is boolean to plot the screening results and mark the suggested thresholds (see step 3). If False, no analysis is done. <br>
`x_min` is the smallest threshold that can be suggested. The whole screening is still plotted. `None` considers all values. <br>

#### Execution
```bash
$ python screen_equal.py
```

#### Outputs
- `outputs/equal_screening.csv` with the number of clusters and the population of the top 10 clusters for each threshold value.
- `outputs/equal_scores.csv` with the number of clusters, Calinski-Harabasz (CH) score, Davies-Bouldin (DB) score (two cluster quality indices), and average MSD of the clusters for each threshold value.
- `outputs/clusters_dict/equal_threshold_{threshold}.pkl` with the clusters of each threshold, if `save_clusters = True`.

### 3. eQual Screening Analysis
When `plot_metrics = True`, `screen_equal.py` analyzes the screening results after the screening:
- `outputs/equal_scores_dbi.png` and `outputs/equal_scores_chi.png` plot the Davies-Bouldin index (DBI) and the Calinski-Harabasz index (CHI) for each threshold. The suggested thresholds (lowest DBI or highest CHI, and the sharpest 2<sup>nd</sup> derivatives) are marked.
- `outputs/equal_screening_nclusters.png` plots the number of clusters for each threshold and marks the threshold with the most clusters.

The same analysis can be run on any screening csv with [`mdance.outputs.plot`](../src/mdance/outputs/plot.py) (`plot_scores` and `plot_n_clusters`).

### 4. Assign labels to the frames
[**scripts/equal/assign_labels.py**](../scripts/equal/assign_labels.py) will assign cluster for each frame. The following parameters to be specified in the script:

    # System parameters
    input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
    N_atoms = 50                                        # Number of atoms in the system
    sieve = 1                                           # Use every sieve-th frame (1 = all frames)

    # eQual parameters
    metric = 'MSD'                                      # Distance metric for clustering
    n_seeds = 3                                         # Seeds tried per iteration (int, or fraction of the data)
    check_sim = True                                    # Reject clusters less similar than sim_threshold
    reject_lowd = True                                  # Reject clusters smaller than min_samples
    sim_threshold = 16                                  # Largest similarity value a cluster may have
    min_samples = 10                                    # Minimum cluster size

    # Output parameters
    threshold = 5.80                                    # Cluster radius threshold (pick from screen_equal.py)
    n_structures = 11                                   # Frames saved per cluster, closest to its medoid
    open_clusters = None                                # Saved clusters .pkl to reuse (None = recluster)
    output_dir = 'outputs'                              # Folder for output files

#### Inputs - New parameters
`threshold` is the threshold value to use for clustering, e.g. chosen from the screening plots. <br>
`n_structures` is the number of closest structures (from the medoid) to extract from each cluster. <br>
`open_clusters` is a saved cluster dictionary to open (e.g. `outputs/clusters_dict/equal_threshold_5.8.pkl` from the screening with `save_clusters = True`). If `None`, it will run the clustering algorithm. <br>
`output_dir` is the directory to store the results. <br>

#### Execution
```bash
python assign_labels.py
```

#### Outputs
All files are saved in `outputs/`, with the `threshold` in the name (e.g. `threshold = 5.8`). Clusters are numbered by size, cluster 0 being the largest. <br>
`equal_best_frames_5.8.csv` contains the `n_structures` most representative frames of each cluster, closest to the medoid first (frame index, cluster index). <br>
`equal_labels_5.8.csv` contains the cluster of each frame, in frame order (frame index, cluster index). Frames rejected as low density are not assigned to any cluster and are left out. <br>
`equal_summary_5.8.csv` contains the population and MSD of each cluster (cluster index, fraction of total frames, number of frames, MSD); the header also gives the total number of frames. <br>

### 5. Extract frames for each cluster (Optional)
**This step is *optional* and for Molecular Dynamics Trajectories only.**

The last step is to extract the designated frames (e.g. `outputs/equal_best_frames_5.8.csv`) from the original trajectory for each cluster.

**Step-by-step tutorial can be found in the [postprocessing notebook](../scripts/outputs/postprocessing.ipynb).**

<kbd> [↩️ Main](../README.md) </kbd>