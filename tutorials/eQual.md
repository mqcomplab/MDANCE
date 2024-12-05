# eQual - Extended Quality Clustering

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

### 2. eQual Screening
[**scripts/equal/screen_equal.py**](../scripts/equal/screen_equal.py) will screen eQual clustering for multiple thresholds and give the most optimal threshold. For the best result, we recommend screening eQual with a wide range of threshold values. <br> 
*Depending on the number of samples or features, consider sieving over wide threshold range. For large dataset, please submit this as a job instead of running on command line.* <br>
The following parameters to be specified in the script:

    # System info - EDIT THESE
    input_traj_numpy = data.sim_traj_numpy
    N_atoms = 50
    sieve = 1

    # eQUAL params - EDIT THESE
    metric = 'MSD'                                                      # Default
    n_seeds = 3
    check_sim = True                                                    # Default
    reject_lowd = True                                                  # Default
    sim_threshold = 16
    min_samples = 10                                                    # Default

    # thresholds params- EDIT THESE
    start_threshold = 5
    end_threshold = 6
    step = 0.1
    save_clusters = False                                                # Default False

#### Inputs

##### System info
`input_traj_numpy` is the numpy array prepared from step 1, if not it will be your loaded dataset. <br>
`N_atoms` is the number of atoms used in the clustering. **For all non-Molecular Dynamics datasets, this is 1.** <br>
`sieve` takes every sieve-th frame from the trajectory for analysis. <br>

##### eQual params
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/mdance/tools/bts.py#L92) for details). <br>
`n_seeds` is the is the number of seeds selected per iteration. If `n_medoids` is greater than 1, then multiple clusters will be proposed; the cluster with the densest and greatest similarity of members will be selected. Performance time will increase with more seeds. <br>
`check_sim` is boolean to check the similarity of the seed to the cluster. 
`reject_lowd` is boolean to reject low density clusters. `sim_threshold` needs to be specified. <br>
`sim_threshold` is the similarity threshold to reject less compact clusters. <br>
`min_samples` is the minimum cluster size to reject low density clusters. Default is 10. <br>

##### Radial threshold screening params
`start_threshold` is the starting value `r_theshold` for screening range. <br>
`end_threshold` is the ending value of `r_theshold` screening range. <br>
`step` is the increment of the `r_theshold` screening range. <br>
`save_clusters` is boolean to save the cluster dictionary. Default is False. <br>

#### Execution
```bash
$ python screen_equal.py
```

#### Outputs
- a csv with the number of clusters, cluster population for each threshold value. 
- a csv with the Calinski-Harabasz (CH) score and Davies-Bouldin (DB) score (two cluster quality indices) for each threshold value.

### 3. eQual Screening Analysis
[**scripts/equal/analysis.ipynb**](../scripts/equal/analysis_db.ipynb.ipynb) will analyze the eQual screening results. 

**Step-by-step tutorial can be found in the [analysis notebook](../scripts/equal/analysis_db.ipynb).**

### 4. Assign labels to the frames
[**scripts/equal/assign_labels.py**](../scripts/equal/assign_labels.py) will assign cluster for each frame. The following parameters to be specified in the script:

    # System info - EDIT THESE
    input_traj_numpy = data.sim_traj_numpy
    N_atoms = 50
    sieve = 1

    # eQUAL params - EDIT THESE
    metric = 'MSD'                                                      # Default 
    n_seeds = 3                                                         # Default
    check_sim = True                                                    # Default
    reject_lowd = True                                                  # Default
    sim_threshold = 16
    min_samples = 10                                                    # Default

    # extract params- EDIT THESE
    threshold = 5.80
    n_structures = 11                                                   # Default
    sorted_by = 'frame'                                                 # Default
    open_clusters = None                                                # Default   

#### Inputs - New parameters
`threshold` is desired threshold value to use for clustering. If `None`, it will use the best threshold value by reading `param_file`. <br>
`n_structures` is the number of closest structure (from medoid) to extract from each cluster. <br>
`sorted_by` is the sorting method for the cluster labels. {'frame', 'cluster'}. Either frames or clusters can be sorted by ascending order. Default is 'frame'. <br>
`open_cluster_dict` is the cluster dictionary file to open. If `None`, it will run the clustering algorithm. <br>

#### Execution
```bash
python assign_labels.py
```

#### Outputs
`best_frames_indices.csv` contains the top *n* number (`n_structures`) of most representative frames for each of the top clusters (`top_num_cluster`). <br>
`frame_vs_cluster.csv` contains cluster assignment per frame. <br>
`sorted_by="frame"` will sort `frame_vs_cluster.csv` by ascending frame number. `sorted_by="cluster"` will sort by ascending cluster number. <br>

### 5. Extract frames for each cluster (Optional)
**This step is *optional* and for Molecular Dynamics Trajectories only.**

The last step is to extract the designated frames from the original trajectory for each cluster.

**Step-by-step tutorial can be found in the [postprocessing notebook](../scripts/outputs/postprocessing.ipynb).**

<kbd> [↩️ Main](../README.md) </kbd>