# *k*-means NANI Tutorial

Table of Contents
=================
- [Overview](#overview)
- [Tutorial](#tutorial)
    - [1. Input Preparations](#1-input-preparations)
    - [2. NANI Screening](#2-nani-screening)
    - [3. Analysis of NANI Screening Results](#3-analysis-of-nani-screening-results)
    - [4. Cluster Assignment](#4-cluster-assignment)
    - [5. Extract frames for each cluster (Optional)](#5-extract-frames-for-each-cluster-optional)

## Overview
This clustering tutorial is meant for datasets for **all** applications (2D fingerprints, mass spectrometry imaging data, etc). Molecular Dynamics Trajectory has a different treatment. If specific step is only for Molecular Dynamics trajectory, it will be specified. Otherwise, it is applicable for all datasets.

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

### 2. NANI Screening
[scripts/nani/screen_nani.py](../scripts/nani/screen_nani.py) will run NANI for a range of clusters and calculate cluster quality metrics. For the best result, we recommend running NANI over a wide range of number of clusters. The following parameters to be specified in the script:

    # System info
    input_traj_numpy = data.sim_traj_numpy
    N_atoms = 50
    sieve = 1

    # NANI parameters
    output_dir = 'outputs'                        
    init_types = ['comp_sim']
    metric = 'MSD'
    start_n_clusters = 2
    end_n_clusters = 30

##### System info
`input_traj_numpy` is the numpy array prepared from step 1, if not it will be your loaded dataset. <br>
`N_atoms` is the number of atoms used in the clustering. **For all non-Molecular Dynamics datasets, this is 1.** <br>
`sieve` takes every sieve-th frame from the trajectory for analysis. <br>
##### NANI parameters
`output_dir` is the directory to store the clustering results. <br>
`init_types` is a **list** of selected seed selectors. User can input one or multiple. Each seed selector will have results in a separate file. <br>
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/tools/bts.py) for details). <br>
`start_n_clusters` is the starting number for screening. **This number must be greater than 2**.<br>
`end_n_clusters` is the ending number for screening. <br>

#### Execution
```bash
$ python screen_nani.py
```

#### Outputs
csv file containing the number of clusters and the corresponding number of iterations, Callinski-Harabasz score, Davies-Bouldin score, and average mean-square deviation for that seed selector. <br>

### 3. Analysis of NANI Screening Results
The clustering screening results will be analyzed using the Davies-Bouldin index (DB). There are two criteria to select the number of clusters: (1) lowest DB and (2) maximum 2<sup>nd</sup> derivative of DB.

**Step-by-step tutorial can be found in the [analysis notebook](../scripts/nani/analysis_db.ipynb).**

### 4. Cluster Assignment
[scripts/nani/assign_labels.py](../scripts/nani/assign_labels.py) will assign labels to the clusters for *k*-means clustering using the initialization methods. 
The following parameters to be specified in the script:

    # System info - EDIT THESE
    input_traj_numpy = '../../data/md/backbone.npy'
    N_atoms = 50
    sieve = 1

    # K-means params - EDIT THESE
    n_clusters = 6
    init_type = 'comp_sim'                                              
    metric = 'MSD'                                                      
    n_structures = 11                                                   
    output_dir = 'outputs'                                              

#### Inputs
##### System info
`input_traj_numpy` is the numpy array prepared from step 1, if not it will be your loaded dataset. <br>
`N_atoms` is the number of atoms used in the clustering. <br>
`sieve` takes every `sieve`th frame from the trajectory for analysis. <br>

##### *k*-means params
`n_clusters` is the number of clusters for labeling. <br>
`init_type` is the seed selector to use. <br>
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/tools/bts.py) for details). <br>
`n_structures` is the number of frames to extract from each cluster. 

#### Execution
```bash
$ python assign_labels.py
```

#### Outputs
1. csv file containing the indices of the best frames in each cluster. 
2. csv file containing the cluster labels for each frame.
3. csv file containing the population of each cluster.

### 5. Extract frames for each cluster (Optional)
[scripts/outputs/postprocessing.ipynb](../scripts/outputs/postprocessing.ipynb) will use the indices from last step to extract the designated frames from the original trajectory for each cluster.

<kbd> [↩️ Main](../README.md) </kbd>