<p align="center"><img src="../docs/img/algorithms/NANI.png" alt="" width="150"></p>

# *k*-means NANI Tutorial

> New to NANI? Try the interactive [NANI in 60 seconds](https://mdance.readthedocs.io/en/latest/nani-in-60-seconds.html) first.

<a href="https://www.youtube.com/watch?v=Y2DNnLseq9E&list=PLPuFwYwFEee8"><img src="https://img.youtube.com/vi/Y2DNnLseq9E/maxresdefault.jpg" alt="NANI video tutorial" width="480"></a>

▶ [Watch the NANI video tutorial](https://www.youtube.com/watch?v=Y2DNnLseq9E&list=PLPuFwYwFEee8) (MDANCE Ep. 1, YouTube)

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

### 2. NANI Screening
[scripts/nani/screen_nani.py](../scripts/nani/screen_nani.py) will run NANI for a range of clusters and calculate cluster quality metrics. For the best result, we recommend running NANI over a wide range of number of clusters. The following parameters to be specified in the script:

    # System parameters
    input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
    N_atoms = 50                                        # Number of atoms in the system
    sieve = 1                                           # Use every sieve-th frame (1 = all frames)

    # NANI parameters
    output_dir = 'outputs'                              # Folder for output files
    init_type = 'strat_all'                             # Seeding method {'strat_all', 'strat_reduced', 'comp_sim', 'div_select', 'k-means++', 'random'}
    metric = 'MSD'                                      # Distance metric for clustering
    start_n_clusters = 5                                # At least 2 clusters
    end_n_clusters = 30                                 # Maximum number of clusters

    # Analysis parameters
    plot_metrics = True                                 # Plot DBI and CHI, and suggest values
    x_min = None                                        # Only suggest values >= x_min (None = all)

##### System parameters
`input_traj_numpy` is the numpy array prepared from step 1, if not it will be your loaded dataset. <br>
`N_atoms` is the number of atoms used in the clustering. **For all non-Molecular Dynamics datasets, this is 1.** <br>
`sieve` takes every sieve-th frame from the trajectory for analysis. <br>
##### NANI parameters
`output_dir` is the directory to store the clustering results. <br>
`init_type` is selected seed selectors (See ``mdance.cluster.nani.KmeansNANI`` for details).  <br>
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/mdance/tools/bts.py#L94) for details). <br>
`start_n_clusters` is the starting number for screening. **This number must be at least 2**.<br>
`end_n_clusters` is the ending number for screening. <br>
##### Analysis parameters
`plot_metrics` is a boolean to plot the screening scores and mark the suggested number of clusters (see step 3). If False, no analysis is done. <br>
`x_min` is the smallest number of clusters that can be suggested. The whole screening is still plotted. `None` considers all values. <br>

#### Execution
```bash
$ python screen_nani.py
```

#### Outputs
`outputs/nani_{init_type}_scores.csv` (e.g. `outputs/nani_strat_all_scores.csv`) containing the number of clusters, Calinski-Harabasz score, Davies-Bouldin score, average mean-square deviation (MSD) of the clusters, and number of iterations for that seed selector. <br>

### 3. Analysis of NANI Screening Results
The clustering screening results are analyzed using the Davies-Bouldin index (DBI) and the Calinski-Harabasz index (CHI). There are two criteria to select the number of clusters: (1) lowest DBI or highest CHI and (2) maximum 2<sup>nd</sup> derivative of DBI or minimum 2<sup>nd</sup> derivative of CHI.

When `plot_metrics = True`, `screen_nani.py` does this analysis after the screening: it saves the DBI and CHI plots next to the screening csv (e.g. `outputs/nani_strat_all_scores_dbi.png` and `outputs/nani_strat_all_scores_chi.png`) with the suggested number of clusters marked. The same analysis can be run on any screening csv with [`mdance.outputs.plot.plot_scores`](../src/mdance/outputs/plot.py).

### 4. Cluster Assignment
[scripts/nani/assign_labels.py](../scripts/nani/assign_labels.py) will assign labels to the clusters for *k*-means clustering using the initialization methods. 
The following parameters to be specified in the script:

    # System parameters
    input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
    N_atoms = 50                                        # Number of atoms in the system
    sieve = 1                                           # Use every sieve-th frame (1 = all frames)

    # NANI parameters
    n_clusters = 6                                      # Number of clusters (pick from screen_nani.py)
    init_type = 'strat_all'                             # Seeding method, same as in screen_nani.py
    metric = 'MSD'                                      # Distance metric for clustering
    n_structures = 11                                   # Frames saved per cluster, closest to its medoid
    output_dir = 'outputs'                              # Folder for output files

#### Inputs
##### System parameters
`input_traj_numpy` is the numpy array prepared from step 1, if not it will be your loaded dataset. <br>
`N_atoms` is the number of atoms used in the clustering. <br>
`sieve` takes every `sieve`th frame from the trajectory for analysis. <br>

##### *k*-means params
`n_clusters` is the number of clusters for labeling. <br>
`init_type` is selected seed selectors (See ``mdance.cluster.nani.KmeansNANI`` for details). <br>
`metric` is the metric used to calculate the similarity between frames (See [`extended_comparisons`](../src/mdance/tools/bts.py#L94) for details). <br>
`n_structures` is the number of frames to extract from each cluster. 

#### Execution
```bash
$ python assign_labels.py
```

#### Outputs
All files are saved in `outputs/`, with `{n_clusters}` in the name (e.g. `n_clusters = 6`). Clusters are numbered by size, cluster 0 being the largest.
1. `nani_best_frames_6.csv`: the `n_structures` frames closest to the medoid of each cluster (frame index, cluster index).
2. `nani_labels_6.csv`: the cluster label of each frame (frame index, cluster index).
3. `nani_summary_6.csv`: the population and MSD of each cluster (cluster index, fraction of total frames, number of frames, MSD); the header also gives the total number of frames.

### 5. Extract frames for each cluster (Optional)
[scripts/outputs/postprocessing.ipynb](../scripts/outputs/postprocessing.ipynb) will use the indices from last step (e.g. `outputs/nani_best_frames_6.csv`) to extract the designated frames from the original trajectory for each cluster.

<kbd> [↩️ Main](../README.md) </kbd>