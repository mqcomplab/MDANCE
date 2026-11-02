# PRIME Tutorial

#### Table of Contents
- [Overview](#overview)
- [Tutorial](#tutorial)
  - [1. Cluster Trajectories](#1-cluster-trajectories)
  - [2. Cluster Normalization](#2-cluster-normalization)
  - [3. Similarity Calculations](#3-similarity-calculations)
  - [4. Representative Frames](#4-representative-frames)
- [Further Reading](#further-reading)

## Overview
This clustering tutorial is meant for datasets Molecular Dynamics Trajectory. PRIME assumes a MD trajectory that has a well-sampled ensemble of conformations. The PRIME algorithm predicts the native structure of a protein from simulation or clustering data. These methods perfectly mapped all the structural motifs in the studied systems and required unprecedented linear scaling.

## Tutorial
The following tutorial will guide you through the process of determining the native structure of a biomolecule using the PRIME algorithm. If you do not have clustered data. Please refer to other clustering algorithms such as [NANI](nani.md) to cluster your data. If you already have clustered data, please skip to step 2.

### 1. Cluster Trajectories
PRIME needs two files from a clustering algorithm such as [NANI](nani.md):
- a labels file with the cluster of each frame (frame index, cluster index), with clusters numbered by size (cluster 0 being the largest), e.g. `nani_labels_6.csv` from `scripts/nani/assign_labels.py`;
- a summary file with the population of each cluster (cluster index, fraction of total frames), e.g. `nani_summary_6.csv`.

The scripts below use the samples included with MDANCE, `mdance.data.labels_6` and `mdance.data.summary_6`. Run the steps in order from the `scripts/prime` directory; each step uses the outputs of the previous one.

### 2. Cluster Normalization
[scripts/prime/normalize.py](../scripts/prime/normalize.py) With already clustered data, this script will normalize the trajectory data between $[0,1]$ using the Min-Max Normalization. 

    # System parameters
    input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
    cluster_labels = data.labels_6                      # Labels file from a clustering script
    output_dir = 'outputs/normed_clusters'              # Folder for the normalized cluster files
    output_base_name = 'normed_clusttraj'               # Prefix for the normalized cluster files
    n_clusters = 6                                      # Number of clusters in the labels file

#### Inputs
##### System parameters
`input_traj_numpy` is the numpy array used in the clustering. <br>
`cluster_labels` is the labels file from step 1. <br>
`output_dir` is the directory where the normalized clustering files will be saved. <br>
`output_base_name` is the base name for the output files. <br>
`n_clusters` is the number of clusters used in the PRIME. If number less than total number of cluster, it will take top *n* number of clusters. <br>

```bash
$ python normalize.py
```

#### Outputs
1. `outputs/normed_clusters/normed_clusttraj.c*.npy` files, normalized clustering files.
2. `outputs/normed_clusters/normed_data.npy`, appended all normed files together.

### 3. Similarity Calculations
`prime_sim` generates a similarity dictionary from running PRIME. 

- `-h` - for help with the argument options.
- `-m` - methods, pairwise, union, medoid, outlier (*required*).
- `-n` - number of clusters (*required*).
- `-i` - similarity index, RR or SM (*required*).
- `-t` - Fraction of outliers to trim in decimals (default is None).
- `-w` - Weighing clusters by frames it contains (default is True).
- `-d` - directory where the `normed_clusttraj.c*.npy` files are located (*required*)
- `-s` - location where `summary` file is located with population of each cluster (*required*)

#### Example 
```bash
$ prime_sim -m union -n 6 -i SM -t 0.1 -d outputs/normed_clusters -s ../nani/outputs/nani_summary_6.csv
```

To generate a similarity dictionary using the normalized clusters from step 2 (make sure you are in the prime directory) using the union method (2.2 in *Fig 2*) and Sokal Michener index. In addition, 10% of the outliers were trimmed. You can either `python exec_similarity.py` (which uses the sample summary `mdance.data.summary_6`) or run the example above.

#### Outputs
`outputs/w_union_SM_t10.txt` file with the similarity dictionary.
The result is a dictionary organized as followes:
Keys are frame #. Values are [cluster 1 similarity, cluster #2 similarity, ..., average similarity of all clusters].

### 4. Representative Frames
`prime_rep` will determine the native structure of the protein using the similarity dictionary generated in step 3.

-`h` - for help with the argument options.
-`m` - methods (for one method, None for all methods)
-`s` - folder to access for `w_union_SM_t10.txt` file
-`i` - similarity index (*required*)
-`t` - Fraction of outliers to trim in decimals (default is None).
-`d` - directory where the `normed_clusttraj.c*` files are located (required if method is None)

#### Example 
```bash
$ prime_rep -m union -s outputs -d outputs/normed_clusters -t 0.1 -i SM
```

#### Outputs
`outputs/w_rep_SM_t10_union.txt` file with the representative frames index. You can either `python exec_rep_frames.py` or run the example above.

## Further Reading
For more information on the PRIME algorithm, please refer to the [PRIME paper](https://doi.org/10.1021/acs.jctc.4c00362). 

Please Cite
```bibtex
@article{chen_protein_2024,
    title = {Protein {Retrieval} via {Integrative} {Molecular} {Ensembles} ({PRIME}) through {Extended} {Similarity} {Indices}},
    author = {Chen, Lexin and Mondal, Arup and Perez, Alberto and Miranda-Quintana, Ramón Alain},
    journal = {Journal of Chemical Theory and Computation},
    volume = {20},
    number = {14},
    pages = {6303--6315},
    year = {2024},
    doi = {10.1021/acs.jctc.4c00362},
    url = {https://doi.org/10.1021/acs.jctc.4c00362},
}
```

<img src="../docs/img/methods.jpg" alt="methods" width="500" height=auto align="center"></a>

*Fig 2. Six techniques of protein refinement. Blue is top cluster.*