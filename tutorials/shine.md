# SHINE - Pathway Clustering

Table of Contents
=================
- [Overview](#overview)
- [When to use SHINE](#when-to-use-shine)
- [Tutorial](#tutorial)
  - [1. Input Preparations](#1-input-preparations)
  - [2. SHINE Clustering](#2-shine-clustering)
  - [3. Analysis of Results](#3-analysis-of-results)
- [Notes and Tips](#notes-and-tips)

## Overview

Sampling Hierarchical Intrinsic *N*-ary Ensembles (SHINE) is a hierarchical clustering tool for pathway analysis. Instead of clustering individual frames, SHINE clusters whole pathways: each pathway is one trajectory, and pathways that sample similar regions end up in the same cluster. This makes it useful for identifying the dominant pathway(s) in an ensemble of enhanced sampling simulations.

**Key characteristics:**
- **Pathway-level clustering:** Each pathway (one trajectory file) is a single object in the hierarchy.
- ***n*-ary distances:** Distances between pathways are computed with the *n*-ary similarity framework.
- **Optional sampling:** Long pathways can be reduced to a representative subset of frames before comparison.
- **Dendrogram output:** Results can be visualized as a dendrogram of the pathways.

## When to use SHINE

SHINE is ideal for:
- **Enhanced sampling simulations:** Finding the dominant pathway(s) among many transition pathways.
- **Comparing pathway ensembles:** Grouping pathways that follow similar routes through conformational space.

## Tutorial

### 1. Input Preparations

SHINE takes one array per pathway, with shape `(n_frames, n_features)`. All pathways must share the same features (same number of columns), but they can have different numbers of frames.

Store one file per pathway, with the pathway index as the last number in the file name (for example, `pathway_0.npy`, `pathway_1.npy`, ...). Both `.npy` and `.csv` files work with the code in this tutorial.

**Example with included data:**
MDANCE includes 80 alanine dipeptide pathways (`pathway_dih_0.csv` to `pathway_dih_79.csv`), with the phi and psi dihedral angles of each frame. They are available at `data.ala_pathways` and are used in this tutorial.

**For Molecular Dynamics trajectories:**
Convert each pathway's trajectory to a numpy array (one file per pathway) as shown in [scripts/inputs/preprocessing.ipynb](../scripts/inputs/preprocessing.ipynb). **Make sure the trajectories are already aligned and/or centered if needed!**

### 2. SHINE Clustering

A runnable script is also provided in [`scripts/shine/run_shine.py`](../scripts/shine/run_shine.py).

The following code loads the pathways and runs SHINE:

```python
import glob

import matplotlib.pyplot as plt
import numpy as np

from mdance import data
from mdance.cluster.shine import Shine

# System parameters - EDIT THESE
file_pattern = str(data.ala_pathways / '*.csv')     # One file per pathway
metric = 'MSD'                                      # Metric used to compare pathways
N_atoms = 1                                         # Number of atoms (1 for non-coordinate features)

# SHINE parameters - EDIT THESE
merge_scheme = 'intra'                              # {'intra', 'inter', 'semi_sum', 'max', 'min', 'haus'}
sampling = 'diversity'                              # {'diversity', 'quota', None}
frac = 0.5                                          # Fraction of frames sampled from each pathway
link = 'ward'                                       # Linkage method for hierarchical clustering
criterion = 'maxclust'                              # Criterion for forming flat clusters
t = 2                                               # Maximum number of clusters (for 'maxclust')

# Load the pathways as (pathway index, frames) pairs
trajs = []
for file in glob.glob(file_pattern):
    traj_idx = file.split('_')[-1].split('.')[0]
    frames = np.load(file) if file.endswith('.npy') else np.genfromtxt(file, delimiter=',')
    trajs.append((traj_idx, frames))

# Run SHINE
mod = Shine(trajs, metric, N_atoms=N_atoms, t=t, criterion=criterion, link=link,
            merge_scheme=merge_scheme, sampling=sampling, frac=frac)
link_matrix, clusters = mod.run()
```

#### Inputs

##### System parameters
`file_pattern` is the pattern matching the pathway files. The pathway index is read from the last number in each file name. <br>
`metric` is the metric used to compare pathways (see [`extended_comparison`](../src/mdance/tools/bts.py#L94) for the options). <br>
`N_atoms` is the number of atoms in each frame. Use `1` for features that are not atom coordinates, such as the dihedral angles in the example data. <br>

##### SHINE parameters
`merge_scheme` is how the distance between two pathways is computed from their frames. `intra` uses the *n*-ary similarity of the two pathways combined; the other options subtract or compare the similarities of the individual pathways, and `haus` uses the Hausdorff distance. <br>
`sampling` is how frames are sampled from each pathway before comparing them. `diversity` uses diversity selection, `quota` picks frames evenly across bins of complementary similarity, and `None` uses all frames. <br>
`frac` is the fraction of frames sampled from each pathway. Pathways with fewer than 50 frames are always used in full. <br>
`link` is the linkage method passed to [`scipy.cluster.hierarchy.linkage`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html). <br>
`criterion` and `t` decide how the hierarchy is cut into flat clusters (see [`scipy.cluster.hierarchy.fcluster`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html)). With `criterion = 'maxclust'`, `t` is the maximum number of clusters. <br>

#### Outputs
`link_matrix` is the hierarchical clustering of the pathways, encoded as a scipy linkage matrix. <br>
`clusters` is the cluster number of each pathway, in the order of `mod.pathways`. <br>

### 3. Analysis of Results

List the pathways in each cluster:

```python
for cluster, pathways in zip(np.unique(clusters), mod.labels()):
    print(f'Cluster {cluster}: pathways {pathways}')
```

Plot the dendrogram of the pathways, with a legend listing the pathways in each cluster:

```python
ax = mod.plot()
plt.savefig(f'shine_dendrogram_{merge_scheme}.png', bbox_inches='tight', dpi=300)
plt.show()
```

Use the dendrogram to choose the number of clusters: change `t` and run SHINE again to cut the hierarchy at a different level.

## Notes and Tips

- **Alignment:** SHINE compares coordinates directly, so MD trajectories should be aligned to a common reference before conversion.
- **Pathway indices:** The index of each pathway comes from its file name, so make sure every file name ends with a unique number.
- **Sampling:** Sampling makes long pathways cheaper to compare. Use `sampling = None` to compare all frames.
