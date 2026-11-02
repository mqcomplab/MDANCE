# DIVINE - Deterministic Divisive Hierarchical Clustering

Table of Contents
=================
- [Overview](#overview)
- [When to use DIVINE](#when-to-use-divine)
- [Tutorial](#tutorial)
  - [1. Input Preparations](#1-input-preparations)
  - [2. DIVINE Clustering](#2-divine-clustering)
  - [3. DIVINE Outputs](#3-divine-outputs)
  - [4. Analysis of Results](#4-analysis-of-results)
  - [5. Assign Labels to Frames (Optional)](#5-assign-labels-to-frames-optional)
- [Notes and Tips](#notes-and-tips)

## Overview

DIVINE (Deterministic Divisive Hierarchical Clustering) is a top-down hierarchical clustering algorithm specifically designed for molecular dynamics analysis. Unlike traditional agglomerative approaches, DIVINE builds the hierarchy by recursively splitting the dataset into finer clusters using deterministic anchor initialization (NANI).

**Key characteristics:**
- **Top-down approach:** Starts with all data and recursively splits into smaller clusters
- **Deterministic:** Uses NANI anchors for reproducible results
- **Efficient:** Completely avoids O(*N²*) pairwise distance matrices
- **Flexible stopping:** Supports fixed cluster count or singleton termination
- **Multi-resolution:** Generate hierarchical trees at different granularities
- **Quality metrics:** Provides Davies-Bouldin (DB) and Calinski-Harabasz (CH) scores at each resolution level

## When to use DIVINE

DIVINE is ideal for:
- **Multi-resolution clustering:** When you want to explore cluster hierarchies at different levels.
- **Large-scale MD datasets:** Analyzing trajectories with hundreds of thousands to millions of frames.
- **Reproducible clustering:** When deterministic results are necessary for comparative studies.
- **Rare event detection:** Identifying sparse conformational states through hierarchical refinement.

## Tutorial

### 1. Input Preparations

Prepare your trajectory data as a numpy array with shape `(n_frames, n_features)`.

**For Molecular Dynamics trajectories:**

Prepare a valid topology file (e.g., `.pdb`, `.prmtop`), trajectory file (e.g., `.dcd`, `.nc`), and atom selection. Convert to aligned/centered coordinates as needed.

**Step-by-step tutorial:** [scripts/inputs/preprocessing.ipynb](../scripts/inputs/preprocessing.ipynb)

**For other datasets (optional normalization):**

If you're using a metric other than mean-square deviation (MSD), normalize your dataset using [scripts/inputs/normalize.py](../scripts/inputs/normalize.py).

**Example with included data:**
The alanine dipeptide trajectory is available at `data.sim_traj_numpy` or can be loaded directly:
```python
import numpy as np
full_traj = np.load('path/to/trajectory.npy')
```

### 2. DIVINE Clustering

A runnable example is provided in:
- [`scripts/divine/run_divine.py`](../scripts/divine/run_divine.py)

This script loads your trajectory and performs top-down hierarchical clustering.

#### Parameters to edit in `run_divine.py`

**System parameters:**
```python
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
N_atoms = 50                                        # Number of atoms in the system
sieve = 1                                           # Use every sieve-th frame (1 = all frames)
```

- `input_traj_numpy` : Path to the numpy array (`.npy` file) from step 1
- `N_atoms` : Number of atoms in your system. **For non-MD datasets, use 1**
- `sieve` : Frame sampling rate (e.g., `10` skips every 10th frame for faster analysis)

**DIVINE parameters:**
```python
split = 'weighted_MSD'                              # Cluster to split next {'MSD', 'radius', 'weighted_MSD'}
anchors = 'nani'                                    # Seed choice for each split {'nani', 'outlier_pair', 'splinter_split'}
init_type = 'strat_all'                             # NANI seeding method, used when anchors = 'nani'
k = 30                                              # Keep splitting until there are k clusters
threshold = 0                                       # Minimum subcluster size, as a fraction of the data
refine = True                                       # Refine each split with k-means
output_dir = 'outputs'                              # Folder for output files
```

**Analysis parameters:**
```python
plot_metrics = True                                 # Plot DBI and CHI, and suggest values
x_min = 5                                           # Only suggest values >= x_min (None = all)
```

**Parameter explanations:**

- `split` : Cluster selection strategy (which cluster to split next)
  - `'weighted_MSD'` : Selects cluster with highest variance (recommended)
  - `'MSD'` : Mean-square deviation based selection
  - `'radius'` : Radius-based selection

- `anchors` : How to select split points
  - `'nani'` : Use NANI deterministic initialization (recommended)
  - `'outlier_pair'` : Use most distant pair
  - `'splinter_split'` : Use splinter detection

- `init_type` : Seed selection method (only for `anchors='nani'`)
  - `'strat_all'` : Stratified sampling (recommended)
  - `'strat_reduced'` : Reduced stratified sampling
  - `'comp_sim'` : Comprehensive similarity
  - `'div_select'` : Diversity selection
  - `'k-means++'` : Standard k-means++ initialization
  - `'random'` : Random seed selection

- `threshold` : Reject clusters smaller than `threshold * total_size`
  - `0` : No filtering (keep all)
  - `0.05` : Reject clusters < 5% of total size

- `refine` : Refine each split with an additional *k*-means clustering (default `True`). Only used with `anchors='outlier_pair'` or `'splinter_split'`; `'nani'` anchors already split with *k*-means, so `refine` has no effect there

- `output_dir` : Directory to store the results

- `plot_metrics` : Plot the cluster quality metrics and mark the suggested number of clusters (see step 4). If `False`, no analysis is done.

- `x_min` : Smallest number of clusters that can be suggested. The whole hierarchy is still plotted. `None` considers all values.

#### Execution

```bash
cd scripts/divine
python run_divine.py
```

The script will run hierarchically, splitting clusters until reaching `k` clusters, and report progress to the console.

### 3. DIVINE Outputs

The script generates two output files in `outputs/` (named after `anchors`, e.g. `nani`):

**Cluster quality metrics:**
- `outputs/divine_nani_scores.csv`
  - Columns: `Number of clusters`, `Calinski-Harabasz`, `Davies-Bouldin`, `Average MSD`
  - Each row represents cluster counts at different hierarchy levels
  - **CH score** (Calinski-Harabasz): Higher is better (max ~∞)
  - **DB score** (Davies-Bouldin): Lower is better (min ~0)

**Cluster structure metrics:**
- `outputs/divine_nani_msds.csv`
  - Columns: `Level`, `Cluster Index`, `Number of frames`, `MSD` (level *L* has *L* + 1 clusters)
  - Tracks splitting history and cluster properties
  - `level` : Hierarchical depth of the split
  - `cluster_index` : Identifier for the cluster
  - `population` : Number of frames in the cluster
  - `msd` : Mean-square deviation (cluster tightness)

### 4. Analysis of Results

When `plot_metrics = True`, `run_divine.py` plots the DB and CH scores of each number of clusters (`outputs/divine_nani_scores_dbi.png` and `outputs/divine_nani_scores_chi.png`) with the suggested number of clusters marked. The same analysis can be run on any scores csv with [`mdance.outputs.plot.plot_scores`](../src/mdance/outputs/plot.py).

Use the cluster quality metrics (DB and CH scores) to determine the optimal number of clusters:

1. **Lowest DB score:** Points to the most compact, well-separated clustering
2. **Maximum 2nd derivative of DB:** Identifies the "elbow" where clustering quality plateaus
3. **CH score:** Complements DB; higher values indicate better-defined clusters

**Interpretation:**
- Large jump in DB score between consecutive levels = significant clustering improvement
- Plateauing DB score = diminishing returns from further splitting

### 5. Assign Labels to Frames (Optional)

Once you've determined the optimal `k`, use [scripts/divine/assign_labels.py](../scripts/divine/assign_labels.py) to generate frame-level cluster assignments.

#### Parameters to edit in `assign_labels.py`

```python
# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
N_atoms = 50                                        # Number of atoms in the system
sieve = 1                                           # Use every sieve-th frame (1 = all frames)

# DIVINE parameters
split = 'weighted_MSD'                              # Cluster to split next {'MSD', 'radius', 'weighted_MSD'}
anchors = 'nani'                                    # Seed choice for each split {'nani', 'outlier_pair', 'splinter_split'}
init_type = 'strat_all'                             # NANI seeding method, used when anchors = 'nani'

k = 6                                               # Number of clusters to assign
threshold = 0                                       # Minimum subcluster size, as a fraction of the data
refine = True                                       # Refine each split with k-means

top_n = 11                                          # Frames saved per cluster, closest to its medoid
output_dir = 'outputs'                              # Folder for output files
```

Use the same settings as in `run_divine.py`, with `k` set to the number of clusters chosen in step 4.

#### Execution

```bash
python assign_labels.py
```

#### Output

All files are saved in `outputs/`, with `k` in the name (e.g. `k = 6`). Clusters are numbered by size, cluster 0 being the largest.
- `divine_labels_6.csv`: the cluster of each frame, in frame order (frame index, cluster index)
- `divine_best_frames_6.csv`: the `top_n` frames closest to the medoid of each cluster, closest first (frame index, cluster index)
- `divine_summary_6.csv`: the population and MSD of each cluster (cluster index, fraction of total frames, number of frames, MSD); the header also gives the total number of frames

### 6. Extract Frames for Each Cluster (Optional - MD only)

After obtaining cluster assignments, extract representative frames from each cluster (e.g. `outputs/divine_best_frames_6.csv`).

**Step-by-step tutorial:** [scripts/outputs/postprocessing.ipynb](../scripts/outputs/postprocessing.ipynb)

## Notes and Tips

- **Determinism:** DIVINE with NANI anchors produces reproducible results across runs due to deterministic seed selection.

- **Computational efficiency:** DIVINE avoids pairwise distance matrices, scaling nearly linearly with dataset size.

- **Parameter tuning:**
  - Start with `split='weighted_MSD'` and `anchors='nani'` (works well for most cases)
  - Increase `k` for finer structural detail; decrease for coarser grouping
  - Use `threshold > 0` to filter out noise-like tiny clusters

- **Quality scores interpretation:**
  - Davies-Bouldin (DB) : Ratio of within-cluster to between-cluster distances
  - Calinski-Harabasz (CH) : Ratio of between-cluster to within-cluster variance
  - Together they provide complementary views of clustering quality

- **Large datasets:**
  - Use `sieve > 1` for fast screening (e.g., `sieve=100` analyzes every 100th frame)
  - Once optimal parameters are found, re-run with `sieve=1` on full dataset

- **Multi-resolution analysis:**
  - Run DIVINE with different `k` values to generate clustering hierarchies at multiple resolutions
  - Useful for understanding stable vs. unstable cluster structures
  - Compare dendrograms across runs to identify consistent pathway structures
