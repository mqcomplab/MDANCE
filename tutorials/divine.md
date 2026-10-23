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

**System Information:**
```python
input_traj_numpy = data.sim_traj_numpy    # Path to trajectory numpy array
N_atoms = 50                              # Number of atoms (1 for non-MD data)
sieve = 1                                  # Take every sieve-th frame
```

- `input_traj_numpy` : Path to the numpy array (`.npy` file) from step 1
- `N_atoms` : Number of atoms in your system. **For non-MD datasets, use 1**
- `sieve` : Frame sampling rate (e.g., `10` skips every 10th frame for faster analysis)

**DIVINE Parameters:**
```python
end = 'k'                      # Stopping criterion: 'k' for fixed count, 'points' for singletons
split = "weighted_MSD"         # Split strategy: 'MSD', 'radius', 'weighted_MSD'
anchors = "nani"               # Anchor method: 'nani', 'outlier_pair', 'splinter_split'
init_type = "strat_all"        # Init strategy: 'strat_all', 'strat_reduced', 'comp_sim', 
                               #               'div_select', 'k-means++', 'random'
k = 30                         # Target number of clusters
threshold = 0                  # Min relative cluster size [0-1]
refine = True                  # Refine splits with k-means
```

**Parameter explanations:**

- `end` : Stopping criterion
  - `'k'` : Stop when reaching `k` clusters (recommended)
  - `'points'` : Stop when all clusters are singletons (exhaustive)

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

#### Execution

```bash
cd scripts/divine
python run_divine.py
```

The script will run hierarchically, splitting clusters until reaching `k` clusters, and report progress to the console.

### 3. DIVINE Outputs

The script generates two output files:

**Cluster quality metrics:**
- `divine_nani_scores.csv`
  - Columns: `n_clusters`, `CH_score`, `DB_score`
  - Each row represents cluster counts at different hierarchy levels
  - **CH score** (Calinski-Harabasz): Higher is better (max ~∞)
  - **DB score** (Davies-Bouldin): Lower is better (min ~0)

**Cluster structure metrics:**
- `divine_nani_msds.csv`
  - Columns: `level`, `cluster_index`, `population`, `msd`
  - Tracks splitting history and cluster properties
  - `level` : Hierarchical depth of the split
  - `cluster_index` : Identifier for the cluster
  - `population` : Number of frames in the cluster
  - `msd` : Mean-square deviation (cluster tightness)

### 4. Analysis of Results

**Step-by-step analysis:** [scripts/divine/analysis_db.ipynb](../scripts/divine/analysis_db.ipynb)

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
# System info - EDIT THESE
input_traj_numpy = '../../data/md/backbone.npy'
N_atoms = 50
sieve = 1

# DIVINE params - EDIT THESE
k = 30                          # Desired number of clusters
split = 'weighted_MSD'          # Split strategy
anchors = 'nani'                # Anchor method
init_type = 'strat_all'         # Initialization strategy
refine = True                   # Refine splits
threshold = 0                   # Min cluster size threshold
output_dir = 'outputs'          # Output directory
```

#### Execution

```bash
python assign_labels.py
```

#### Output

- Cluster assignments file containing frame indices and cluster labels
- Frame indices compatible with trajectory extraction tools

### 6. Extract Frames for Each Cluster (Optional - MD only)

After obtaining cluster assignments, extract representative frames from each cluster.

**Step-by-step tutorial:** [scripts/outputs/postprocessing.ipynb](../scripts/outputs/postprocessing.ipynb)

## Notes and Tips

- **Determinism:** DIVINE with NANI anchors produces reproducible results across runs due to deterministic seed selection.

- **Computational efficiency:** DIVINE avoids pairwise distance matrices, scaling nearly linearly with dataset size.

- **Parameter tuning:**
  - Start with `split='weighted_MSD'` and `anchors='nani'` (works well for most cases)
  - Increase `k` for finer structural detail; decrease for coarser grouping
  - Use `threshold > 0` to filter out noise-like tiny clusters

- **Stopping criteria:**
  - `end='k'` : Use when you have a target cluster count
  - `end='points'` : Use to explore the full hierarchy (slower, more memory-intensive)

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
