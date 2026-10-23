# PRISM - Pathway Representation via Intrinsic Structural Medoids

Table of Contents
=================
- [Overview](#overview)
- [When to use PRISM](#when-to-use-prism)
- [Tutorial](#tutorial)
  - [1. Input Preparations](#1-input-preparations)
  - [2. PRISM Clustering](#2-prism-clustering)
  - [3. Outputs](#3-outputs)
  - [4. Pathway Medoid Analysis (Optional)](#4-pathway-medoid-analysis-optional)
- [Notes](#notes-and-tips)

## Overview

PRISM (Pathway Representation via Intrinsic Structural Medoids) is a state-aware pathway clustering framework designed for comparing and classifying multiple molecular pathways or trajectories. 

PRISM proceeds in three stages:

1. **Representative-set construction (structural medoids):**
   Each pathway is mapped to a compact set of representative conformations (medoids) obtained from k-means clustering in your chosen feature space. Three construction strategies are supported:

   - **Option 1 (Global medoid sharing):** Pool all frames across all pathways, cluster into `k` clusters, compute one medoid per cluster, then assign each pathway the subset of global medoids whose clusters contain frames from that pathway.
   
   - **Option 2 (Per-pathway independent medoids):** Cluster each pathway independently into `k` clusters and compute its medoids. Representatives are pathway-specific with no enforced sharing.
   
   - **Option 3 (Two-stage refinement):** First perform Option 2 to obtain local medoids, pool all local medoids, cluster into `k_final` clusters, compute refined global medoids, then assign each pathway the refined medoids matching its local medoids.

2. **Pathway dissimilarity (weighted average Hausdorff distance):**
   For two pathways represented by sets A and B of medoids, PRISM computes a symmetric dissimilarity based on mean nearest-neighbor distances in both directions. Multiple weighting schemes are supported.

3. **Hierarchical clustering:**
   The resulting pathway dissimilarity matrix is clustered using hierarchical agglomerative clustering (HAC), enabling flexible pathway grouping and dendrogram visualization.

## When to use PRISM

PRISM is ideal for:
- **Enhanced sampling analysis:** Comparing pathways from enhanced sampling simulations (e.g., metadynamics, steered MD).
- **Multi-trajectory comparison:** Clustering multiple independent MD simulations to identify similar dynamics pathways.
- **Pathway classification:** Grouping trajectory ensembles into biologically meaningful pathway classes.
- **Conformational pathway analysis:** Summarizing complex multi-frame trajectories with compact medoid sets for efficient comparison.
- **Large-scale trajectory studies:** Handling hundreds or thousands of pathways with minimal memory overhead.

## Tutorial

### 1. Input Preparations

Each pathway (trajectory) should be prepared as a 2D numpy array with shape `(n_frames, n_features)`. All pathways must have the same number of features (columns).

**Supported input formats:**
- **CSV files:** Load with `np.genfromtxt(file, delimiter=',')`
- **NumPy files (.npy):** Load with `np.load(file)`
- **Other formats:** Adapt the parsing logic in the script as needed

**Example dataset:**
MDANCE includes a sample dataset of 80 alanine dipeptide pathways with dihedral angle features at [`src/mdance/data/ala_pathways/`](../src/mdance/data/ala_pathways/). These files (`pathway_dih_0.csv` through `pathway_dih_79.csv`) contain phi and psi dihedral angles from multiple MD simulations of alanine dipeptide, making them ideal for testing PRISM.

**Important:** 
- Ensure all trajectories are aligned and/or centered if needed before clustering.
- All pathways must share the same feature space (same number of columns).

For Molecular Dynamics trajectories, see [scripts/inputs/preprocessing.ipynb](../scripts/inputs/preprocessing.ipynb) for step-by-step trajectory preprocessing.

### 2. PRISM Clustering

A runnable example is provided in:
- [`scripts/prism/run_prism.py`](../scripts/prism/run_prism.py)

This script:
1. Loads multiple pathway files from a directory pattern
2. Constructs medoid representations for each pathway
3. Computes pathway dissimilarities
4. Performs hierarchical clustering
5. Writes pathway cluster assignments and medoid indices

#### Parameters to edit in `run_prism.py`

**System Parameters:**
```python
file_pattern = '/data/*.csv'        # Pattern pointing to trajectory files
metric = 'MSD'                      # Metric for medoid computation (e.g., 'MSD', 'RMSD')
```

- `file_pattern` : glob pattern pointing to your input files. Each match is treated as a separate pathway.
- `metric` : Feature space metric identifier used during medoid selection (e.g., 'MSD' for mean-square deviation).

**PRISM Parameters:**
```python
option = 2                          # Medoid construction: 1=global, 2=per-traj, 3=two-stage
k = 20                              # Number of clusters for medoid construction
k_final = k                         # Second-stage cluster count (used only if option == 3)
weight_scheme = 'weighted_avg'      # Average Hausdorff normalization scheme: 
                                    # 'unnormalized', 'weighted_avg', 'sym_avg', 'product_avg'
```

- `option` : Strategy for building medoid sets (1, 2, or 3; see Overview).
- `k` : Number of clusters used when building representative medoids.
- `k_final` : Number of clusters in the second refinement stage (only for `option==3`).
- `weight_scheme` : Weighting method for the Hausdorff distance calculation.

**Hierarchical Clustering Parameters:**
```python
t = 2                               # Threshold for fcluster (interpreted by criterion)
criterion = 'maxclust'              # fcluster criterion: 'maxclust', 'distance', etc.
linkage_method = 'ward'             # HAC linkage method: 'ward', 'average', 'complete', etc.
```

- `t` : Threshold parameter passed to `scipy.cluster.hierarchy.fcluster`. For `criterion='maxclust'`, `t` is the maximum number of pathway clusters.
- `criterion` : Clustering criterion (e.g., `'maxclust'` for fixed max cluster count).
- `linkage_method` : Linkage method for hierarchical clustering (e.g., `'ward'` minimizes variance).

**Output Parameters:**
```python
save_medoid_indices = True          # Save medoid frame indices as CSV
```

#### Data Loading

Modify the data loading section to match your file format:

```python
# For CSV files (default):
for file in glob.glob(file_pattern):
    traj = file.split('_')[-1].split('.')[0]
    frame = np.genfromtxt(file, delimiter=',')
    frames_all.append((traj, frame))

# For NumPy files:
for file in glob.glob(file_pattern, recursive=True):
    traj = file.split('/')[-1].split('.')[0]
    frame = np.load(file)
    frames_all.append((traj, frame))
```

Each entry in `frames_all` is a tuple: `(unique_pathway_id, 2D_array)`.

**Quick start with example data:**
To run PRISM on the included alanine dipeptide pathway dataset:

```python
file_pattern = '/path/to/MDANCE/src/mdance/data/ala_pathways/*.csv'
```

This will load all 80 alanine dipeptide pathways for clustering.

#### Execution

```bash
cd scripts/prism
python run_prism.py
```

### 3. Outputs

The script generates the following files:

**Cluster assignments:**
- `prism_opt{option}_k{k}_labels.csv` 
  - Columns: `traj_id`, `cluster`
  - Each row assigns a pathway to a cluster

**Dendrogram visualization:**
- `prism_opt{option}_k{k}.png` 
  - Hierarchical clustering dendrogram showing pathway relationships

**Medoid frame indices (if enabled):**
- `prism_opt{option}_k{k}_medoid_indices.csv` 
  - Columns: `assigned_traj`, `source_traj`, `frame_index`
  - Maps medoid frames back to their source trajectories


### 4. Pathway Medoid Analysis (Optional)

After obtaining medoid indices from Step 3, you can extract and analyze representative frames for each medoid:

1. Use the medoid frame indices CSV to identify key frames in each pathway.
2. Extract these frames from the original trajectory files.
3. Perform structural analysis (RMSD, visualization, etc.) on the medoids.
4. Compare medoid sets across pathways to understand pathway differences.

For MD trajectory post-processing, see [postprocessing notebook](../scripts/outputs/postprocessing.ipynb).

## Notes and Tips

- **Memory efficiency:** PRISM avoids computing full pairwise distance matrices between frames by using compact medoid representations, making it suitable for large-scale pathway analysis.

- **Medoid selection strategy:** Choose `option` based on your use case:
  - `option=1`: Use when pathways are likely to share similar structural features.
  - `option=2`: Use when pathways are expected to be distinct (recommended starting point).
  - `option=3`: Use for refined comparison after identifying pathway families with option 2.

- **Number of medoids (k):** Higher `k` captures finer structural details but increases computational cost and medoid set complexity. Start with `k=10-20` and adjust based on your insights.

- **Dendrogram interpretation:** Heights in the dendrogram indicate the dissimilarity level at which pathways merge. Large vertical gaps suggest distinct pathway clusters.

- **Clustering criterion:** Use `criterion='maxclust'` with `t` set to your desired number of pathway clusters. Alternatively, use `criterion='distance'` to cut at a specific dissimilarity threshold.
