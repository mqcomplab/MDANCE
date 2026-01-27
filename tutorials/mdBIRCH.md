# mdBIRCH - Online CF-tree clustering for MD data

Table of Contents
=================
- [Overview](#overview)
- [When to use mdBIRCH](#when-to-use-mdbirch)
- [Tutorial](#tutorial)
  - [1. Input Preparations](#1-input-preparations)
  - [2. mdBIRCH clustering](#2-mdbirch-clustering)
  - [3. Outputs](#3-outputs)
  - [4. Extract frames for each cluster (Optional)](#4-extract-frames-for-each-cluster-optional)
- [Notes](#notes-and-tips)

## Overview
mdBIRCH is an online clustering algorithm that incrementally builds a **CF-tree** (Clustering Feature tree) in a *single pass* over frames. 
Each incoming frame is either merged into its closest existing subcluster (if a threshold criterion is satisfied) or it creates a new subcluster.

In this repository, mdBIRCH is implemented as `mdance.cluster.mdbirch.mdBirch` and uses a **radius-based merge** criterion controlled by a user-provided `threshold`.
The merge decision is designed to be physically interpretable for molecular trajectories: larger `threshold` values yield fewer (broader) clusters; smaller values yield more (finer) clusters.

## When to use mdBIRCH
**Current best practice (today):** we primarily use mdBIRCH on **finished simulations** (post hoc clustering). It’s fast, memory-bounded, and provides a straightforward “one-pass” partitioning of an already-generated trajectory.

**Future direction:** mdBIRCH becomes even more powerful when paired with an engine that enables *instantaneous clustering*—i.e., streaming frames during simulation and assigning them to clusters online in real time. In that setup, mdBIRCH acts as the online clustering backbone, while the engine handles the streaming + orchestration layer.

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

### 2. mdBIRCH clustering
A minimal runnable example is provided in:

- `run_mdbirch.py`

This script:
1) loads your trajectory numpy array  
2) configures the merge rule (radius-based)  
3) fits the mdBIRCH model  
4) writes a `frame,label` CSV for downstream analysis

#### Parameters to edit in `run_mdbirch.py`
    # Parameters and configuration
    input_traj_numpy = data.sim_traj_numpy
    sieve = 1
    threshold = 3.06

- `input_traj_numpy` : path to the `.npy` array from step 1  
- `sieve` : take every `sieve`-th frame (e.g., `10` for a quick scan)  
- `threshold` : the clustering tolerance that controls granularity  

#### Merge criterion (radius)
For MD use cases in this repo, we assume your frames are represented as **reference-aligned Cartesian coordinates** (so the feature dimension is implicitly `3 * n_atoms_selected`).
As a result, you typically do **not** need to think about a separate “feature” setting—the script will infer the dimensionality directly from the loaded numpy array.

#### Execution
```bash
python run_mdbirch.py
```

### 3. Outputs
The script prints:
- the total number of clusters discovered

and writes:
- `mdbirch_labels_<threshold>.csv` containing:
  - `frame` : frame index (0-based in the sieved trajectory)
  - `label` : cluster id assigned by mdBIRCH

Internally, you can also access:
- `clusters = model.get_cluster_mol_ids()` → list of lists of frame indices per cluster  
- `centroids = model.get_centroids()` → centroids in feature-space for each cluster  

### 4. Extract frames for each cluster (Optional)
**This step is optional and for Molecular Dynamics trajectories only.**

Once you have `frame,label` assignments, you can extract representative frames from each cluster using the standard postprocessing workflow:

**Step-by-step tutorial can be found in the [postprocessing notebook](../scripts/outputs/postprocessing.ipynb).**

## Notes
- **Interpretation:** mdBIRCH produces clusters that are easy to reason about because the merge rule is threshold-controlled and the clustering is constructed incrementally.

