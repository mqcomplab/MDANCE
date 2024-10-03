<p align="center">
<img src="https://raw.githubusercontent.com/mqcomplab/MDANCE/main/docs/_static/mdance.png" width="300" height=auto align="center"></a></p>

MDANCE (Molecular Dynamics Analysis with *N*-ary Clustering Ensembles) is a flexible *n*-ary clustering package that provides a set of tools for clustering Molecular Dynamics trajectories. The package is written in Python and an extension of the *n*-ary similarity framework. The package is designed to be modular and extensible, allowing for the addition of new clustering algorithms and similarity metrics.Research contained in this package was supported by the National Institute of General Medical Sciences of the National Institutes of Health under award number R35GM150620.

## Menu
- [Installation](#installation)
- [Background](#background)
- [Clustering Algorithms](#clustering-algorithms)
  - [NANI](#nani)
- [Clustering Postprocessing](#clustering-postprocessing)
  - [PRIME](#prime)

## Installation
## Installation
```bash
$ pip install mdance
```
To check for proper installation, run the following command:
```python
>>> import mdance
>>> mdance.__version__
```

## Background
Molecular Dynamics (MD) simulations are a powerful tool for studying the dynamics of biomolecules. However, the analysis of MD trajectories is challenging due to the large amount of data generated. Clustering is an unsupervised machine learning approach to group similar frames into clusters. The clustering results can be used to reveal the structure of the data, identify the most representative structures, and to study the dynamics of the system.

## Clustering Algorithms
### NANI
<p align="center">
<img src="https://raw.githubusercontent.com/mqcomplab/MDANCE/main/docs/_static/nani-logo.PNG" width="150" height=auto align="center"></a></p>

<h3 align="center">
    <p><b>🪄NANI🪄the first installment of MDANCE</b></p>
    </h3>

*k*-Means *N*-Ary Natural Initiation (NANI) is an algorithm for selecting initial centroids for *k*-Means clustering. NANI is an extension of the *k*-Means++ algorithm. NANI stratifies the data to high density region and perform diversity selection on top of the it to select the initial centroids. This is a deterministic algorithm that will always select the same initial centroids for the same dataset and improve on *k*-means++ by reducing the number of iterations required to converge and improve the clustering quality.


#### Example Usage:

```python
>>> from mdance.cluster.nani import KmeansNANI
>>> data = np.load('data.npy')
>>> N = 4
>>> mod = KmeansNANI(data, n_clusters=N, metric='MSD', N_atoms=1)
>>> initiators = mod.initiate_kmeans()
>>> initiators = initiators[:N]
>>> kmeans = KMeans(N, init=initiators, n_init=1, random_state=None)
>>> kmeans.fit(data)
```

**A tutorial is available for NANI [here](https://mdance.readthedocs.io/en/latest/tutorials/nani.html).**

For more information on the NANI algorithm, please refer to the [NANI paper](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00308).

## Clustering Postprocessing
### PRIME

<p align="center">
<img src="https://raw.githubusercontent.com/mqcomplab/MDANCE/main/docs/_static/logo.png" width="800" height=auto align="center"></a></p>

<h3 align="center">
    <p><b>🪄 Predict Protein Structure with Precision 🪄</b></p>
    </h3>

<table>
  <tr>
    <td>
      <p>Protein Retrieval via Integrative Molecular Ensembles (PRIME)</b> is a novel algorithm that predicts the native structure of a protein from simulation or clustering data. These methods perfectly mapped all the structural motifs in the studied systems and required unprecedented linear scaling.</p>
    </td>
    <td>
      <figure>
        <img src="https://raw.githubusercontent.com/mqcomplab/MDANCE/main/docs/img/2k2e.png" alt="2k2e" width="300" height="auto">
        <figcaption><i>Fig 1. Superposition of the most representative structures found with extended indices (yellow) and experimental native structures (blue) of 2k2e.</i></figcaption>
      </figure>
    </td>
  </tr>
</table>

**A tutorial is available for PRIME [here](https://mdance.readthedocs.io/en/latest/tutorials/prime.html).**

For more information on the PRIME algorithm, please refer to the [PRIME paper](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00362). 

### Collab or Contribute?!
Please! Don't hesitate to reach out!
