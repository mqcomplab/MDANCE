# Molecular Dynamics Analysis with *N*-ary Clustering Ensembles (MDANCE)
Molecular Dynamics Analysis with *N*-ary Clustering Ensembles (MDANCE) is a flexible *n*-ary clustering package that provides a set of tools for clustering Molecular Dynamics trajectories. The package is written in Python and an extension of the *n*-ary similarity framework. The package is designed to be modular and extensible, allowing for the addition of new clustering algorithms and similarity metrics.

## Table of Contents
- [Background](#background)
- [Clustering Algorithms](#clustering-algorithms)
  - [NANI](#nani)
- [Clustering Postprocessing](#clustering-postprocessing)
  - [PRIME](#prime)
- [Installation](#installation)

## Background
Molecular Dynamics (MD) simulations are a powerful tool for studying the dynamics of biomolecules. However, the analysis of MD trajectories is challenging due to the large amount of data generated. Clustering is an unsupervised machine learning approach to group similar frames into clusters. The clustering results can be used to reveal the structure of the data, identify the most representative structures, and to study the dynamics of the system.

## Clustering Algorithms
### NANI
<p align="center">
<img src="img/nani-logo.PNG" width="200" height=auto align="center"></a></p>

<h3 align="center">
    <p><b>🪄NANI🪄the first installment of MDANCE</b></p>
    </h3>

*k*-Means *N*-Ary Natural Initiation (NANI) is an algorithm for selecting initial centroids for *k*-Means clustering. NANI is an extension of the *k*-Means++ algorithm. NANI stratifies the data to high density region and perform diversity selection on top of the it to select the initial centroids. This is a deterministic algorithm that will always select the same initial centroids for the same dataset and improve on *k*-means++ by reducing the number of iterations required to converge and improve the clustering quality.

**A tutorial is available for NANI at [tutorials/nani.md](tutorials/nani.md).**

## Clustering Postprocessing
### PRIME
<h3 align="center"> 
    <img src="img/logo.png" width="800" height=auto align="center"></a>
    &nbsp
    <p><b>🪄 Predict Protein Structure with Precision 🪄</b></p>
    </h3>

<table>
  <tr>
    <td>
      <p>Protein Retrieval via Integrative Molecular Ensembles (PRIME)</b> is a novel algorithm that predicts the native structure of a protein from simulation or clustering data. These methods perfectly mapped all the structural motifs in the studied systems and required unprecedented linear scaling.</p>
    </td>
    <td>
      <figure>
        <img src="img/2k2e.png" alt="2k2e" width="300" height="auto">
        <figcaption><i>Fig 1. Superposition of the most representative structures found with extended indices (yellow) and experimental native structures (blue) of 2k2e.</i></figcaption>
      </figure>
    </td>
  </tr>
</table>

For more information on the PRIME algorithm, please refer to the [PRIME paper](https://www.biorxiv.org/content/10.1101/2024.03.19.585783v1). 

## Installation
```
git clone https://github.com/mqcomplab/MDANCE.git
```