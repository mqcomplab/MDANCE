# MDANCE
Molecular Dynamics Analysis with *N*-ary Clustering Ensembles (MDANCE) is a flexible *n*-ary clustering package that provides a set of tools for clustering Molecular Dynamics trajectories. The package is written in Python and an extension of the *n*-ary similarity framework. The package is designed to be modular and extensible, allowing for the addition of new clustering algorithms and similarity metrics.

## Table of Contents
- [Background](#background)
- [Clustering Algorithms](#clustering-algorithms)
  - [NANI](#nani)
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

*k*-Means *N*-Ary Natural Implementation (NANI) is an algorithm for selecting initial centroids for *k*-Means clustering. NANI is an extension of the *k*-Means++ algorithm. NANI stratifies the data to high density region and perform diversity selection on top of the it to select the initial centroids. This is a deterministic algorithm that will always select the same initial centroids for the same dataset and improve on *k*-means++ by reducing the number of iterations required to converge and improve the clustering quality.

**A tutorial is available for NANI at [tutorials/nani.md](tutorials/nani.md).**

## Installation
```
git clone https://github.com/mqcomplab/MDANCE.git
```
