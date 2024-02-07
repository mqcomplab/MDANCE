# MDANCE
MDANCE is a flexible n-ary clustering package for all applications.

## Available Clustering Algorithms
<p align="center">
<img src="img/nani-logo.PNG" width="200" height=auto align="center"></a></p>

<h3 align="center">
    <p><b>🪄NANI🪄the first installment of MDANCE</b></p>
    </h3>

*k*-Means N-Ary Natural Implementation (NANI) is an algorithm for selecting initial centroids for *k*-Means clustering. *k*-Means NANI is an extension of the *k*-Means++ algorithm. *k*-Means++ selects the initial centroids by randomly selecting a point from the dataset and then selecting the next centroid based on the distance from the previous centroid. *k*-MeansNANI stratifies the data to high density region and perform diversity selection on top of the it to select the initial centroids. This is a deterministic algorithm that will always select the same initial centroids for the same dataset and improve on *k*-means++ by reducing the number of iterations required to converge and improve the clustering quality.

**A tutorial is available for NANI at [tutorials/nani.md](tutorials/nani.md).**