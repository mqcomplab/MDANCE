from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

X, true_labels = make_blobs(n_samples=1000, centers=7, n_features=2, random_state=0)
# different color for each label
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=true_labels, s=7, cmap='viridis')
plt.savefig('blobs.png')