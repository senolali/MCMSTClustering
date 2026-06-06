# MCMSTClustering

[![PyPI version](https://badge.fury.io/py/mcmst-clust.svg)](https://badge.fury.io/py/mcmst-clust)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MCMSTClustering** is a density-based clustering algorithm that detects **arbitrary-shaped clusters** in datasets containing outliers, imbalanced class distributions, and varying-density regions — all simultaneously.

> Şenol, A. (2023). MCMSTClustering: defining non-spherical clusters by using minimum spanning tree over KD-tree-based micro-clusters. *Neural Computing and Applications*, 35, 13239–13259. [https://doi.org/10.1007/s00521-023-08386-3](https://doi.org/10.1007/s00521-023-08386-3)

---

## How It Works

The algorithm runs in three stages:

1. **Micro-cluster formation** — A KD-Tree is built over the data. Range searches of radius `r` group dense regions into micro-clusters; a candidate region must contain at least `N` points.

2. **Macro-cluster construction** — Prim's Minimum Spanning Tree is run on the micro-cluster centroids. Edges longer than `2 × r` are excluded (sparse graph), keeping runtime practical. Connected components with at least `n_micro` micro-clusters become macro-clusters.

3. **Cluster regulation** — Unassigned points within distance `2 × r` of any micro-cluster centroid are absorbed, closing gaps between micro-clusters of the same macro-cluster.

Points remaining unassigned are labelled `-1` (noise/outlier).

---

## Installation

```bash
pip install mcmst-clust
```

Or install from source:

```bash
git clone https://github.com/senolali/MCMSTClustering.git
cd MCMSTClustering
pip install -e .
```

---

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mcmstclustering import MCMSTClustering

# Generate two-blob dataset
rng = np.random.default_rng(42)
X = np.vstack([
    rng.normal([0.2, 0.2], 0.05, (150, 2)),
    rng.normal([0.8, 0.8], 0.05, (150, 2)),
])

# Normalise to [0, 1] — strongly recommended
X = MinMaxScaler().fit_transform(X)

# Fit
model = MCMSTClustering(N=5, r=0.08, n_micro=2)
labels = model.fit_predict(X)

print(f"Clusters found : {model.n_clusters_}")
print(f"Micro-clusters : {len(model.micro_clusters_)}")
print(f"Noise points   : {(labels == -1).sum()}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: raw data
axes[0].scatter(X[:, 0], X[:, 1], c="steelblue", s=20, alpha=0.6)
axes[0].set_title("Raw Data")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

# Right: clustering result
noise = labels == -1
axes[1].scatter(X[~noise, 0], X[~noise, 1], c=labels[~noise],
                cmap="tab10", s=20, alpha=0.8, label="Cluster points")
if noise.any():
    axes[1].scatter(X[noise, 0], X[noise, 1], c="lightgray",
                    s=20, marker="x", label="Noise")
axes[1].scatter(model.micro_cluster_centers_[:, 0],
                model.micro_cluster_centers_[:, 1],
                c="red", s=60, marker="+", linewidths=1.5,
                label="Micro-cluster centers")
axes[1].set_title(f"MCMSTClustering — {model.n_clusters_} clusters found")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `N` | int | 5 | Minimum points required to form a micro-cluster |
| `r` | float | 0.05 | Search radius for micro-cluster formation (Euclidean) |
| `n_micro` | int | 3 | Minimum micro-clusters to form a macro-cluster |

> **Tip:** Normalise your data to `[0, 1]` with `sklearn.preprocessing.MinMaxScaler` before fitting. The paper uses this normalisation throughout all experiments.

---

## Attributes (after fitting)

| Attribute | Description |
|-----------|-------------|
| `labels_` | Cluster label per sample; `-1` = noise |
| `n_clusters_` | Number of macro-clusters found |
| `micro_clusters_` | List of index arrays, one per micro-cluster |
| `micro_cluster_centers_` | Centroid array of shape `(n_mc, n_features)` |
| `n_features_in_` | Number of features seen during fit |

---

## Scikit-learn Compatibility

`MCMSTClustering` extends `sklearn.base.BaseEstimator` and `ClusterMixin`, so it works seamlessly with scikit-learn utilities:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from mcmstclustering import MCMSTClustering

pipe = Pipeline([
    ("scaler", MinMaxScaler()),
    ("clustering", MCMSTClustering(N=5, r=0.06, n_micro=3)),
])
labels = pipe.fit_predict(X)
```

---

## Example — Visualisation

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from mcmstclustering import MCMSTClustering

X, _ = make_moons(n_samples=400, noise=0.07, random_state=0)
X = MinMaxScaler().fit_transform(X)

model = MCMSTClustering(N=5, r=0.08, n_micro=3).fit(X)

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="tab10", s=10)
plt.title(f"MCMSTClustering — {model.n_clusters_} clusters found")
plt.tight_layout()
plt.savefig("moons_result.png", dpi=150)
plt.show()
```

---

## Comparison with State-of-the-Art

From the original paper (22 datasets, 4 indices: ARI, Accuracy, Purity, NMI):

| Algorithm | Best results (out of 88 tests) |
|-----------|-------------------------------|
| **MCMSTClustering** | **62** |
| MST_Clustering | 51 |
| genieclust | 48 |
| DBSCAN | 45 |
| Agglomerative | 17 |
| k-means | 17 |
| Spectral | 9 |

---

## Citation

If you use MCMSTClustering in your research, please cite:

```bibtex
@article{senol2023mcmstclustering,
  title     = {MCMSTClustering: defining non-spherical clusters by using
               minimum spanning tree over KD-tree-based micro-clusters},
  author    = {{\c{S}}enol, Ali},
  journal   = {Neural Computing and Applications},
  volume    = {35},
  pages     = {13239--13259},
  year      = {2023},
  publisher = {Springer},
  doi       = {10.1007/s00521-023-08386-3}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
