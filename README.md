[![PyPI version](https://badge.fury.io/py/mcmstclustering.svg)](https://badge.fury.io/py/mcmstclustering)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

# Motivation

MCMSTClustering is an MST-based clustering algorithm designed to handle high-dimensional, imbalanced, varying-density, and arbitrary-shaped datasets. It first forms micro-clusters using KD-Tree range search, then builds a Minimum Spanning Tree over these micro-clusters to detect non-spherical macro-clusters. A final cluster-regulation step refines boundaries and improves clustering quality. Experiments show that MCMSTClustering outperforms several state-of-the-art methods with strong accuracy and efficient runtime.

## Installation

```bash
pip install mcmst-clust
```

## Usage

```bash
from mcmst_clust import MCMSTClustering
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score

data = load_wine()
X = data.data
y = data.target

model = MCMSTClustering(N=19, r=0.49, n_micro=3, random_state=42) 
labels = model.fit_predict(X)

print("n_micro:", model.n_micro_clusters_, "n_macro:", model.n_macro_clusters_)
print("ARI:", adjusted_rand_score(y, labels))
```

## Oerview

MCMSTClustering (Defining Non-Spherical Clusters by using Minimum Spanning Tree over KD-Tree-based Micro-Clusters) is designed to overcome limitations of conventional clustering algorithms when handling:

	- High-dimensional data
	
	- Imbalanced datasets
	
	- Clusters with varying densities
	
	- Noisy data/outliers
	
	- Arbitrary-shaped clusters
	

The algorithm consists of three main steps:

	1. Micro-cluster Formation: Defines micro-clusters using a KD-Tree data structure with range search.
	
	2. Macro-cluster Construction: Builds a minimum spanning tree (MST) over the micro-clusters to form macro-clusters.
	
	3. Cluster Regulation: Refines the clusters to improve accuracy and overall clustering quality.
	

Extensive experiments against state-of-the-art algorithms show that MCMSTClustering achieves high-quality clustering results with acceptable runtime.

Key Features

	- Clusters datasets with high quality

	- Detects arbitrary-shaped clusters

	- Robust against outliers/noisy data

	- Handles clusters with varying densities

	- Efficient on imbalanced datasets


## Cite

If you use the code in your works, please cite the paper given below:
```bash
Şenol, A. MCMSTClustering: defining non-spherical clusters by using minimum 
spanning tree over KD-tree-based micro-clusters. Neural Comput & Applic 35, 
13239–13259 (2023). https://doi.org/10.1007/s00521-023-08386-3
```

## BibTeX

```bash
@article{csenol2023mcmstclustering,
  title={MCMSTClustering: defining non-spherical clusters by using minimum spanning tree over KD-tree-based micro-clusters},
  author={{\c{S}}enol, Ali},
  journal={Neural Computing and Applications},
  volume={35},
  number={18},
  pages={13239--13259},
  year={2023},
  publisher={Springer}
}
```