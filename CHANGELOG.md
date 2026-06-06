# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] — 2026-06-06

### Changed
- Complete rewrite of core algorithm for full scikit-learn compatibility
  (`ClusterMixin`, `BaseEstimator`, `fit` / `fit_predict` / `__repr__`).
- KD-Tree range search now uses `scipy.spatial.KDTree` for robustness.
- Prim's MST implemented with sparse edge pruning (edges > 2r excluded).
- Cluster regulation absorbs unassigned points within distance 2r of any
  micro-cluster centroid.
- Added `micro_cluster_centers_`, `n_clusters_`, `n_features_in_` attributes.
- Added parameter validation with informative error messages.
- Warning raised when no micro-clusters are found.
- Full test suite (71 tests, 4 Python versions via GitHub Actions CI).
- Automated PyPI publishing via GitHub Actions Trusted Publishing on version tags.

## [1.1.0] — 2025-12-05

### Added
- Initial release of MCMSTClustering.
- Three-stage algorithm: micro-cluster formation (KD-Tree), macro-cluster
  construction (Prim's MST), and cluster regulation.
- Full scikit-learn compatibility (`BaseEstimator`, `ClusterMixin`).
- Comprehensive test suite with `pytest`.
- Example script with four synthetic datasets.
