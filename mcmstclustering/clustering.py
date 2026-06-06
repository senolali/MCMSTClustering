"""
MCMSTClustering algorithm implementation.

The algorithm operates in three stages:
    1. Micro-cluster definition via KD-Tree range search
    2. Macro-cluster construction via Prim's Minimum Spanning Tree
    3. Cluster regulation to fill gaps between micro-clusters

Reference:
    Şenol, A. (2023). MCMSTClustering: defining non-spherical clusters by using
    minimum spanning tree over KD-tree-based micro-clusters.
    Neural Computing and Applications, 35, 13239–13259.
    https://doi.org/10.1007/s00521-023-08386-3
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted


class MCMSTClustering(ClusterMixin, BaseEstimator):
    """Minimum Spanning Tree clustering over KD-Tree-based micro-clusters.

    MCMSTClustering detects arbitrary-shaped clusters in datasets that may
    contain outliers, imbalanced class distributions, and varying-density
    regions. It proceeds in three stages:

    1. **Micro-cluster formation** – A KD-Tree is built over the data and
       range searches of radius *r* are used to group dense regions into
       micro-clusters.  A candidate region must contain at least *N* points
       to become a micro-cluster.

    2. **Macro-cluster construction** – Prim's Minimum Spanning Tree
       algorithm is run on the micro-cluster centroids.  Edges longer than
       ``2 * r`` are excluded before MST construction, making the graph
       sparse and reducing runtime.  A connected component must contain at
       least *n_micro* micro-clusters to be promoted to a macro-cluster.

    3. **Cluster regulation** – Unassigned points whose distance to the
       nearest micro-cluster centroid is ≤ ``2 * r`` are absorbed into that
       micro-cluster, closing spatial gaps between micro-clusters that
       belong to the same macro-cluster.

    Points that remain unassigned after all three stages are labelled ``-1``
    (noise / outlier).

    Parameters
    ----------
    N : int, default=5
        Minimum number of points required to form a micro-cluster.
    r : float, default=0.3
        Radius used for the KD-Tree range search when forming micro-clusters.
        All distances are Euclidean.  Data should ideally be normalised to
        ``[0, 1]`` before fitting (see *Notes*).
    n_micro : int, default=3
        Minimum number of micro-clusters required to form a macro-cluster
        (connected component in the MST).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label for each sample.  Noise points are labelled ``-1``.
    n_clusters_ : int
        Number of macro-clusters found (excluding noise).
    micro_clusters_ : list of ndarray
        Indices of the original data points belonging to each micro-cluster.
    micro_cluster_centers_ : ndarray of shape (n_micro_clusters, n_features)
        Centroid of each micro-cluster.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Notes
    -----
    The paper normalises all datasets to ``[0, 1]`` using min-max scaling
    before running the algorithm.  It is strongly recommended to do the same::

        from sklearn.preprocessing import MinMaxScaler
        X_scaled = MinMaxScaler().fit_transform(X)
        model = MCMSTClustering(N=5, r=0.05, n_micro=3).fit(X_scaled)

    Time complexity:  O(m·n²·log n), where *m* is the number of
    micro-clusters and *n* is the number of data points.  In practice the
    constant factor is much smaller because the MST graph is sparse
    (edges > 2r are removed before construction).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from mcmstclustering import MCMSTClustering
    >>> rng = np.random.default_rng(42)
    >>> X = np.vstack([rng.normal([0, 0], 0.1, (100, 2)),
    ...                rng.normal([1, 1], 0.1, (100, 2))])
    >>> X = MinMaxScaler().fit_transform(X)
    >>> model = MCMSTClustering(N=4, r=0.08, n_micro=2).fit(X)
    >>> model.n_clusters_
    2

    References
    ----------
    Şenol, A. (2023). MCMSTClustering: defining non-spherical clusters by
    using minimum spanning tree over KD-tree-based micro-clusters.
    *Neural Computing and Applications*, 35, 13239–13259.
    https://doi.org/10.1007/s00521-023-08386-3
    """

    def __init__(
        self,
        N: int = 5,
        r: float = 0.3,
        n_micro: int = 3,
    ) -> None:
        self.N = N
        self.r = r
        self.n_micro = n_micro

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y=None) -> "MCMSTClustering":
        """Compute MCMSTClustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self : MCMSTClustering
            Fitted estimator.
        """
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        self._validate_params()
        self.n_features_in_ = X.shape[1]

        # Stage 1 – micro-clusters
        micro_clusters, mc_labels = self._define_micro_clusters(X)

        if len(micro_clusters) == 0:
            warnings.warn(
                "No micro-clusters were found. Try decreasing N or increasing r.",
                UserWarning,
                stacklevel=2,
            )
            self.labels_ = np.full(len(X), -1, dtype=int)
            self.n_clusters_ = 0
            self.micro_clusters_ = []
            self.micro_cluster_centers_ = np.empty((0, X.shape[1]))
            return self

        centers = np.array(
            [X[mc].mean(axis=0) for mc in micro_clusters], dtype=np.float64
        )

        # Stage 2 – macro-clusters via Prim's MST
        macro_labels = self._define_macro_clusters(centers)

        # Stage 3 – cluster regulation
        mc_labels = self._regulate_clusters(X, micro_clusters, centers, mc_labels)

        # Build final sample-level labels
        self.labels_ = self._build_labels(
            X, micro_clusters, mc_labels, macro_labels
        )

        self.micro_clusters_ = micro_clusters
        self.micro_cluster_centers_ = centers

        unique_macro = set(macro_labels[macro_labels >= 0])
        self.n_clusters_ = len(unique_macro)

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.fit(X, y).labels_

    # ------------------------------------------------------------------
    # Stage 1 – define micro-clusters  (Algorithm 3 in the paper)
    # ------------------------------------------------------------------

    def _define_micro_clusters(
        self, X: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Build micro-clusters using KD-Tree range search.

        Each point is used as a candidate origin.  Points within radius *r*
        that form a group of at least *N* members become a micro-cluster.
        The loop iterates until the set of unassigned points stabilises.

        Returns
        -------
        micro_clusters : list of ndarray
            Each element is an int array of indices belonging to that MC.
        mc_labels : ndarray of shape (n_samples,)
            Per-sample micro-cluster index; -1 for unassigned.
        """
        n = len(X)
        mc_labels = np.full(n, -1, dtype=int)
        micro_clusters: list[np.ndarray] = []

        # Iterative pass – mirrors Algorithm 3
        prev_n_mc = -1
        while prev_n_mc != len(micro_clusters):
            prev_n_mc = len(micro_clusters)

            # Work only on idle (unassigned) points
            idle_mask = mc_labels == -1
            idle_idx = np.where(idle_mask)[0]

            if len(idle_idx) == 0:
                break

            tree = KDTree(X[idle_idx])

            for local_j, global_j in enumerate(idle_idx):
                if mc_labels[global_j] != -1:
                    # Already assigned during this sweep
                    continue

                # Indices into idle_idx array returned by range search
                local_neighbours = tree.query_ball_point(X[global_j], r=self.r)
                global_neighbours = idle_idx[local_neighbours]

                if len(global_neighbours) >= self.N:
                    mc_id = len(micro_clusters)
                    micro_clusters.append(global_neighbours)
                    mc_labels[global_neighbours] = mc_id

        return micro_clusters, mc_labels

    # ------------------------------------------------------------------
    # Stage 2 – macro-clusters via Prim's MST  (Algorithm 4)
    # ------------------------------------------------------------------

    def _define_macro_clusters(self, centers: np.ndarray) -> np.ndarray:
        """Run Prim's MST on micro-cluster centroids and extract components.

        Edges whose weight exceeds ``2 * r`` are never added (sparse graph).
        Each connected component with at least *n_micro* nodes becomes a
        macro-cluster.

        Parameters
        ----------
        centers : ndarray of shape (n_mc, n_features)

        Returns
        -------
        macro_labels : ndarray of shape (n_mc,)
            Macro-cluster index for each micro-cluster; -1 = isolated.
        """
        n_mc = len(centers)
        if n_mc == 0:
            return np.array([], dtype=int)

        threshold = 2.0 * self.r

        # Build adjacency: for each node find neighbours within threshold
        tree = KDTree(centers)
        # query_ball_tree returns lists of neighbour indices
        neighbours: list[list[int]] = tree.query_ball_tree(tree, r=threshold)

        # Prim's algorithm on the sparse graph
        in_mst = np.zeros(n_mc, dtype=bool)
        parent = np.full(n_mc, -1, dtype=int)
        key = np.full(n_mc, np.inf)
        key[0] = 0.0

        # Simple O(n²) Prim – acceptable because n_mc << n_samples
        mst_edges: list[tuple[int, int]] = []  # (u, v)
        for _ in range(n_mc):
            # Pick minimum key vertex not yet in MST
            candidates = np.where(~in_mst)[0]
            u = candidates[np.argmin(key[candidates])]
            in_mst[u] = True

            if parent[u] != -1:
                mst_edges.append((parent[u], u))

            for v in neighbours[u]:
                if in_mst[v]:
                    continue
                w = float(np.linalg.norm(centers[u] - centers[v]))
                if w < key[v]:
                    key[v] = w
                    parent[v] = u

        # Connected components of MST edges → macro-clusters
        adj: list[list[int]] = [[] for _ in range(n_mc)]
        for u, v in mst_edges:
            adj[u].append(v)
            adj[v].append(u)

        macro_labels = np.full(n_mc, -1, dtype=int)
        macro_id = 0
        for start in range(n_mc):
            if macro_labels[start] != -1:
                continue
            # BFS
            component = self._bfs(adj, start)
            if len(component) >= self.n_micro:
                for node in component:
                    macro_labels[node] = macro_id
                macro_id += 1

        return macro_labels

    # ------------------------------------------------------------------
    # Stage 3 – cluster regulation  (Algorithm 5)
    # ------------------------------------------------------------------

    def _regulate_clusters(
        self,
        X: np.ndarray,
        micro_clusters: list[np.ndarray],
        centers: np.ndarray,
        mc_labels: np.ndarray,
    ) -> np.ndarray:
        """Assign unassigned points to nearest MC if distance ≤ 2r.

        This closes spatial gaps between micro-clusters that naturally occur
        because each micro-cluster is a ball of radius *r*.

        Returns
        -------
        mc_labels : ndarray of shape (n_samples,)  (updated in-place copy)
        """
        mc_labels = mc_labels.copy()
        threshold = 2.0 * self.r

        unassigned = np.where(mc_labels == -1)[0]
        if len(unassigned) == 0 or len(centers) == 0:
            return mc_labels

        tree = KDTree(centers)
        dists, indices = tree.query(X[unassigned], k=1)

        for local_i, global_i in enumerate(unassigned):
            if dists[local_i] <= threshold:
                nearest_mc = int(indices[local_i])
                mc_labels[global_i] = nearest_mc
                micro_clusters[nearest_mc] = np.append(
                    micro_clusters[nearest_mc], global_i
                )

        return mc_labels

    # ------------------------------------------------------------------
    # Label assembly
    # ------------------------------------------------------------------

    def _build_labels(
        self,
        X: np.ndarray,
        micro_clusters: list[np.ndarray],
        mc_labels: np.ndarray,
        macro_labels: np.ndarray,
    ) -> np.ndarray:
        """Map per-sample micro-cluster assignments to macro-cluster labels."""
        n = len(X)
        labels = np.full(n, -1, dtype=int)

        for mc_id, mc_points in enumerate(micro_clusters):
            macro = int(macro_labels[mc_id]) if mc_id < len(macro_labels) else -1
            if macro >= 0:
                labels[mc_points] = macro

        return labels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bfs(adj: list[list[int]], start: int) -> list[int]:
        """Return all nodes reachable from *start* via BFS."""
        visited = [False] * len(adj)
        queue = [start]
        visited[start] = True
        component: list[int] = []
        while queue:
            node = queue.pop(0)
            component.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        return component

    def _validate_params(self) -> None:
        if not isinstance(self.N, (int, np.integer)) or self.N < 1:
            raise ValueError(f"N must be a positive integer, got {self.N!r}.")
        if not isinstance(self.r, (float, int, np.floating)) or self.r <= 0:
            raise ValueError(f"r must be a positive float, got {self.r!r}.")
        if not isinstance(self.n_micro, (int, np.integer)) or self.n_micro < 1:
            raise ValueError(
                f"n_micro must be a positive integer, got {self.n_micro!r}."
            )

    # ------------------------------------------------------------------
    # Scikit-learn compatibility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MCMSTClustering(N={self.N}, r={self.r}, n_micro={self.n_micro})"
        )
