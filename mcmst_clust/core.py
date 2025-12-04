"""
MCMSTClustering - Revised (bugfixes for regulation, ordering, macro-criteria, scaler reuse)
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings
from typing import List, Tuple, Optional, Dict


class MCMSTClustering:
    def __init__(self, N: int = 5, r: float = 0.1, n_micro: int = 3,
                 random_state: Optional[int] = None):
        self.N = int(N)
        self.r = float(r)
        self.n_micro = int(n_micro)
        self.random_state = random_state

        # RNG: deterministic if random_state provided, nondeterministic otherwise
        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()

        # public attributes
        self.micro_clusters_ = None
        self.macro_clusters_ = None
        self.labels_ = None
        self.n_micro_clusters_ = 0
        self.n_macro_clusters_ = 0

        # internal storages
        self._processed_data = None
        self._mc_ids = np.array([], dtype=int)
        self._mc_counts = np.array([], dtype=int)
        self._mc_macro_ids = np.array([], dtype=int)
        self._mc_centers = np.empty((0, 0), dtype=float)

        # scaler stored after fit
        self._scaler = None

    # -----------------------
    # Public API
    # -----------------------
    def fit(self, X: np.ndarray) -> 'MCMSTClustering':
        X = self._validate_data(X)
        self._scaler = MinMaxScaler()
        Xn = self._scaler.fit_transform(X)
        self._initialize_data(Xn)

        self._define_micro_clusters()
        self._regulate_clusters()
        self._define_macro_clusters()
        self._assign_labels()
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        warnings.warn("MCMSTClustering.predict() assigns points to nearest micro-cluster. Use fit_predict() for best results.")
        X = self._validate_data(X)
        if self._scaler is None:
            raise ValueError("Model must be fitted before calling predict().")
        Xn = self._scaler.transform(X)

        labels = np.zeros(len(Xn), dtype=int)
        if self.n_micro_clusters_ > 0:
            tree = KDTree(self._mc_centers)
            dist, idx = tree.query(Xn, k=1)
            within = dist[:, 0] <= 2 * self.r
            mc_idx = idx[:, 0]
            labels[within] = self._mc_macro_ids[mc_idx[within]]
        return labels

    # -----------------------
    # Internal utils
    # -----------------------
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input must be 2D array.")
        if X.shape[0] < self.N:
            raise ValueError(f"Number of samples ({X.shape[0]}) must be >= N ({self.N}).")
        return X

    def _initialize_data(self, X: np.ndarray):
        n_samples, d = X.shape
        indices = np.arange(1, n_samples + 1).reshape(-1, 1)
        mc_zeros = np.zeros((n_samples, 1))
        macro_zeros = np.zeros((n_samples, 1))
        self._processed_data = np.hstack([indices, mc_zeros, macro_zeros, X])
        self._mc_ids = np.array([], dtype=int)
        self._mc_counts = np.array([], dtype=int)
        self._mc_macro_ids = np.array([], dtype=int)
        self._mc_centers = np.empty((0, d), dtype=float)
        self.micro_clusters_ = np.empty((0, d + 3), float)
        self.macro_clusters_ = np.empty((0, 4), dtype=object)
        self.n_micro_clusters_ = 0
        self.n_macro_clusters_ = 0

    def _update_public_micro_clusters(self):
        if self._mc_centers.size == 0:
            self.micro_clusters_ = np.empty((0, self._processed_data.shape[1]), float)
            return
        ids = self._mc_ids.reshape(-1, 1).astype(float)
        counts = self._mc_counts.reshape(-1, 1).astype(float)
        macro_ids = self._mc_macro_ids.reshape(-1, 1).astype(float)
        self.micro_clusters_ = np.hstack([ids, counts, macro_ids, self._mc_centers])

    # -----------------------
    # Micro clusters
    # -----------------------
    def _define_micro_clusters(self):
        while True:
            unassigned_mask = (self._processed_data[:, 1] == 0)
            unassigned_idx = np.flatnonzero(unassigned_mask)
            if unassigned_idx.size < self.N:
                break

            coords_unassigned = self._processed_data[unassigned_mask, 3:]
            tree = KDTree(coords_unassigned)

            # ALWAYS use rng.shuffle so deterministic when random_state set
            order = np.arange(unassigned_idx.size)
            self.rng.shuffle(order)

            created = False
            for o in order:
                orig_i = unassigned_idx[o]
                point = self._processed_data[orig_i, 3:].reshape(1, -1)
                neigh_rel = tree.query_radius(point, r=self.r)[0]
                if neigh_rel.size >= self.N:
                    neigh_orig = unassigned_idx[neigh_rel]
                    self.n_micro_clusters_ += 1
                    pts = self._processed_data[neigh_orig, 3:]
                    center = np.mean(pts, axis=0)

                    # append micro-cluster internal arrays
                    self._mc_ids = np.append(self._mc_ids, self.n_micro_clusters_)
                    self._mc_counts = np.append(self._mc_counts, len(neigh_orig))
                    self._mc_macro_ids = np.append(self._mc_macro_ids, 0)
                    if self._mc_centers.size == 0:
                        self._mc_centers = center.reshape(1, -1)
                    else:
                        self._mc_centers = np.vstack([self._mc_centers, center])

                    # assign to processed_data
                    self._processed_data[neigh_orig, 1] = self.n_micro_clusters_

                    # update public view
                    self._update_public_micro_clusters()

                    created = True
                    break
            if not created:
                break

    # -----------------------
    # Regulate clusters (fixed: do not change micro centers)
    # -----------------------
    def _regulate_clusters(self):
        unassigned_mask = (self._processed_data[:, 1] == 0)
        unassigned_idx = np.flatnonzero(unassigned_mask)
        if unassigned_idx.size == 0 or self.n_micro_clusters_ == 0:
            return

        tree = KDTree(self._mc_centers)
        order = np.arange(unassigned_idx.size)
        self.rng.shuffle(order)

        for o in order:
            orig_i = unassigned_idx[o]
            point = self._processed_data[orig_i, 3:].reshape(1, -1)
            dist, idx = tree.query(point, k=1)
            if dist[0, 0] <= 2 * self.r:
                mc_idx = int(idx[0, 0])
                mc_id = int(self._mc_ids[mc_idx])

                # assign point to micro-cluster (do NOT change center)
                self._processed_data[orig_i, 1] = mc_id

                # update count only (centers unchanged)
                self._mc_counts[mc_idx] = int(self._mc_counts[mc_idx]) + 1

        # update public view (centers unchanged)
        self._update_public_micro_clusters()

    # -----------------------
    # MST & macro clusters (MARCO CRITERION FIXED TO AND)
    # -----------------------
    def _build_mst_graph(self) -> np.ndarray:
        if self.n_micro_clusters_ == 0:
            return np.array([])
        centers = self._mc_centers
        n = centers.shape[0]
        if n == 1:
            return np.zeros((1, 1))
        dist_matrix = squareform(pdist(centers))
        adj_matrix = np.where(dist_matrix <= 2 * self.r, dist_matrix, 0.0)
        np.fill_diagonal(adj_matrix, 0.0)
        return adj_matrix

    def _prim_mst(self, adj_matrix: np.ndarray) -> List[Tuple[int, int]]:
        n = len(adj_matrix)
        if n <= 1:
            return []
        in_mst = np.zeros(n, dtype=bool)
        edge_list = []
        start_node = int(np.argmax(self._mc_counts))
        in_mst[start_node] = True
        INF = np.inf
        min_to_tree = np.where(adj_matrix[start_node] > 0, adj_matrix[start_node], INF)
        parent = np.full(n, -1, dtype=int)
        parent[:] = start_node
        parent[start_node] = -1

        for _ in range(n - 1):
            candidates = np.where(~in_mst & (min_to_tree < INF))[0]
            if candidates.size == 0:
                break
            next_idx = int(candidates[np.argmin(min_to_tree[candidates])])
            edge_list.append((int(parent[next_idx]) + 1, int(next_idx) + 1))
            in_mst[next_idx] = True
            for j in range(n):
                if not in_mst[j] and adj_matrix[next_idx, j] > 0 and adj_matrix[next_idx, j] < min_to_tree[j]:
                    min_to_tree[j] = adj_matrix[next_idx, j]
                    parent[j] = next_idx
        return edge_list

    def _find_connected_components(self, adj_matrix: np.ndarray, mst_edges: List[Tuple[int, int]]) -> List[List[int]]:
        n = len(adj_matrix)
        if n == 0:
            return []
        adj_list = {i: [] for i in range(n)}
        for u, v in mst_edges:
            u_idx, v_idx = u - 1, v - 1
            adj_list[u_idx].append(v_idx)
            adj_list[v_idx].append(u_idx)

        visited = np.zeros(n, dtype=bool)
        components = []
        for i in range(n):
            if not visited[i]:
                stack = [i]
                comp = []
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        comp.append(node + 1)
                        stack.extend(adj_list[node])
                if comp:
                    components.append(comp)
        return components

    def _define_macro_clusters(self):
        adj_matrix = self._build_mst_graph()
        if adj_matrix.size == 0:
            return
        mst_edges = self._prim_mst(adj_matrix)
        components = self._find_connected_components(adj_matrix, mst_edges)

        for comp in components:
            total_points = int(sum(self._mc_counts[int(mc_id) - 1] for mc_id in comp))
            # FIX: require BOTH conditions (AND) to be satisfied
            if len(comp) >= self.n_micro and total_points >= self.N * self.n_micro:
                self.n_macro_clusters_ += 1
                for mc_id in comp:
                    idx = int(mc_id) - 1
                    self._mc_macro_ids[idx] = self.n_macro_clusters_
                new_row = np.array([self.n_macro_clusters_, len(comp), comp, None], dtype=object)
                self.macro_clusters_ = np.vstack([self.macro_clusters_, new_row]) if self.macro_clusters_.size else new_row.reshape(1, -1)

        # handle large individual micro-clusters
        for i in range(len(self._mc_ids)):
            if self._mc_macro_ids[i] == 0 and self._mc_counts[i] >= self.N * self.n_micro:
                self.n_macro_clusters_ += 1
                self._mc_macro_ids[i] = self.n_macro_clusters_
                new_row = np.array([self.n_macro_clusters_, 1, [int(self._mc_ids[i])], None], dtype=object)
                self.macro_clusters_ = np.vstack([self.macro_clusters_, new_row]) if self.macro_clusters_.size else new_row.reshape(1, -1)

        self._update_public_micro_clusters()

    # -----------------------
    # Label assign
    # -----------------------
    def _assign_labels(self):
        n = len(self._processed_data)
        self.labels_ = np.zeros(n, dtype=int)
        if self.n_micro_clusters_ == 0:
            return
        mc_to_macro = {int(self._mc_ids[i]): int(self._mc_macro_ids[i]) for i in range(len(self._mc_ids))}
        mc_ids_all = self._processed_data[:, 1].astype(int)
        self.labels_ = np.array([mc_to_macro.get(int(x), 0) for x in mc_ids_all], dtype=int)

    # -----------------------
    # sklearn-like helpers
    # -----------------------
    def get_params(self, deep: bool = True) -> Dict:
        return {'N': self.N, 'r': self.r, 'n_micro': self.n_micro, 'random_state': self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        # reset RNG if random_state changed
        self.rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random.RandomState()
        return self

    def score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        labels = self.fit_predict(X)
        if y is not None:
            return adjusted_rand_score(y, labels)
        else:
            return silhouette_score(X, labels)
