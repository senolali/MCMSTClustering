"""
examples/demo.py
~~~~~~~~~~~~~~~~
Demonstrates MCMSTClustering on several synthetic datasets from the paper.

Requirements (beyond the package itself):
    pip install matplotlib

Run:
    python examples/demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons, make_blobs

from mcmstclustering import MCMSTClustering


def make_two_spirals(n: int = 200, noise: float = 0.02, seed: int = 0):
    """Generate two interleaved spirals."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    noise_xy = rng.normal(0, noise, (n, 2))
    X0 = np.column_stack([t * np.cos(t), t * np.sin(t)]) + noise_xy
    X1 = np.column_stack([-t * np.cos(t), -t * np.sin(t)]) + noise_xy
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def make_outlier_blobs(seed: int = 0):
    """Four blobs with random outliers."""
    rng = np.random.default_rng(seed)
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.3, random_state=seed)
    outliers = rng.uniform(X.min(), X.max(), (20, 2))
    X = np.vstack([X, outliers])
    y = np.concatenate([y, [-1] * 20])
    return X, y


# --------------------------------------------------------------------------
datasets = {
    "Two Moons\n(N=4, r=0.06, n_micro=3)": (
        MinMaxScaler().fit_transform(make_moons(n_samples=400, noise=0.07, random_state=1)[0]),
        MCMSTClustering(N=4, r=0.06, n_micro=3),
    ),
    "Four Blobs\n(N=5, r=0.07, n_micro=3)": (
        MinMaxScaler().fit_transform(make_blobs(400, 4, cluster_std=0.4, random_state=2)[0]),
        MCMSTClustering(N=5, r=0.07, n_micro=3),
    ),
    "Two Spirals\n(N=4, r=0.08, n_micro=3)": (
        MinMaxScaler().fit_transform(make_two_spirals(200, noise=0.03)[0]),
        MCMSTClustering(N=4, r=0.08, n_micro=3),
    ),
    "Blobs + Outliers\n(N=5, r=0.07, n_micro=3)": (
        MinMaxScaler().fit_transform(make_outlier_blobs()[0]),
        MCMSTClustering(N=5, r=0.07, n_micro=3),
    ),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("MCMSTClustering — Demo", fontsize=14, fontweight="bold")

for ax, (title, (X, model)) in zip(axes.flat, datasets.items()):
    labels = model.fit_predict(X)
    noise_mask = labels == -1
    cluster_mask = ~noise_mask

    ax.scatter(
        X[cluster_mask, 0], X[cluster_mask, 1],
        c=labels[cluster_mask], cmap="tab10", s=12, alpha=0.8,
        label="cluster points",
    )
    if noise_mask.any():
        ax.scatter(
            X[noise_mask, 0], X[noise_mask, 1],
            c="lightgrey", s=12, marker="x", label="noise",
        )
    ax.scatter(
        model.micro_cluster_centers_[:, 0],
        model.micro_cluster_centers_[:, 1],
        c="black", s=30, marker="+", linewidths=1, label="MC centres",
    )
    ax.set_title(
        f"{title}\n"
        f"clusters={model.n_clusters_}  "
        f"MCs={len(model.micro_clusters_)}  "
        f"noise={(labels == -1).sum()}",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.savefig("mcmst_demo.png", dpi=150, bbox_inches="tight")
print("Saved mcmst_demo.png")
plt.show()
