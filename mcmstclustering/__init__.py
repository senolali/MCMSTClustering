"""
MCMSTClustering: Defining Non-Spherical Clusters by using Minimum Spanning
Tree over KD-Tree-based Micro-Clusters.

Reference:
    Şenol, A. (2023). MCMSTClustering: defining non-spherical clusters by using
    minimum spanning tree over KD-tree-based micro-clusters.
    Neural Computing and Applications, 35, 13239–13259.
    https://doi.org/10.1007/s00521-023-08386-3
"""

from .clustering import MCMSTClustering
from .version import __version__

__all__ = ["MCMSTClustering", "__version__"]
__author__ = "Ali Şenol"
__email__ = "alisenol@tarsus.edu.tr"
