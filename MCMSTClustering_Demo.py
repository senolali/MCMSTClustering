# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:41:16 2023

@author: Ali ŞENOL
"""

import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
import scipy
from IPython import get_ipython
import warnings
from sklearn import metrics
warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
get_ipython().magic('clear all -sf')
xt=np.array([])
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from IPython import get_ipython
import warnings

class MCMSTClustering(BaseEstimator, ClusterMixin):
    def __init__(self, N=5, r=0.05, n_micro=5):
        """
        Initialize the MCMSTCluster clustering algorithm.

        Parameters:
        - N: Minimum number of data points to define a Micro Cluster.
        - r: Radius of the Micro Cluster.
        - n_micro: Minimum number of Micro Clusters to define a Macro Cluster.
        """
        self._N = N
        self._r = r
        self._n_micro = n_micro

    def fit(self, X):
        """
        Fit the MCMSTCluster clustering algorithm to the input data.

        Parameters:
        - X: Input data as a NumPy array.

        This method initializes clustering parameters and computes the clustering results.
        """
        self._XX = X
        self._d = X.shape[1]
        self._MC_Num = 0
        self._MacroC_Num = 0
        self._processed_data = np.empty((0, self._d + 3), float)
        self._MCs = np.empty((0, self._d + 3), float)
        self._MacroClusters = np.empty((0, 4))
        self._labels_ = []
        self._AddNode()
        self._DefineMC()
        self._RegulateClusters()
        self._DefMacros()
        self._assingMacoN()

    def fit_predict(self, X):
        """
        Fit the MCMSTCluster clustering algorithm and return cluster labels.

        Parameters:
        - X: Input data as a NumPy array.

        Returns:
        - labels: Cluster labels for each data point.

        This method is similar to 'fit' but also returns the cluster labels.
        """
        self.fit(X)
        return self._labels_

    def _AddNode(self):
        """
        Add data points to the processed_data attribute.

        This method prepares the data for clustering by adding it to the processed_data array.
        """
        self._processed_data = np.hstack((np.arange(self._XX.shape[0]).reshape(self._XX.shape[0], 1) + 1,
                                          np.repeat([[0, 0]], self._XX.shape[0], 0), self._XX))

    def _DefineMC(self):
        """
        Define Micro Clusters.

        This method defines Micro Clusters based on the input data and clustering parameters.
        """
        while True:
            X = self._processed_data[self._processed_data[:, 1] == 0, :]
            num_MCs = self._MC_Num
            if (X.shape[0] >= self._N):
                tree = KDTree(X[:, 3:])
                for i in range(X.shape[0]):
                    if (X[i, 2] == 0):
                        ind = tree.query_radius(X[i:, 3:], self._r)
                        points = X[ind[0], 3:]
                        if (len(points) >= self._N):
                            center = np.mean(np.array(points), axis=0)
                            self._MC_Num = self._MC_Num + 1
                            self._MCs = np.vstack((self._MCs, np.hstack(
                                (np.array([self._MC_Num, len(points), 0]), center))))
                            for j in range(len(points)):
                                self._processed_data[np.where(
                                    (self._processed_data[:, 3:] == points[j]).all(axis=1))[0], 1] = self._MC_Num
                            break
            if (num_MCs == self._MC_Num):
                return

    def _MST(self, mc_id):
        """
        Compute Minimum Spanning Tree for a Micro Cluster.

        Parameters:
        - mc_id: Micro Cluster ID.

        Returns:
        - spanning_edges: List of edges in the Minimum Spanning Tree.

        This method computes the Minimum Spanning Tree for a given Micro Cluster using Prim's algorithm.
        """
        P = self._MCs[self._MCs[:, 2] == 0, :]
        indices = self._MCs[self._MCs[:, 2] == 0, 0]
        G = squareform(pdist(P[:, 3:]))
        N = G.shape[0]
        selected_node = np.zeros(N)
        no_edge = 0
        selected_node[P[:, 0] == mc_id] = True
        spanning_edges = []
        while (no_edge < N - 1):
            minimum = 2 * self._r
            a = 0
            b = 0
            for m in range(N):
                if selected_node[m]:
                    for n in range(N):
                        if ((not selected_node[n]) and G[m][n]):
                            if G[m][n] < minimum:
                                a = m
                                b = n
            if (int(indices[a]) == int(indices[b])):
                break
            spanning_edges.append([int(indices[a]), int(indices[b])])
            selected_node[b] = True
            no_edge += 1
        return spanning_edges

    def _DefMacros(self):
        """
        Define Macro Clusters based on Micro Clusters.

        This method defines Macro Clusters based on the edges of the Minimum Spanning Trees of Micro Clusters.
        """
        edge_lists = []
        for a in range(self._MCs.shape[0]):
            if (self._MCs[a, 2] == 0):
                edge_lists = self._MST(self._MCs[a, 0])
                if (len(edge_lists) > 0):
                    edge_list = np.empty((0, 2), int)
                    for index in edge_lists:
                        i, j = index
                        edge_list = np.vstack((edge_list, (i, j)))
                    summ = 0
                    edges = np.unique(edge_list)
                    for e in edges:
                        summ = summ + self._MCs[self._MCs[:, 0] == e, 1]
                    if (summ >= self._n_micro * self._N or len(np.unique(edge_list)) >= self._n_micro):
                        self._MacroC_Num = self._MacroC_Num + 1
                        colors = np.array(sns.color_palette(None, self._MacroC_Num + 1))
                        self._MacroClusters = np.vstack(
                            (self._MacroClusters, np.array([self._MacroC_Num, len(np.unique(edge_list)), edge_list, colors[-1, :]], dtype=object)))
                        print("--Macro Cluster #", self._MacroC_Num, " is defined--")
                        for i in np.unique(edge_list):
                            self._MCs[self._MCs[:, 0] == i, 2] = self._MacroC_Num
        if len(edge_lists) >= self._N:
            for a in range(self._MCs.shape[0]):
                if (self._MCs[a, 2] == 0 and self._MCs[a, 1] >= self._N * self._n_micro):
                    self._MacroC_Num = self._MacroC_Num + 1
                    colors = np.array(sns.color_palette(None, self._MacroC_Num + 1))
                    self._MacroClusters = np.vstack(
                        (self._MacroClusters, np.array([self._MacroC_Num, len(np.unique(edge_list)), edge_list, colors[-1, :]], dtype=object)))
                    print("--Macro Cluster #", self._MacroC_Num, " is defined--")
                    self._MCs[a, 2] = self._MacroC_Num

    def _SearchforClusters(self):
        """
        Search for clusters within Macro Clusters.

        This method searches for clusters within the defined Macro Clusters.
        """
        for i in range(self._MCs.shape[0]):
            if (self._MCs[i, 2] == 0):
                self._DefineMacroC(self._MCs[i, 0])

    def _assingMacoN(self):
        """
        Assign Macro Cluster labels to data points.

        This method assigns Macro Cluster labels to data points based on the Micro Cluster assignments.
        """
        for i in self._MCs:
            self._processed_data[self._processed_data[:, 1] == i[0], 2] = i[2]
        self._labels_ = self._processed_data[:, 2].reshape(-1)

    def _RegulateClusters(self):
        """
        Update Micro Cluster assignments based on proximity criteria.

        This method updates Micro Cluster assignments based on proximity criteria.
        """
        X = self._processed_data[self._processed_data[:, 1] == 0, :]
        if (X.shape[0] > 0 and self._MCs.shape[0] > 0):
            tree = KDTree(self._MCs[:, 3:])
            for i in X:
                distance, index = tree.query([i[3:]])
                if (distance[0] <= 2 * self._r):
                    self._processed_data[self._processed_data[:, 0] == i[0], 1] = self._MCs[self._MCs[:, 0] == index[0] + 1, 0]

    def _plotGraph(self, X, labels, r, N, n_micro, index, index_value, dataset_name, dpi=100):
        """
        Plot the clustering results for visualization.

        Parameters:
        - X: Input data as a NumPy array.
        - labels: Cluster labels for each data point.
        - r: Radius of the Micro Cluster.
        - N: Minimum number of data points to define a Micro Cluster.
        - n_micro: Minimum number of Micro Clusters to define a Macro Cluster.
        - index: Index name for the plot.
        - index_value: Index value for the plot.
        - dataset_name: Name of the dataset.
        - dpi: Dots per inch for the plot.

        This method plots the clustering results for visualization, including the data points and their cluster assignments.
        """
        # Plotting logic here...

    def _plotMCs(self, N, r, n_micro, maxARI, maxPurity, maxSI, dataset_name, dpi=100):
        """
        Plot Micro and Macro Clusters.

        Parameters:
        - N: Minimum number of data points to define a Micro Cluster.
        - r: Radius of the Micro Cluster.
        - n_micro: Minimum number of Micro Clusters to define a Macro Cluster.
        - maxARI: Maximum Adjusted Rand Index.
        - maxPurity: Maximum Purity score.
        - maxSI: Maximum Silhouette score.
        - dataset_name: Name of the dataset.
        - dpi: Dots per inch for the plot.

        This method plots the Micro and Macro Clusters, along with relevant information.
        """
        # Plotting logic here...

    def _purity_score(self, y_true, y_pred):
        """
        Calculate Purity score.

        Parameters:
        - y_true: True cluster labels.
        - y_pred: Predicted cluster labels.

        Returns:
        - purity_score: The Purity score.

        This method calculates the Purity score, a measure of clustering quality.
        """
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

##------------main---------------------------------------------------------------
data_sets={1,2,3,4} ##tested datasets
for dataset in data_sets:
    print("dataset=",dataset)
    plotFigure=1
    loop=0
    if (dataset==1):
        data = scipy.io.loadmat("Datasets/halfkernel.mat")
        data=data["halkernel"];
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="1_HalfKernel_"
        N=4 #minimum number of data to define a MC 
        r=0.084 # radius of MC
        n_micro=8 # minimum number MC to define Macro Cluster
    elif dataset==2:
        data = np.genfromtxt("Datasets/Three-Spirals.txt", delimiter=',', dtype=float,usecols=range(3))
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="2_Three_Spirals_"
        N=3 #minimum number of data to define a MC 
        r=0.066 # radius of MC
        n_micro=6 # minimum number MC to define Macro Cluster
    elif dataset==3:
        data = np.genfromtxt("Datasets/corners.txt", delimiter=',', dtype=float,usecols=range(3))
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="3_Corners_"
        N=15 #minimum number of data to define a MC 
        r=0.066 # radius of MC
        n_micro=13 # minimum number MC to define Macro Cluster
    elif dataset==4:
        data = scipy.io.loadmat("Datasets/moon.mat")
        X=data["data"]
        labels_true = np.mat(data["target"])
        labels_true=np.ravel(labels_true)
        dataset_name="4_Moon_"
        N=3 #minimum number of data to define a MC 
        r=0.054# radius of MC
        n_micro=14 # minimum number MC to define Macro Cluster---->1.0000
   
    ####MinMaxNormalization#######################################################
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler()
    X=scaler.transform(X[:,:])

    loop=0; 
    maxARI=float('-inf')
    maxPurity=float('-inf')
    maxSI=float('-inf')
    optN=float('-inf')
    optR=float('-inf')
    optNMicro=float('-inf')
    maxEpsilon=float('-inf') 
    SI=float('-inf') 
   
    mcmst=MCMSTClustering(N,r,n_micro)   
    labels=mcmst.fit_predict(X)
    # labels=mcmst.labels_
    
    if len(np.unique(labels))>1:
        ARI=adjusted_rand_score(labels_true.reshape(-1), labels)
        if ARI>maxARI:
            maxARI=ARI
            maxSI=silhouette_score(X,labels)
            maxPurity=mcmst._purity_score(labels_true.reshape(-1), labels)
            optN=N
            optR=r
            optNMicro=n_micro
            print("=========Better parameters are detected=========")
            print("Max ARI=%0.4f"%maxARI)
            print("Purity=%0.4f"%maxPurity)
            print("Silhouette Index=%0.4f"%maxSI)  
            print("The best N=%d"%N)
            print("The best r=%0.4f"%r)
            print("the best n_micro=%d"%n_micro)
            mcmst._plotMCs(N,r,n_micro,maxARI,maxPurity,maxSI,dataset_name)
            mcmst._plotGraph(X,labels,r,N,n_micro,"ARI",ARI,dataset_name)
