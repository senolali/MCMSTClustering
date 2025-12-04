"""
Utility functions for MCMSTClustering.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import ParameterSampler


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate purity score for clustering.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True cluster labels.
    y_pred : array-like, shape (n_samples,)
        Predicted cluster labels.
        
    Returns
    -------
    purity : float
        Purity score between 0 and 1.
    """
    from sklearn.metrics import confusion_matrix
    
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def find_optimal_parameters(X: np.ndarray, y: np.ndarray = None, 
                           n_iter: int = 100, random_state: Optional[int] = None) -> Dict:
    """
    Find optimal parameters for MCMSTClustering using random search.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster.
    y : array-like, shape (n_samples,), optional
        True labels for evaluation.
    n_iter : int, default=100
        Number of parameter settings to sample.
    random_state : int or None, default=None
        Random seed.
        
    Returns
    -------
    best_params : dict
        Best parameters found.
    best_score : float
        Best score achieved.
    """
    from .core import MCMSTClustering
    
    # Define parameter distributions
    param_dist = {
        'N': list(range(2, 20)),
        'r': list(np.arange(0.01, 0.5, 0.01)),
        'n_micro': list(range(2, 10))
    }
    
    best_score = -1
    best_params = None
    
    for params in ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state):
        try:
            model = MCMSTClustering(**params, random_state=random_state)
            labels = model.fit_predict(X)
            
            if y is not None:
                score = adjusted_rand_score(y, labels)
            else:
                from sklearn.metrics import silhouette_score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                else:
                    continue
            
            if score > best_score:
                best_score = score
                best_params = params
                
        except Exception:
            continue
    
    return {'params': best_params, 'score': best_score}


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Evaluate clustering results with multiple metrics.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True cluster labels.
    y_pred : array-like, shape (n_samples,)
        Predicted cluster labels.
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                homogeneity_score, completeness_score, v_measure_score)
    
    return {
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
        'Purity': purity_score(y_true, y_pred),
        'Homogeneity': homogeneity_score(y_true, y_pred),
        'Completeness': completeness_score(y_true, y_pred),
        'V-measure': v_measure_score(y_true, y_pred)
    }