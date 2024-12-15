from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

def compare_clustering_algorithms(X: np.ndarray, true_labels: np.ndarray, n_clusters: int = 7, random_state: int = 42) -> pd.DataFrame:
    """
    Compare different clustering algorithms on the same dataset.
    
    Args:
        X: Feature matrix to cluster
        true_labels: Ground truth labels for adjusted rand score
        n_clusters: Number of clusters to create
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame containing performance metrics for each algorithm
    """
    # Initialize clustering algorithms
    algorithms = {
        'K-Means': KMeans(
            n_clusters=n_clusters, 
            random_state=random_state, 
            n_init=10
            ),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
        'GMM': GaussianMixture(
            n_components=n_clusters, 
            covariance_type='full', 
            random_state=random_state
            )
    }
    
    # Store results
    results = []
    
    # Run and evaluate each algorithm
    for name, algorithm in algorithms.items():
        # Fit and predict clusters
        clusters = algorithm.fit_predict(X)
        
        # Calculate metrics
        metrics = {
            'Algorithm': name,
            'Silhouette Score': silhouette_score(X, clusters),
            'Davies-Bouldin Index': davies_bouldin_score(X, clusters),
            'Adjusted Rand Index': adjusted_rand_score(true_labels, clusters)
        }
        
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Set Algorithm as index for better display
    results_df.set_index('Algorithm', inplace=True)
    
    return results_df

def get_best_algorithm(results_df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    """
    Determine the best performing algorithm based on metrics.
    
    Args:
        results_df: DataFrame with clustering results
        
    Returns:
        Tuple of (best algorithm name, metrics for best algorithm)
    """
    # For Silhouette Score: higher is better
    # For Davies-Bouldin Index: lower is better
    # For Adjusted Rand Index: higher is better
    
    # Normalize scores to 0-1 range
    normalized_scores = results_df.copy()
    
    # Normalize Silhouette Score (higher is better)
    normalized_scores['Silhouette Score'] = (results_df['Silhouette Score'] - results_df['Silhouette Score'].min()) / \
                                          (results_df['Silhouette Score'].max() - results_df['Silhouette Score'].min())
    
    # Normalize Davies-Bouldin Index (lower is better, so invert)
    normalized_scores['Davies-Bouldin Index'] = 1 - (results_df['Davies-Bouldin Index'] - results_df['Davies-Bouldin Index'].min()) / \
                                              (results_df['Davies-Bouldin Index'].max() - results_df['Davies-Bouldin Index'].min())
    
    # Normalize Adjusted Rand Index (higher is better)
    normalized_scores['Adjusted Rand Index'] = (results_df['Adjusted Rand Index'] - results_df['Adjusted Rand Index'].min()) / \
                                             (results_df['Adjusted Rand Index'].max() - results_df['Adjusted Rand Index'].min())
    
    # Calculate mean score for each algorithm
    mean_scores = normalized_scores.mean(axis=1)
    
    # Get best algorithm
    best_algorithm = mean_scores.idxmax()
    
    return best_algorithm, results_df.loc[best_algorithm].to_dict()
