import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class UnsupervisedEvaluator:
    def __init__(self, X, true_labels):
        """
        Initialize evaluator with data and ground truth labels.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            true_labels: Ground truth labels
        """
        self.X = X
        self.le = LabelEncoder()
        self.true_labels_encoded = self.le.fit_transform(true_labels)
        self.n_classes = len(np.unique(true_labels))
        
    def evaluate_clustering(self, cluster_labels):
        """
        Evaluate clustering results using multiple metrics.
        
        Args:
            cluster_labels: Predicted cluster assignments
            
        Returns:
            dict: Dictionary containing various evaluation metrics
        """
        metrics = {
            'adjusted_rand': adjusted_rand_score(self.true_labels_encoded, cluster_labels),
            'normalized_mutual_info': normalized_mutual_info_score(
                self.true_labels_encoded, cluster_labels
            ),
            'adjusted_mutual_info': adjusted_mutual_info_score(
                self.true_labels_encoded, cluster_labels
            ),
            'homogeneity': homogeneity_score(self.true_labels_encoded, cluster_labels),
            'completeness': completeness_score(self.true_labels_encoded, cluster_labels),
            'silhouette': silhouette_score(self.X, cluster_labels)
        }
        return metrics
    
    def plot_confusion_matrix(self, cluster_labels, figsize=(10, 8)):
        """
        Plot confusion matrix between true labels and cluster assignments.
        
        Args:
            cluster_labels: Predicted cluster assignments
            figsize: Size of the plot
        """
        # Create confusion matrix
        conf_mat = np.zeros((self.n_classes, len(np.unique(cluster_labels))))
        for i in range(len(self.true_labels_encoded)):
            conf_mat[self.true_labels_encoded[i], cluster_labels[i]] += 1
            
        # Normalize by row
        conf_mat_norm = conf_mat / conf_mat.sum(axis=1, keepdims=True)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(conf_mat_norm, 
                   xticklabels=[f'Cluster {i}' for i in range(conf_mat.shape[1])],
                   yticklabels=self.le.classes_,
                   annot=True, 
                   fmt='.2f', 
                   cmap='YlOrRd')
        plt.title('Normalized Confusion Matrix: True Labels vs Cluster Assignments')
        plt.ylabel('True Label')
        plt.xlabel('Cluster Assignment')
        
    def evaluate_nmf(self, W, H, n_top_features=10):
        """
        Evaluate NMF results and analyze components.
        
        Args:
            W: NMF weight matrix
            H: NMF component matrix
            n_top_features: Number of top features to analyze per component
            
        Returns:
            dict: Dictionary containing evaluation metrics and component analysis
        """
        # Cluster assignments based on maximum weight in W
        nmf_labels = np.argmax(W, axis=1)
        
        # Calculate clustering metrics
        clustering_metrics = self.evaluate_clustering(nmf_labels)
        
        # Analyze components
        component_analysis = []
        for i in range(H.shape[0]):
            # Get top features for this component
            top_idx = np.argsort(-H[i])[:n_top_features]
            component_analysis.append({
                'component': i,
                'top_features': top_idx,
                'weights': H[i][top_idx]
            })
        
        return {
            'clustering_metrics': clustering_metrics,
            'component_analysis': component_analysis
        }
    
    def plot_nmf_components(self, H, image_shape, n_components=None, figsize=(15, 10)):
        """
        Visualize NMF components as images.
        
        Args:
            H: NMF component matrix
            image_shape: Original shape of images (height, width)
            n_components: Number of components to plot (default: all)
            figsize: Size of the plot
        """
        if n_components is None:
            n_components = H.shape[0]
        
        n_cols = min(5, n_components)
        n_rows = (n_components - 1) // n_cols + 1
        
        plt.figure(figsize=figsize)
        for i in range(min(n_components, H.shape[0])):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(H[i].reshape(image_shape), cmap='viridis')
            plt.title(f'Component {i}')
            plt.axis('off')
        plt.tight_layout()
        
    def plot_reconstruction_quality(self, X_original, X_reconstructed, n_samples=5, figsize=(15, 6)):
        """
        Compare original and reconstructed images.
        
        Args:
            X_original: Original image data
            X_reconstructed: Reconstructed image data from NMF
            n_samples: Number of random samples to plot
            figsize: Size of the plot
        """
        indices = np.random.choice(X_original.shape[0], n_samples, replace=False)
        
        plt.figure(figsize=figsize)
        for i, idx in enumerate(indices):
            # Original
            plt.subplot(2, n_samples, i + 1)
            plt.imshow(X_original[idx].reshape(int(np.sqrt(X_original.shape[1])), -1), cmap='viridis')
            plt.title(f'Original {i+1}')
            plt.axis('off')
            
            # Reconstructed
            plt.subplot(2, n_samples, n_samples + i + 1)
            plt.imshow(X_reconstructed[idx].reshape(int(np.sqrt(X_reconstructed.shape[1])), -1), cmap='viridis')
            plt.title(f'Reconstructed {i+1}')
            plt.axis('off')
        plt.tight_layout()