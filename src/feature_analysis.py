import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from scipy import stats

class FeatureAnalyzer:
    """Class for analyzing and visualizing feature distributions in a dataset."""
    
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """
        Initialize the analyzer with a dataset.
        
        Args:
            df: DataFrame containing features
            target_col: Optional column name for target variable (for grouped analysis)
        """
        self.df = df
        self.target_col = target_col
        
    def _create_feature_groups(self) -> dict:
        """Group features by their prefix to organize visualizations."""
        feature_groups = {}
        
        for col in self.df.columns:
            # Skip target column if specified
            if col == self.target_col:
                continue
                
            # Extract prefix (e.g., 'vertical_', 'horizontal_')
            prefix = col.split('_')[0] if '_' in col else 'other'
            
            if prefix not in feature_groups:
                feature_groups[prefix] = []
            feature_groups[prefix].append(col)
            
        return feature_groups
    
    def plot_feature_distributions(self, n_cols: int = 3, figsize_per_plot: tuple = (5, 4)):
        """
        Create distribution plots for all features.
        
        Args:
            n_cols: Number of columns in the subplot grid
            figsize_per_plot: Figure size for each individual plot
        """
        feature_groups = self._create_feature_groups()
        
        for group_name, features in feature_groups.items():
            n_features = len(features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig = plt.figure(figsize=(figsize_per_plot[0] * n_cols, 
                                    figsize_per_plot[1] * n_rows))
            fig.suptitle(f'{group_name.title()} Features Distribution', y=1.02, fontsize=16)
            
            for idx, feature in enumerate(features, 1):
                ax = plt.subplot(n_rows, n_cols, idx)
                
                # Create violin plot with boxplot inside
                sns.violinplot(data=self.df, y=feature, ax=ax)
                
                # Add individual points with jitter if not too many data points
                if len(self.df) < 1000:
                    sns.stripplot(data=self.df, y=feature, color='red', alpha=0.3, 
                                size=2, jitter=0.2, ax=ax)
                
                # Calculate basic statistics
                mean_val = self.df[feature].mean()
                median_val = self.df[feature].median()
                skew_val = stats.skew(self.df[feature].dropna())
                
                # Add statistics to plot
                stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nSkew: {skew_val:.2f}'
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8))
                
                # Rotate x-labels if needed
                plt.xticks(rotation=45)
                
                # Set title as feature name
                ax.set_title(feature.replace('_', ' ').title())
            
            plt.tight_layout()
            plt.show()
    
    def plot_correlation_matrix(self, figsize: tuple = (12, 10)):
        """
        Create a correlation matrix heatmap for all features.
        
        Args:
            figsize: Size of the correlation matrix plot
        """
        # Calculate correlation matrix
        corr_matrix = self.df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, center=0, cmap='coolwarm',
                   annot=True, fmt='.2f', square=True, linewidths=.5,
                   cbar_kws={"shrink": .5})
        
        plt.title('Feature Correlation Matrix', pad=20)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, n_features: int = 20):
        """
        Plot feature importance based on variance.
        
        Args:
            n_features: Number of top features to show
        """
        # Calculate variance for each feature
        variances = self.df.var().sort_values(ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(12, 6))
        variances.head(n_features).plot(kind='bar')
        plt.title('Feature Importance (Based on Variance)')
        plt.xlabel('Features')
        plt.ylabel('Variance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def analyze_features(self):
        """Perform complete feature analysis with all visualizations."""
        print("=== Feature Analysis Report ===")
        print(f"\nNumber of features: {len(self.df.columns)}")
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        print("\nPlotting feature distributions...")
        self.plot_feature_distributions()
        
        print("\nPlotting correlation matrix...")
        self.plot_correlation_matrix()
        
        print("\nPlotting feature importance...")
        self.plot_feature_importance()
