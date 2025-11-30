
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: visualization_module.py
# execution: true

"""
Visualization Module for Portfolio Clustering Framework
========================================================
Generates publication-quality visualizations for educational purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PortfolioVisualizer:
    """
    Visualization class for portfolio clustering analysis.
    
    Generates:
    - Dendrogram (hierarchical clustering tree)
    - Correlation heatmap
    - Cluster correlation heatmap
    - Portfolio composition pie chart
    - Historical performance comparison
    """
    
    def __init__(self, framework):
        """
        Initialize visualizer with framework instance.
        
        Parameters:
        -----------
        framework : PortfolioClusteringFramework
            Initialized framework with completed analysis
        """
        self.framework = framework
        self.figures = {}
        
    def plot_dendrogram(self, figsize=(14, 8), save_path='dendrogram.png'):
        """
        Plot hierarchical clustering dendrogram.
        
        Educational Note:
        - Height represents distance between clusters
        - Horizontal lines show cluster merges
        - Vertical distance indicates dissimilarity
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        save_path : str
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create dendrogram
        dendro = dendrogram(
            self.framework.linkage_matrix,
            labels=self.framework.tickers,
            ax=ax,
            leaf_font_size=12,
            color_threshold=0.7 * max(self.framework.linkage_matrix[:, 2])
        )
        
        ax.set_title('Hierarchical Clustering Dendrogram\nAsset Classification by Correlation',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Assets', fontsize=14, fontweight='bold')
        ax.set_ylabel('Distance (Dissimilarity)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add educational annotation
        ax.text(0.02, 0.98, 
                'Lower height = More similar assets\nClusters form at merge points',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dendrogram saved: {save_path}")
        
        self.figures['dendrogram'] = fig
        return fig
    
    def plot_correlation_heatmap(self, figsize=(12, 10), save_path='correlation_heatmap.png'):
        """
        Plot correlation matrix heatmap.
        
        Educational Note:
        - Red = Positive correlation (assets move together)
        - Blue = Negative correlation (assets move opposite)
        - Darker colors = Stronger relationships
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(self.framework.correlation_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            self.framework.correlation_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            ax=ax,
            vmin=-1,
            vmax=1
        )
        
        ax.set_title('Asset Correlation Matrix\nPearson Correlation of Log Returns',
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add educational annotation
        ax.text(1.15, 0.5,
                'Interpretation:\n\n'
                '  1.0 = Perfect positive\n'
                '  0.0 = No correlation\n'
                ' -1.0 = Perfect negative\n\n'
                'Diversification benefits\n'
                'from low correlations',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved: {save_path}")
        
        self.figures['correlation_heatmap'] = fig
        return fig
    
    def plot_cluster_heatmap(self, figsize=(12, 10), save_path='cluster_heatmap.png'):
        """
        Plot correlation heatmap organized by clusters.
        
        Educational Note:
        - Assets grouped by cluster assignment
        - Shows within-cluster vs between-cluster correlations
        - Helps visualize cluster quality
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str
            Path to save figure
        """
        # Reorder correlation matrix by clusters
        cluster_order = []
        cluster_labels = []
        
        for cluster_id in sorted(self.framework.clusters.keys()):
            assets = self.framework.clusters[cluster_id]
            cluster_order.extend(assets)
            cluster_labels.extend([f"C{cluster_id}"] * len(assets))
        
        reordered_corr = self.framework.correlation_matrix.loc[cluster_order, cluster_order]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            reordered_corr,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            ax=ax,
            vmin=-1,
            vmax=1
        )
        
        # Add cluster boundaries
        cumsum = 0
        for cluster_id in sorted(self.framework.clusters.keys()):
            n_assets = len(self.framework.clusters[cluster_id])
            ax.add_patch(Rectangle((cumsum, cumsum), n_assets, n_assets,
                                   fill=False, edgecolor='black', lw=3))
            cumsum += n_assets
        
        ax.set_title('Correlation Matrix Organized by Clusters\nAssets Grouped by Hierarchical Clustering',
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add cluster labels on right side
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(np.arange(len(cluster_order)) + 0.5)
        ax2.set_yticklabels(cluster_labels, fontsize=8)
        ax2.set_ylabel('Cluster ID', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cluster heatmap saved: {save_path}")
        
        self.figures['cluster_heatmap'] = fig
        return fig
    
    def plot_portfolio_composition(self, figsize=(14, 8), save_path='portfolio_composition.png'):
        """
        Plot portfolio composition by asset and cluster.
        
        Educational Note:
        - Shows weight allocation across assets
        - Grouped by cluster for diversification visualization
        - Equal cluster weighting ensures balanced exposure
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Map assets to clusters
        asset_to_cluster = {}
        for cluster_id, assets in self.framework.clusters.items():
            for asset in assets:
                asset_to_cluster[asset] = cluster_id
        
        # Prepare data
        weights_df = pd.DataFrame({
            'Asset': self.framework.portfolio_weights.index,
            'Weight': self.framework.portfolio_weights.values,
            'Cluster': [asset_to_cluster[a] for a in self.framework.portfolio_weights.index]
        }).sort_values('Cluster')
        
        # Plot 1: Bar chart by asset
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.framework.clusters)))
        cluster_colors = {cid: colors[i] for i, cid in enumerate(sorted(self.framework.clusters.keys()))}
        bar_colors = [cluster_colors[asset_to_cluster[a]] for a in weights_df['Asset']]
        
        bars = ax1.bar(range(len(weights_df)), weights_df['Weight'], color=bar_colors, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(weights_df)))
        ax1.set_xticklabels(weights_df['Asset'], rotation=45, ha='right')
        ax1.set_ylabel('Portfolio Weight', fontsize=12, fontweight='bold')
        ax1.set_title('Portfolio Weights by Asset\nGrouped by Cluster', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add cluster labels
        for i, (asset, cluster) in enumerate(zip(weights_df['Asset'], weights_df['Cluster'])):
            ax1.text(i, weights_df.iloc[i]['Weight'] + 0.005, f'C{cluster}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Plot 2: Pie chart by cluster
        cluster_weights = weights_df.groupby('Cluster')['Weight'].sum()
        
        wedges, texts, autotexts = ax2.pie(
            cluster_weights.values,
            labels=[f'Cluster {cid}\n({len(self.framework.clusters[cid])} assets)' 
                   for cid in cluster_weights.index],
            autopct='%1.1f%%',
            colors=[cluster_colors[cid] for cid in cluster_weights.index],
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        ax2.set_title('Portfolio Allocation by Cluster\nEqual Cluster Weighting Strategy', 
                     fontsize=14, fontweight='bold')
        
        # Add legend with asset names
        legend_labels = []
        for cid in sorted(self.framework.clusters.keys()):
            assets = ', '.join(self.framework.clusters[cid])
            legend_labels.append(f"C{cid}: {assets}")
        
        ax2.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=9, title='Cluster Assets', title_fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Portfolio composition saved: {save_path}")
        
        self.figures['portfolio_composition'] = fig
        return fig
    
    def plot_performance(self, figsize=(14, 10), save_path='performance_comparison.png'):
        """
        Plot historical performance comparison.
        
        Educational Note:
        - Compares clustered portfolio vs equal-weight portfolio
        - Shows cumulative returns over time
        - Demonstrates diversification benefits
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Calculate portfolio returns
        clustered_returns = (self.framework.returns * self.framework.portfolio_weights).sum(axis=1)
        equal_weight = pd.Series(1/len(self.framework.tickers), index=self.framework.tickers)
        equal_returns = (self.framework.returns * equal_weight).sum(axis=1)
        
        # Calculate cumulative returns
        clustered_cumulative = (1 + clustered_returns).cumprod()
        equal_cumulative = (1 + equal_returns).cumprod()
        
        # Plot 1: Cumulative returns
        ax1 = axes[0, 0]
        ax1.plot(clustered_cumulative.index, clustered_cumulative.values, 
                label='Clustered Portfolio', linewidth=2, color='darkblue')
        ax1.plot(equal_cumulative.index, equal_cumulative.values,
                label='Equal-Weight Portfolio', linewidth=2, color='darkred', linestyle='--')
        ax1.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=10, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # Plot 2: Rolling volatility (30-day)
        ax2 = axes[0, 1]
        clustered_vol = clustered_returns.rolling(30).std() * np.sqrt(252)
        equal_vol = equal_returns.rolling(30).std() * np.sqrt(252)
        
        ax2.plot(clustered_vol.index, clustered_vol.values,
                label='Clustered Portfolio', linewidth=2, color='darkblue')
        ax2.plot(equal_vol.index, equal_vol.values,
                label='Equal-Weight Portfolio', linewidth=2, color='darkred', linestyle='--')
        ax2.set_title('Rolling Volatility (30-day, Annualized)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volatility', fontsize=10, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot 3: Drawdown
        ax3 = axes[1, 0]
        clustered_dd = (clustered_cumulative / clustered_cumulative.cummax() - 1)
        equal_dd = (equal_cumulative / equal_cumulative.cummax() - 1)
        
        ax3.fill_between(clustered_dd.index, 0, clustered_dd.values,
                        label='Clustered Portfolio', alpha=0.5, color='darkblue')
        ax3.fill_between(equal_dd.index, 0, equal_dd.values,
                        label='Equal-Weight Portfolio', alpha=0.5, color='darkred')
        ax3.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown', fontsize=10, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot 4: Performance metrics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate metrics
        metrics = {
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'Calmar Ratio'
            ],
            'Clustered': [
                f"{(clustered_cumulative.iloc[-1] - 1):.2%}",
                f"{clustered_returns.mean() * 252:.2%}",
                f"{clustered_returns.std() * np.sqrt(252):.2%}",
                f"{(clustered_returns.mean() * 252) / (clustered_returns.std() * np.sqrt(252)):.2f}",
                f"{clustered_dd.min():.2%}",
                f"{(clustered_returns.mean() * 252) / abs(clustered_dd.min()):.2f}"
            ],
            'Equal-Weight': [
                f"{(equal_cumulative.iloc[-1] - 1):.2%}",
                f"{equal_returns.mean() * 252:.2%}",
                f"{equal_returns.std() * np.sqrt(252):.2%}",
                f"{(equal_returns.mean() * 252) / (equal_returns.std() * np.sqrt(252)):.2f}",
                f"{equal_dd.min():.2%}",
                f"{(equal_returns.mean() * 252) / abs(equal_dd.min()):.2f}"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics)
        
        table = ax4.table(cellText=metrics_df.values,
                         colLabels=metrics_df.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(metrics_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(metrics_df) + 1):
            for j in range(len(metrics_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax4.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Portfolio Performance Analysis\nCluster-Based vs Equal-Weight Strategy',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Performance comparison saved: {save_path}")
        
        self.figures['performance'] = fig
        return fig
    
    def generate_all_visualizations(self):
        """
        Generate all visualizations at once.
        
        Returns:
        --------
        dict : Dictionary of all figure objects
        """
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.plot_dendrogram()
        self.plot_correlation_heatmap()
        self.plot_cluster_heatmap()
        self.plot_portfolio_composition()
        self.plot_performance()
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS COMPLETE")
        print("="*60)
        
        return self.figures


# Example usage
if __name__ == "__main__":
    # Import the framework
    from portfolio_clustering_framework import PortfolioClusteringFramework
    
    # Run analysis
    print("Running portfolio analysis...")
    framework = PortfolioClusteringFramework()
    results = framework.run_full_analysis()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = PortfolioVisualizer(framework)
    figures = visualizer.generate_all_visualizations()
    
    print("\n✓ All visualizations generated successfully!")
    print("✓ Files saved in current directory")