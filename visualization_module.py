
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
Module de Visualisation 
=============================
Génère toutes les visualisations dans le dossier de sortie du framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from matplotlib.patches import Rectangle
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PortfolioVisualizer:
    """
    Classe de visualisation pour le framework de clustering.
    Toutes les images sont sauvegardées dans le dossier visualizations/.
    """
    
    def __init__(self, framework):
        """
        Initialiser avec une instance du framework.
        
        Parameters:
        -----------
        framework : PortfolioClusteringFramework
            Instance du framework avec analyse complétée
        """
        self.framework = framework
        self.viz_dir = os.path.join(framework.output_dir, 'visualizations')
        self.figures = {}
        
    def plot_dendrogram(self, figsize=(14, 8)):
        """Générer le dendrogramme."""
        save_path = os.path.join(self.viz_dir, 'dendrogram.png')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        dendrogram(
            self.framework.linkage_matrix,
            labels=self.framework.tickers,
            ax=ax,
            leaf_font_size=12,
            color_threshold=0.7 * max(self.framework.linkage_matrix[:, 2])
        )
        
        ax.set_title('Dendrogramme de Clustering Hiérarchique\nClassification des Actifs par Corrélation',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Actifs', fontsize=14, fontweight='bold')
        ax.set_ylabel('Distance (Dissimilarité)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        ax.text(0.02, 0.98, 
                'Hauteur basse = Actifs similaires\nLes clusters se forment aux points de fusion',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Dendrogramme: {save_path}")
        self.figures['dendrogram'] = save_path
        return save_path
    
    def plot_correlation_heatmap(self, figsize=(12, 10)):
        """Générer la heatmap de corrélation."""
        save_path = os.path.join(self.viz_dir, 'correlation_heatmap.png')
        
        fig, ax = plt.subplots(figsize=figsize)
        
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
            cbar_kws={"shrink": 0.8, "label": "Coefficient de Corrélation"},
            ax=ax,
            vmin=-1,
            vmax=1
        )
        
        ax.set_title('Matrice de Corrélation des Actifs\nCorrélation de Pearson des Rendements Logarithmiques',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.text(1.15, 0.5,
                'Interprétation:\n\n'
                '  1.0 = Corrélation positive parfaite\n'
                '  0.0 = Aucune corrélation\n'
                ' -1.0 = Corrélation négative parfaite\n\n'
                'Bénéfices de diversification\n'
                'avec corrélations faibles',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Heatmap de corrélation: {save_path}")
        self.figures['correlation_heatmap'] = save_path
        return save_path
    
    def plot_cluster_heatmap(self, figsize=(12, 10)):
        """Générer la heatmap organisée par clusters."""
        save_path = os.path.join(self.viz_dir, 'cluster_heatmap.png')
        
        cluster_order = []
        cluster_labels = []
        
        for cluster_id in sorted(self.framework.clusters.keys()):
            assets = self.framework.clusters[cluster_id]
            cluster_order.extend(assets)
            cluster_labels.extend([f"C{cluster_id}"] * len(assets))
        
        reordered_corr = self.framework.correlation_matrix.loc[cluster_order, cluster_order]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            reordered_corr,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Coefficient de Corrélation"},
            ax=ax,
            vmin=-1,
            vmax=1
        )
        
        cumsum = 0
        for cluster_id in sorted(self.framework.clusters.keys()):
            n_assets = len(self.framework.clusters[cluster_id])
            ax.add_patch(Rectangle((cumsum, cumsum), n_assets, n_assets,
                                   fill=False, edgecolor='black', lw=3))
            cumsum += n_assets
        
        ax.set_title('Matrice de Corrélation Organisée par Clusters\nActifs Groupés par Clustering Hiérarchique',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(np.arange(len(cluster_order)) + 0.5)
        ax2.set_yticklabels(cluster_labels, fontsize=8)
        ax2.set_ylabel('ID Cluster', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Heatmap par cluster: {save_path}")
        self.figures['cluster_heatmap'] = save_path
        return save_path
    
    def plot_portfolio_composition(self, figsize=(14, 8)):
        """Générer le graphique de composition du portefeuille."""
        save_path = os.path.join(self.viz_dir, 'portfolio_composition.png')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        asset_to_cluster = {}
        for cluster_id, assets in self.framework.clusters.items():
            for asset in assets:
                asset_to_cluster[asset] = cluster_id
        
        weights_df = pd.DataFrame({
            'Asset': self.framework.portfolio_weights.index,
            'Weight': self.framework.portfolio_weights.values,
            'Cluster': [asset_to_cluster[a] for a in self.framework.portfolio_weights.index]
        }).sort_values('Cluster')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.framework.clusters)))
        cluster_colors = {cid: colors[i] for i, cid in enumerate(sorted(self.framework.clusters.keys()))}
        bar_colors = [cluster_colors[asset_to_cluster[a]] for a in weights_df['Asset']]
        
        ax1.bar(range(len(weights_df)), weights_df['Weight'], color=bar_colors, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(weights_df)))
        ax1.set_xticklabels(weights_df['Asset'], rotation=45, ha='right')
        ax1.set_ylabel('Poids du Portefeuille', fontsize=12, fontweight='bold')
        ax1.set_title('Poids du Portefeuille par Actif\nGroupés par Cluster', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        for i, (asset, cluster) in enumerate(zip(weights_df['Asset'], weights_df['Cluster'])):
            ax1.text(i, weights_df.iloc[i]['Weight'] + 0.005, f'C{cluster}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        cluster_weights = weights_df.groupby('Cluster')['Weight'].sum()
        
        wedges, texts, autotexts = ax2.pie(
            cluster_weights.values,
            labels=[f'Cluster {cid}\n({len(self.framework.clusters[cid])} actifs)' 
                   for cid in cluster_weights.index],
            autopct='%1.1f%%',
            colors=[cluster_colors[cid] for cid in cluster_weights.index],
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        ax2.set_title('Allocation du Portefeuille par Cluster\nStratégie d\'Allocation Équilibrée', 
                     fontsize=14, fontweight='bold')
        
        legend_labels = []
        for cid in sorted(self.framework.clusters.keys()):
            assets = ', '.join(self.framework.clusters[cid])
            legend_labels.append(f"C{cid}: {assets}")
        
        ax2.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=9, title='Actifs par Cluster', title_fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Composition du portefeuille: {save_path}")
        self.figures['portfolio_composition'] = save_path
        return save_path
    
    def plot_performance(self, figsize=(14, 10)):
        """Générer l'analyse de performance."""
        save_path = os.path.join(self.viz_dir, 'performance_comparison.png')
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        clustered_returns = (self.framework.returns * self.framework.portfolio_weights).sum(axis=1)
        equal_weight = pd.Series(1/len(self.framework.tickers), index=self.framework.tickers)
        equal_returns = (self.framework.returns * equal_weight).sum(axis=1)
        
        clustered_cumulative = (1 + clustered_returns).cumprod()
        equal_cumulative = (1 + equal_returns).cumprod()
        
        ax1 = axes[0, 0]
        ax1.plot(clustered_cumulative.index, clustered_cumulative.values, 
                label='Portefeuille Clustérisé', linewidth=2, color='darkblue')
        ax1.plot(equal_cumulative.index, equal_cumulative.values,
                label='Portefeuille Équipondéré', linewidth=2, color='darkred', linestyle='--')
        ax1.set_title('Comparaison des Rendements Cumulés', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Rendement Cumulé', fontsize=10, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        ax2 = axes[0, 1]
        clustered_vol = clustered_returns.rolling(30).std() * np.sqrt(252)
        equal_vol = equal_returns.rolling(30).std() * np.sqrt(252)
        
        ax2.plot(clustered_vol.index, clustered_vol.values,
                label='Portefeuille Clustérisé', linewidth=2, color='darkblue')
        ax2.plot(equal_vol.index, equal_vol.values,
                label='Portefeuille Équipondéré', linewidth=2, color='darkred', linestyle='--')
        ax2.set_title('Volatilité Glissante (30 jours, Annualisée)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volatilité', fontsize=10, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        ax3 = axes[1, 0]
        clustered_dd = (clustered_cumulative / clustered_cumulative.cummax() - 1)
        equal_dd = (equal_cumulative / equal_cumulative.cummax() - 1)
        
        ax3.fill_between(clustered_dd.index, 0, clustered_dd.values,
                        label='Portefeuille Clustérisé', alpha=0.5, color='darkblue')
        ax3.fill_between(equal_dd.index, 0, equal_dd.values,
                        label='Portefeuille Équipondéré', alpha=0.5, color='darkred')
        ax3.set_title('Analyse des Drawdowns', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown', fontsize=10, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics = {
            'Métrique': [
                'Rendement Total',
                'Rendement Annualisé',
                'Volatilité Annualisée',
                'Ratio de Sharpe',
                'Drawdown Maximum',
                'Ratio de Calmar'
            ],
            'Clustérisé': [
                f"{(clustered_cumulative.iloc[-1] - 1):.2%}",
                f"{clustered_returns.mean() * 252:.2%}",
                f"{clustered_returns.std() * np.sqrt(252):.2%}",
                f"{(clustered_returns.mean() * 252) / (clustered_returns.std() * np.sqrt(252)):.2f}",
                f"{clustered_dd.min():.2%}",
                f"{(clustered_returns.mean() * 252) / abs(clustered_dd.min()):.2f}"
            ],
            'Équipondéré': [
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
        
        for i in range(len(metrics_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(metrics_df) + 1):
            for j in range(len(metrics_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax4.set_title('Résumé des Métriques de Performance', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Analyse de Performance du Portefeuille\nStratégie Basée sur les Clusters vs Équipondérée',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Analyse de performance: {save_path}")
        self.figures['performance'] = save_path
        return save_path
    
    def generate_all_visualizations(self):
        """Générer toutes les visualisations."""
        print("\n" + "="*70)
        print("GÉNÉRATION DES VISUALISATIONS")
        print("="*70 + "\n")
        
        self.plot_dendrogram()
        self.plot_correlation_heatmap()
        self.plot_cluster_heatmap()
        self.plot_portfolio_composition()
        self.plot_performance()
        
        print("\n" + "="*70)
        print(f"✓ Toutes les visualisations sauvegardées dans:")
        print(f"  {self.viz_dir}/")
        print("="*70)
        
        return self.figures


# Test du module
if __name__ == "__main__":
    from portfolio_clustering_framework import PortfolioClusteringFramework
    
    print("Test du module de visualisation \n")
    
    # Créer un framework simple
    framework = PortfolioClusteringFramework(preset='simple')
    framework.run_full_analysis()
    
    # Générer les visualisations
    visualizer = PortfolioVisualizer(framework)
    figures = visualizer.generate_all_visualizations()
    
    print(f"\n✓ Test réussi! {len(figures)} visualisations créées")