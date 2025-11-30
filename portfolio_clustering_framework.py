
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: portfolio_clustering_framework.py
# execution: true

"""
Portfolio Clustering Framework 
==================================
Version améliorée avec:
- Organisation des fichiers dans un dossier unique
- Interface simplifiée pour listes d'actifs personnalisées
- Gestion automatique des dossiers de sortie
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

class PortfolioClusteringFramework:
    """
    Framework principal pour le clustering hiérarchique de portefeuille.
    
    Version 2.0 avec organisation simplifiée des fichiers.
    """
    
    # Listes d'actifs prédéfinies pour faciliter l'utilisation
    PRESET_PORTFOLIOS = {
        'default': {
            'name': 'Portfolio Diversifié Standard',
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'EFA', 'TLT', 'AGG', 'GLD', 'USO', 'VNQ', 'EEM'],
            'description': '12 actifs diversifiés: tech, finance, santé, international, obligations, matières premières'
        },
        'tech': {
            'name': 'Portfolio Technologie',
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX'],
            'description': 'Focus sur les grandes valeurs technologiques'
        },
        'conservative': {
            'name': 'Portfolio Conservateur',
            'tickers': ['AGG', 'TLT', 'BND', 'VNQ', 'GLD', 'JNJ', 'PG', 'KO'],
            'description': 'Obligations, défensives et valeurs refuges'
        },
        'global': {
            'name': 'Portfolio Global',
            'tickers': ['VTI', 'VEA', 'VWO', 'BND', 'BNDX', 'GLD', 'VNQ', 'VNQI'],
            'description': 'ETFs diversifiés mondialement'
        },
        'simple': {
            'name': 'Portfolio Simple',
            'tickers': ['SPY', 'AGG', 'GLD', 'VNQ'],
            'description': 'Portfolio minimaliste 4 actifs'
        }
    }
    
    def __init__(self, tickers=None, preset='default', start_date=None, end_date=None, output_dir=None):
        """
        Initialiser le framework avec une liste d'actifs.
        
        Parameters:
        -----------
        tickers : list, optional
            Liste personnalisée de tickers. Si None, utilise le preset.
        preset : str, optional
            Nom du portfolio prédéfini ('default', 'tech', 'conservative', 'global', 'simple')
        start_date : str, optional
            Date de début (YYYY-MM-DD). Par défaut: 3 ans avant aujourd'hui.
        end_date : str, optional
            Date de fin (YYYY-MM-DD). Par défaut: aujourd'hui.
        output_dir : str, optional
            Dossier de sortie. Par défaut: 'portfolio_analysis_YYYYMMDD_HHMMSS'
        
        Examples:
        ---------
        # Utiliser un preset
        framework = PortfolioClusteringFramework(preset='tech')
        
        # Liste personnalisée
        framework = PortfolioClusteringFramework(tickers=['AAPL', 'MSFT', 'GLD', 'TLT'])
        
        # Avec dates personnalisées
        framework = PortfolioClusteringFramework(
            tickers=['AAPL', 'MSFT', 'GLD'],
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        """
        # Déterminer la liste d'actifs
        if tickers is not None:
            self.tickers = tickers
            self.portfolio_name = 'Portfolio Personnalisé'
            self.portfolio_description = f'{len(tickers)} actifs sélectionnés'
        elif preset in self.PRESET_PORTFOLIOS:
            preset_info = self.PRESET_PORTFOLIOS[preset]
            self.tickers = preset_info['tickers']
            self.portfolio_name = preset_info['name']
            self.portfolio_description = preset_info['description']
        else:
            raise ValueError(f"Preset '{preset}' non reconnu. Choix: {list(self.PRESET_PORTFOLIOS.keys())}")
        
        # Dates
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            start = datetime.now() - timedelta(days=3*365)
            self.start_date = start.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
        
        # Créer le dossier de sortie
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f'portfolio_analysis_{timestamp}'
        else:
            self.output_dir = output_dir
        
        # Créer les sous-dossiers
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'reports'), exist_ok=True)
        
        # Initialiser les conteneurs de données
        self.prices = None
        self.returns = None
        self.correlation_matrix = None
        self.distance_matrix = None
        self.linkage_matrix = None
        self.clusters = None
        self.portfolio_weights = None
        
        print("="*70)
        print("PORTFOLIO CLUSTERING FRAMEWORK ")
        print("="*70)
        print(f"\n📊 Portfolio: {self.portfolio_name}")
        print(f"📝 Description: {self.portfolio_description}")
        print(f"📈 Actifs: {len(self.tickers)}")
        print(f"📅 Période: {self.start_date} à {self.end_date}")
        print(f"📁 Dossier de sortie: {self.output_dir}/")
        print(f"\nTickers: {', '.join(self.tickers)}")
    
    @classmethod
    def list_presets(cls):
        """
        Afficher tous les portfolios prédéfinis disponibles.
        """
        print("\n" + "="*70)
        print("PORTFOLIOS PRÉDÉFINIS DISPONIBLES")
        print("="*70)
        
        for key, info in cls.PRESET_PORTFOLIOS.items():
            print(f"\n🔹 '{key}' - {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Actifs ({len(info['tickers'])}): {', '.join(info['tickers'])}")
    
    def load_data(self, verbose=True):
        """Charger les données depuis Yahoo Finance."""
        if verbose:
            print("\n" + "="*70)
            print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
            print("="*70)
        
        if verbose:
            print(f"\n📥 Téléchargement des données pour {len(self.tickers)} actifs...")
        
        try:
            raw_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            if isinstance(raw_data.columns, pd.MultiIndex):
                data = raw_data['Close']
            else:
                data = raw_data[['Close']].copy()
                data.columns = self.tickers
            
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
            return None
        
        if verbose:
            print(f"✓ Données téléchargées: {data.shape}")
            print(f"✓ Période: {data.index[0]} à {data.index[-1]}")
        
        # Nettoyage
        missing_pct = (data.isnull().sum() / len(data)) * 100
        
        if verbose and missing_pct.sum() > 0:
            print("\n⚠️  Données manquantes détectées:")
            for ticker in data.columns:
                pct = missing_pct[ticker]
                if pct > 0:
                    print(f"   {ticker}: {pct:.2f}%")
        
        threshold = 30
        valid_assets = missing_pct[missing_pct <= threshold].index.tolist()
        removed_assets = [t for t in data.columns if t not in valid_assets]
        
        if removed_assets and verbose:
            print(f"\n❌ Actifs supprimés (>{threshold}% manquant): {removed_assets}")
        
        data = data[valid_assets]
        data = data.ffill().bfill()
        
        self.tickers = valid_assets
        self.prices = data
        
        if verbose:
            print(f"\n✓ Données nettoyées: {data.shape}")
            print(f"✓ Actifs retenus: {len(self.tickers)}")
        
        return data
    
    def calculate_returns(self, verbose=True):
        """Calculer les rendements logarithmiques."""
        if self.prices is None:
            print("❌ Erreur: Charger les données d'abord avec load_data()")
            return None
        
        if verbose:
            print("\n" + "="*70)
            print("ÉTAPE 2: CALCUL DES RENDEMENTS")
            print("="*70)
        
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        
        if verbose:
            print(f"\n✓ Rendements logarithmiques calculés")
            print(f"✓ Forme: {self.returns.shape}")
            print("\n📊 Statistiques (annualisées):")
            print(f"{'Actif':<10} {'Rendement':<12} {'Volatilité':<12} {'Sharpe':<10}")
            print("-" * 44)
            
            for ticker in self.returns.columns:
                mean_ret = self.returns[ticker].mean() * 252
                std_ret = self.returns[ticker].std() * np.sqrt(252)
                sharpe = mean_ret / std_ret if std_ret > 0 else 0
                print(f"{ticker:<10} {mean_ret:>11.2%} {std_ret:>11.2%} {sharpe:>9.2f}")
        
        return self.returns
    
    def calculate_correlation(self, method='pearson', verbose=True):
        """Calculer la matrice de corrélation et de distance."""
        if self.returns is None:
            print("❌ Erreur: Calculer les rendements d'abord avec calculate_returns()")
            return None, None
        
        if verbose:
            print("\n" + "="*70)
            print("ÉTAPE 3: ANALYSE DE CORRÉLATION")
            print("="*70)
        
        self.correlation_matrix = self.returns.corr(method=method)
        
        if verbose:
            print(f"\n✓ Matrice de corrélation calculée ({method})")
            print(f"✓ Dimension: {self.correlation_matrix.shape}")
            
            corr_values = self.correlation_matrix.values[np.triu_indices_from(
                self.correlation_matrix.values, k=1)]
            print(f"\n📊 Statistiques de corrélation:")
            print(f"   Moyenne: {corr_values.mean():.3f}")
            print(f"   Min: {corr_values.min():.3f}")
            print(f"   Max: {corr_values.max():.3f}")
        
        self.distance_matrix = np.sqrt(2 * (1 - self.correlation_matrix))
        
        if verbose:
            print(f"\n✓ Matrice de distance calculée")
            print(f"   Formule: d = √(2(1-ρ))")
        
        return self.correlation_matrix, self.distance_matrix
    
    def perform_clustering(self, method='ward', max_clusters=None, verbose=True):
        """Effectuer le clustering hiérarchique."""
        if self.distance_matrix is None:
            print("❌ Erreur: Calculer la corrélation d'abord avec calculate_correlation()")
            return None
        
        if verbose:
            print("\n" + "="*70)
            print("ÉTAPE 4: CLUSTERING HIÉRARCHIQUE")
            print("="*70)
        
        distance_condensed = squareform(self.distance_matrix, checks=False)
        self.linkage_matrix = linkage(distance_condensed, method=method)
        
        if verbose:
            print(f"\n✓ Clustering effectué (méthode: {method})")
        
        if max_clusters is None:
            distances = self.linkage_matrix[:, 2]
            gaps = np.diff(distances)
            n_assets = len(self.tickers)
            max_k = min(n_assets // 2, 6)
            optimal_k = np.argmax(gaps[-max_k:]) + 2
            n_clusters = optimal_k
            
            if verbose:
                print(f"✓ Nombre optimal de clusters: {n_clusters} (déterminé automatiquement)")
        else:
            n_clusters = max_clusters
            if verbose:
                print(f"✓ Nombre de clusters: {n_clusters} (spécifié)")
        
        cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        
        self.clusters = {}
        for i, ticker in enumerate(self.tickers):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(ticker)
        
        if verbose:
            print(f"\n📊 Attribution des actifs aux clusters:")
            for cluster_id in sorted(self.clusters.keys()):
                assets = self.clusters[cluster_id]
                print(f"   Cluster {cluster_id}: {', '.join(assets)} ({len(assets)} actifs)")
        
        return self.clusters
    
    def optimize_portfolio(self, allocation_method='equal_cluster', verbose=True):
        """Calculer les poids du portefeuille."""
        if self.clusters is None:
            print("❌ Erreur: Effectuer le clustering d'abord avec perform_clustering()")
            return None
        
        if verbose:
            print("\n" + "="*70)
            print("ÉTAPE 5: OPTIMISATION DU PORTEFEUILLE")
            print("="*70)
        
        weights = {}
        
        if allocation_method == 'equal_cluster':
            n_clusters = len(self.clusters)
            cluster_weight = 1.0 / n_clusters
            
            if verbose:
                print(f"\n✓ Méthode: Allocation équilibrée par cluster")
                print(f"✓ Poids par cluster: {cluster_weight:.2%}")
            
            for cluster_id, assets in self.clusters.items():
                n_assets_in_cluster = len(assets)
                asset_weight = cluster_weight / n_assets_in_cluster
                for asset in assets:
                    weights[asset] = asset_weight
        
        elif allocation_method == 'equal_asset':
            n_assets = len(self.tickers)
            weight = 1.0 / n_assets
            
            if verbose:
                print(f"\n✓ Méthode: Allocation équipondérée")
                print(f"✓ Poids par actif: {weight:.2%}")
            
            for asset in self.tickers:
                weights[asset] = weight
        
        self.portfolio_weights = pd.Series(weights)
        
        if verbose:
            print(f"\n📊 Poids du portefeuille:")
            print(f"{'Actif':<10} {'Poids':<10} {'Cluster':<10}")
            print("-" * 30)
            
            asset_to_cluster = {}
            for cluster_id, assets in self.clusters.items():
                for asset in assets:
                    asset_to_cluster[asset] = cluster_id
            
            for asset in self.portfolio_weights.index:
                weight = self.portfolio_weights[asset]
                cluster = asset_to_cluster.get(asset, 'N/A')
                print(f"{asset:<10} {weight:>9.2%} {cluster:>9}")
        
        # Métriques de diversification
        self._calculate_diversification_metrics(verbose=verbose)
        
        return self.portfolio_weights
    
    def _calculate_diversification_metrics(self, verbose=True):
        """Calculer les métriques de diversification."""
        if verbose:
            print(f"\n📊 Métriques de diversification:")
        
        herfindahl = (self.portfolio_weights ** 2).sum()
        effective_n = 1 / herfindahl
        
        if verbose:
            print(f"   Nombre effectif d'actifs: {effective_n:.2f}")
            print(f"   Concentration (Herfindahl): {herfindahl:.4f}")
        
        weights_array = self.portfolio_weights.values.reshape(-1, 1)
        cov_matrix = self.returns.cov().values
        portfolio_variance = float((weights_array.T @ cov_matrix @ weights_array)[0, 0])
        portfolio_vol = np.sqrt(portfolio_variance * 252)
        
        individual_vols = self.returns.std() * np.sqrt(252)
        weighted_avg_vol = (self.portfolio_weights * individual_vols).sum()
        
        div_ratio = weighted_avg_vol / portfolio_vol
        
        if verbose:
            print(f"   Volatilité du portefeuille: {portfolio_vol:.2%}")
            print(f"   Volatilité moyenne pondérée: {weighted_avg_vol:.2%}")
            print(f"   Ratio de diversification: {div_ratio:.2f}")
        
        return {
            'effective_n': effective_n,
            'herfindahl': herfindahl,
            'portfolio_vol': portfolio_vol,
            'diversification_ratio': div_ratio
        }
    
    def run_full_analysis(self):
        """Exécuter l'analyse complète."""
        print("\n" + "="*70)
        print("ANALYSE COMPLÈTE DU PORTEFEUILLE")
        print("="*70)
        
        self.load_data()
        self.calculate_returns()
        self.calculate_correlation()
        self.perform_clustering()
        self.optimize_portfolio()
        
        print("\n" + "="*70)
        print("✓ ANALYSE TERMINÉE")
        print("="*70)
        
        return {
            'prices': self.prices,
            'returns': self.returns,
            'correlation': self.correlation_matrix,
            'clusters': self.clusters,
            'weights': self.portfolio_weights
        }


# Démonstration
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DÉMONSTRATION DU FRAMEWORK ")
    print("="*70)
    
    # Afficher les presets disponibles
    PortfolioClusteringFramework.list_presets()
    
    print("\n" + "="*70)
    print("EXEMPLE 1: Utilisation d'un preset")
    print("="*70)
    
    # Utiliser le preset 'simple'
    framework1 = PortfolioClusteringFramework(preset='simple')
    results1 = framework1.run_full_analysis()
    
    print("\n" + "="*70)
    print("EXEMPLE 2: Liste personnalisée")
    print("="*70)
    
    # Liste personnalisée
    my_tickers = ['AAPL', 'MSFT', 'GLD', 'TLT', 'VNQ']
    framework2 = PortfolioClusteringFramework(tickers=my_tickers)
    results2 = framework2.run_full_analysis()
    
    print("\n✓ Framework testé avec succès!")
    print(f"✓ Dossiers créés:")
    print(f"   - {framework1.output_dir}/")
    print(f"   - {framework2.output_dir}/")