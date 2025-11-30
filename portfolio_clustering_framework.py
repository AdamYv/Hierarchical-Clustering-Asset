
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
Hierarchical Clustering Portfolio Framework
============================================
An educational open-source framework for asset classification and portfolio diversification
using hierarchical clustering techniques.

Author: Educational Framework
License: MIT
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
warnings.filterwarnings('ignore')

class PortfolioClusteringFramework:
    """
    Main framework class for hierarchical clustering-based portfolio optimization.
    
    This educational framework demonstrates:
    - Asset correlation analysis
    - Hierarchical clustering for asset classification
    - Diversification-optimized portfolio construction
    """
    
    def __init__(self, tickers=None, start_date=None, end_date=None):
        """
        Initialize the framework with asset universe and time period.
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols. If None, uses default educational portfolio.
        start_date : str, optional
            Start date for historical data (YYYY-MM-DD). Default: 3 years ago.
        end_date : str, optional
            End date for historical data (YYYY-MM-DD). Default: today.
        """
        # Default educational asset universe - diverse across asset classes
        self.default_tickers = [
            # US Large Cap Stocks
            'AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ',
            # International Stocks
            'EFA',  # MSCI EAFE ETF
            # Bonds
            'TLT',  # Long-term Treasury
            'AGG',  # Aggregate Bond
            # Commodities
            'GLD',  # Gold
            'USO',  # Oil
            # Real Estate
            'VNQ',  # REIT ETF
            # Emerging Markets
            'EEM',  # Emerging Markets ETF
        ]
        
        self.tickers = tickers if tickers is not None else self.default_tickers
        
        # Set date range - default to 3 years for educational purposes
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            start = datetime.now() - timedelta(days=3*365)
            self.start_date = start.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
        
        # Initialize data containers
        self.prices = None
        self.returns = None
        self.correlation_matrix = None
        self.distance_matrix = None
        self.linkage_matrix = None
        self.clusters = None
        self.portfolio_weights = None
        
        print("Portfolio Clustering Framework Initialized")
        print(f"Assets: {len(self.tickers)}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Tickers: {', '.join(self.tickers)}")
    
    def load_data(self, verbose=True):
        """
        Load historical price data from Yahoo Finance.
        
        Educational Note:
        - Uses Close prices (already adjusted for splits and dividends in yfinance)
        - Handles missing data through forward-fill then backward-fill
        - Removes assets with >30% missing data
        
        Returns:
        --------
        pd.DataFrame : Cleaned price data
        """
        if verbose:
            print("\n" + "="*60)
            print("STEP 1: DATA LOADING")
            print("="*60)
        
        # Download data from Yahoo Finance
        if verbose:
            print(f"\nDownloading data for {len(self.tickers)} assets...")
        
        try:
            # Download all data
            raw_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            # Extract Close prices (already adjusted in yfinance)
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Multiple tickers case - extract Close prices
                data = raw_data['Close']
            else:
                # Single ticker case
                data = raw_data[['Close']].copy()
                data.columns = self.tickers
            
            # Ensure we have a DataFrame
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        if verbose:
            print(f"Initial data shape: {data.shape}")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Data cleaning: handle missing values
        missing_pct = (data.isnull().sum() / len(data)) * 100
        
        if verbose and missing_pct.sum() > 0:
            print("\nMissing data analysis:")
            for ticker in data.columns:
                pct = missing_pct[ticker]
                if pct > 0:
                    print(f"  {ticker}: {pct:.2f}% missing")
        
        # Remove assets with >30% missing data
        threshold = 30
        valid_assets = missing_pct[missing_pct <= threshold].index.tolist()
        removed_assets = [t for t in data.columns if t not in valid_assets]
        
        if removed_assets and verbose:
            print(f"\nRemoving assets with >{threshold}% missing data: {removed_assets}")
        
        data = data[valid_assets]
        
        # Forward fill then backward fill remaining missing values
        data = data.ffill().bfill()
        
        # Update tickers list
        self.tickers = valid_assets
        self.prices = data
        
        if verbose:
            print(f"\nFinal data shape: {data.shape}")
            print(f"Assets retained: {len(self.tickers)}")
            print("Data cleaning complete ✓")
        
        return data
    
    def calculate_returns(self, verbose=True):
        """
        Calculate logarithmic returns from price data.
        
        Educational Note:
        - Log returns are time-additive: log(P_t/P_0) = log(P_t/P_1) + log(P_1/P_0)
        - More suitable for statistical analysis (closer to normal distribution)
        - Formula: r_t = ln(P_t / P_{t-1})
        
        Returns:
        --------
        pd.DataFrame : Log returns
        """
        if self.prices is None:
            print("Error: No price data loaded. Run load_data() first.")
            return None
        
        if verbose:
            print("\n" + "="*60)
            print("STEP 2: RETURN CALCULATION")
            print("="*60)
        
        # Calculate log returns
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        
        if verbose:
            print("\nLog returns calculated")
            print(f"Shape: {self.returns.shape}")
            print("\nReturn statistics (annualized):")
            print(f"{'Asset':<10} {'Mean':<10} {'Std Dev':<10} {'Sharpe':<10}")
            print("-" * 40)
            
            for ticker in self.returns.columns:
                mean_ret = self.returns[ticker].mean() * 252  # Annualize
                std_ret = self.returns[ticker].std() * np.sqrt(252)  # Annualize
                sharpe = mean_ret / std_ret if std_ret > 0 else 0
                print(f"{ticker:<10} {mean_ret:>9.2%} {std_ret:>9.2%} {sharpe:>9.2f}")
        
        return self.returns
    
    def calculate_correlation(self, method='pearson', verbose=True):
        """
        Calculate correlation matrix and convert to distance matrix.
        
        Educational Note:
        - Pearson correlation measures linear relationships
        - Distance = sqrt(2 * (1 - correlation))
        - This ensures: correlation=1 → distance=0, correlation=-1 → distance=2
        
        Parameters:
        -----------
        method : str
            Correlation method: 'pearson', 'spearman', or 'kendall'
        
        Returns:
        --------
        tuple : (correlation_matrix, distance_matrix)
        """
        if self.returns is None:
            print("Error: No returns calculated. Run calculate_returns() first.")
            return None, None
        
        if verbose:
            print("\n" + "="*60)
            print("STEP 3: CORRELATION ANALYSIS")
            print("="*60)
        
        # Calculate correlation matrix
        self.correlation_matrix = self.returns.corr(method=method)
        
        if verbose:
            print(f"\nCorrelation method: {method}")
            print(f"Matrix shape: {self.correlation_matrix.shape}")
            print("\nCorrelation statistics:")
            corr_values = self.correlation_matrix.values[np.triu_indices_from(
                self.correlation_matrix.values, k=1)]
            print(f"  Mean correlation: {corr_values.mean():.3f}")
            print(f"  Min correlation: {corr_values.min():.3f}")
            print(f"  Max correlation: {corr_values.max():.3f}")
        
        # Convert correlation to distance
        # Distance formula: d = sqrt(2 * (1 - correlation))
        # This ensures proper metric properties for hierarchical clustering
        self.distance_matrix = np.sqrt(2 * (1 - self.correlation_matrix))
        
        if verbose:
            print("\nDistance matrix calculated")
            print(f"  Distance range: [{self.distance_matrix.values.min():.3f}, "
                  f"{self.distance_matrix.values.max():.3f}]")
        
        return self.correlation_matrix, self.distance_matrix
    
    def perform_clustering(self, method='ward', max_clusters=None, verbose=True):
        """
        Perform hierarchical clustering on assets.
        
        Educational Note:
        - Ward's method minimizes within-cluster variance
        - Other methods: 'single', 'complete', 'average'
        - Optimal clusters determined by maximum distance gap in dendrogram
        
        Parameters:
        -----------
        method : str
            Linkage method for hierarchical clustering
        max_clusters : int, optional
            Maximum number of clusters. If None, determined automatically.
        
        Returns:
        --------
        dict : Cluster assignments for each asset
        """
        if self.distance_matrix is None:
            print("Error: No distance matrix. Run calculate_correlation() first.")
            return None
        
        if verbose:
            print("\n" + "="*60)
            print("STEP 4: HIERARCHICAL CLUSTERING")
            print("="*60)
        
        # Convert distance matrix to condensed form for scipy
        distance_condensed = squareform(self.distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        self.linkage_matrix = linkage(distance_condensed, method=method)
        
        if verbose:
            print(f"\nLinkage method: {method}")
            print(f"Linkage matrix shape: {self.linkage_matrix.shape}")
        
        # Determine optimal number of clusters
        if max_clusters is None:
            # Use elbow method: find largest gap in merge distances
            distances = self.linkage_matrix[:, 2]
            gaps = np.diff(distances)
            
            # Find optimal clusters (between 2 and n/2)
            n_assets = len(self.tickers)
            max_k = min(n_assets // 2, 6)  # Cap at 6 for interpretability
            
            optimal_k = np.argmax(gaps[-max_k:]) + 2
            n_clusters = optimal_k
            
            if verbose:
                print("\nAutomatic cluster determination:")
                print(f"  Optimal number of clusters: {n_clusters}")
        else:
            n_clusters = max_clusters
            if verbose:
                print(f"\nUsing specified number of clusters: {n_clusters}")
        
        # Assign assets to clusters
        cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        
        # Create cluster dictionary
        self.clusters = {}
        for i, ticker in enumerate(self.tickers):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(ticker)
        
        if verbose:
            print("\nCluster assignments:")
            for cluster_id in sorted(self.clusters.keys()):
                assets = self.clusters[cluster_id]
                print(f"  Cluster {cluster_id}: {', '.join(assets)} ({len(assets)} assets)")
        
        return self.clusters
    
    def optimize_portfolio(self, allocation_method='equal_cluster', verbose=True):
        """
        Calculate portfolio weights optimized for diversification.
        
        Educational Note:
        - Equal cluster allocation ensures diversification across asset classes
        - Within each cluster, equal weighting (naive diversification)
        - This approach maximizes cluster-level diversification
        
        Parameters:
        -----------
        allocation_method : str
            'equal_cluster': Equal weight per cluster, then equal within cluster
            'equal_asset': Simple equal weighting across all assets
        
        Returns:
        --------
        pd.Series : Portfolio weights for each asset
        """
        if self.clusters is None:
            print("Error: No clusters defined. Run perform_clustering() first.")
            return None
        
        if verbose:
            print("\n" + "="*60)
            print("STEP 5: PORTFOLIO OPTIMIZATION")
            print("="*60)
        
        weights = {}
        
        if allocation_method == 'equal_cluster':
            # Equal allocation to each cluster
            n_clusters = len(self.clusters)
            cluster_weight = 1.0 / n_clusters
            
            if verbose:
                print("\nAllocation method: Equal weight per cluster")
                print(f"Number of clusters: {n_clusters}")
                print(f"Weight per cluster: {cluster_weight:.2%}")
            
            # Within each cluster, equal weight to each asset
            for cluster_id, assets in self.clusters.items():
                n_assets_in_cluster = len(assets)
                asset_weight = cluster_weight / n_assets_in_cluster
                
                for asset in assets:
                    weights[asset] = asset_weight
        
        elif allocation_method == 'equal_asset':
            # Simple equal weighting
            n_assets = len(self.tickers)
            weight = 1.0 / n_assets
            
            if verbose:
                print("\nAllocation method: Equal weight per asset")
                print(f"Weight per asset: {weight:.2%}")
            
            for asset in self.tickers:
                weights[asset] = weight
        
        self.portfolio_weights = pd.Series(weights)
        
        if verbose:
            print("\nPortfolio weights:")
            print(f"{'Asset':<10} {'Weight':<10} {'Cluster':<10}")
            print("-" * 30)
            
            # Find cluster for each asset
            asset_to_cluster = {}
            for cluster_id, assets in self.clusters.items():
                for asset in assets:
                    asset_to_cluster[asset] = cluster_id
            
            for asset in self.portfolio_weights.index:
                weight = self.portfolio_weights[asset]
                cluster = asset_to_cluster.get(asset, 'N/A')
                print(f"{asset:<10} {weight:>9.2%} {cluster:>9}")
        
        # Calculate diversification metrics
        self._calculate_diversification_metrics(verbose=verbose)
        
        return self.portfolio_weights
    
    def _calculate_diversification_metrics(self, verbose=True):
        """
        Calculate portfolio diversification metrics.
        
        Educational Note:
        - Effective N: Number of "independent" assets
        - Diversification Ratio: Weighted avg volatility / Portfolio volatility
        - Higher values indicate better diversification
        """
        if verbose:
            print("\nDiversification metrics:")
        
        # Herfindahl index (concentration)
        herfindahl = (self.portfolio_weights ** 2).sum()
        effective_n = 1 / herfindahl
        
        if verbose:
            print(f"  Effective number of assets: {effective_n:.2f}")
            print(f"  Concentration (Herfindahl): {herfindahl:.4f}")
        
        # Portfolio variance - convert to numpy arrays for proper matrix multiplication
        weights_array = self.portfolio_weights.values.reshape(-1, 1)
        cov_matrix = self.returns.cov().values  # Convert to numpy array
        portfolio_variance = float((weights_array.T @ cov_matrix @ weights_array)[0, 0])
        portfolio_vol = np.sqrt(portfolio_variance * 252)  # Annualized
        
        # Weighted average volatility
        individual_vols = self.returns.std() * np.sqrt(252)
        weighted_avg_vol = (self.portfolio_weights * individual_vols).sum()
        
        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol
        
        if verbose:
            print(f"  Portfolio volatility (annual): {portfolio_vol:.2%}")
            print(f"  Weighted avg volatility: {weighted_avg_vol:.2%}")
            print(f"  Diversification ratio: {div_ratio:.2f}")
        
        return {
            'effective_n': effective_n,
            'herfindahl': herfindahl,
            'portfolio_vol': portfolio_vol,
            'diversification_ratio': div_ratio
        }
    
    def run_full_analysis(self):
        """
        Execute complete analysis pipeline.
        
        Returns:
        --------
        dict : Results dictionary with all outputs
        """
        print("\n" + "="*60)
        print("HIERARCHICAL CLUSTERING PORTFOLIO FRAMEWORK")
        print("Educational Open-Source Implementation")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Calculate returns
        self.calculate_returns()
        
        # Step 3: Correlation analysis
        self.calculate_correlation()
        
        # Step 4: Hierarchical clustering
        self.perform_clustering()
        
        # Step 5: Portfolio optimization
        self.optimize_portfolio()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return {
            'prices': self.prices,
            'returns': self.returns,
            'correlation': self.correlation_matrix,
            'clusters': self.clusters,
            'weights': self.portfolio_weights
        }


# Example usage
if __name__ == "__main__":
    # Initialize framework with default educational portfolio
    framework = PortfolioClusteringFramework()
    
    # Run complete analysis
    results = framework.run_full_analysis()
    
    print("\n✓ Framework execution completed successfully!")
    print("✓ Results available in 'results' dictionary")
    print("✓ Framework object stored in 'framework' variable")