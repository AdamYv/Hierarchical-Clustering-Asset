
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: report_generator.py
# execution: true

"""
Report Generator for Portfolio Clustering Framework
====================================================
Generates comprehensive PDF reports and CSV exports.
"""

import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os

class PortfolioReportGenerator:
    """
    Generate comprehensive PDF reports and CSV exports.
    
    Outputs:
    - PDF report with analysis and visualizations
    - CSV files for prices, returns, correlations, and weights
    """
    
    def __init__(self, framework, visualizer=None):
        """
        Initialize report generator.
        
        Parameters:
        -----------
        framework : PortfolioClusteringFramework
            Completed framework instance
        visualizer : PortfolioVisualizer, optional
            Visualizer instance with generated figures
        """
        self.framework = framework
        self.visualizer = visualizer
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2e5c8a'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Section style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#1f4788'),
            borderPadding=5,
            backColor=colors.HexColor('#e8f4f8')
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        ))
    
    def export_csv_files(self, output_dir='./'):
        """
        Export all data to CSV files.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save CSV files
        
        Returns:
        --------
        dict : Paths to saved CSV files
        """
        print("\n" + "="*60)
        print("EXPORTING CSV FILES")
        print("="*60 + "\n")
        
        files = {}
        
        # Export prices
        prices_path = os.path.join(output_dir, 'asset_prices.csv')
        self.framework.prices.to_csv(prices_path)
        files['prices'] = prices_path
        print(f"✓ Prices exported: {prices_path}")
        
        # Export returns
        returns_path = os.path.join(output_dir, 'asset_returns.csv')
        self.framework.returns.to_csv(returns_path)
        files['returns'] = returns_path
        print(f"✓ Returns exported: {returns_path}")
        
        # Export correlation matrix
        corr_path = os.path.join(output_dir, 'correlation_matrix.csv')
        self.framework.correlation_matrix.to_csv(corr_path)
        files['correlation'] = corr_path
        print(f"✓ Correlation matrix exported: {corr_path}")
        
        # Export portfolio weights
        weights_path = os.path.join(output_dir, 'portfolio_weights.csv')
        weights_df = pd.DataFrame({
            'Asset': self.framework.portfolio_weights.index,
            'Weight': self.framework.portfolio_weights.values
        })
        
        # Add cluster information
        asset_to_cluster = {}
        for cluster_id, assets in self.framework.clusters.items():
            for asset in assets:
                asset_to_cluster[asset] = cluster_id
        weights_df['Cluster'] = weights_df['Asset'].map(asset_to_cluster)
        
        weights_df.to_csv(weights_path, index=False)
        files['weights'] = weights_path
        print(f"✓ Portfolio weights exported: {weights_path}")
        
        # Export cluster assignments
        clusters_path = os.path.join(output_dir, 'cluster_assignments.csv')
        cluster_data = []
        for cluster_id, assets in self.framework.clusters.items():
            for asset in assets:
                cluster_data.append({'Asset': asset, 'Cluster': cluster_id})
        pd.DataFrame(cluster_data).to_csv(clusters_path, index=False)
        files['clusters'] = clusters_path
        print(f"✓ Cluster assignments exported: {clusters_path}")
        
        # Export summary statistics
        stats_path = os.path.join(output_dir, 'summary_statistics.csv')
        stats_data = []
        for ticker in self.framework.returns.columns:
            mean_ret = self.framework.returns[ticker].mean() * 252
            std_ret = self.framework.returns[ticker].std() * np.sqrt(252)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            
            stats_data.append({
                'Asset': ticker,
                'Cluster': asset_to_cluster[ticker],
                'Annual_Return': mean_ret,
                'Annual_Volatility': std_ret,
                'Sharpe_Ratio': sharpe,
                'Portfolio_Weight': self.framework.portfolio_weights[ticker]
            })
        pd.DataFrame(stats_data).to_csv(stats_path, index=False)
        files['statistics'] = stats_path
        print(f"✓ Summary statistics exported: {stats_path}")
        
        print("\n" + "="*60)
        print("CSV EXPORT COMPLETE")
        print("="*60)
        
        return files
    
    def generate_pdf_report(self, output_path='portfolio_clustering_report.pdf'):
        """
        Generate comprehensive PDF report.
        
        Parameters:
        -----------
        output_path : str
            Path for output PDF file
        
        Returns:
        --------
        str : Path to generated PDF
        """
        print("\n" + "="*60)
        print("GENERATING PDF REPORT")
        print("="*60 + "\n")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for PDF elements
        story = []
        
        # Title page
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(
            "Portfolio Clustering Framework",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(
            "Hierarchical Clustering for Asset Classification<br/>and Portfolio Diversification",
            self.styles['CustomSubtitle']
        ))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}<br/>"
            f"<b>Analysis Period:</b> {self.framework.start_date} to {self.framework.end_date}<br/>"
            f"<b>Number of Assets:</b> {len(self.framework.tickers)}<br/>"
            f"<b>Number of Clusters:</b> {len(self.framework.clusters)}",
            self.styles['CustomBody']
        ))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            "<i>Educational Open-Source Framework</i><br/>"
            "<i>Free Data Sources: Yahoo Finance</i>",
            self.styles['CustomBody']
        ))
        
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        summary_text = f"""
        This report presents a comprehensive analysis of portfolio diversification using hierarchical 
        clustering techniques. The framework analyzed {len(self.framework.tickers)} assets over a 
        {(pd.to_datetime(self.framework.end_date) - pd.to_datetime(self.framework.start_date)).days // 365}-year 
        period, identifying {len(self.framework.clusters)} distinct asset clusters based on correlation patterns.
        <br/><br/>
        <b>Key Findings:</b><br/>
        • The clustering algorithm successfully identified natural groupings in the asset universe<br/>
        • Portfolio diversification ratio of {self._calculate_div_ratio():.2f} indicates strong diversification benefits<br/>
        • Equal cluster weighting strategy ensures balanced exposure across asset classes<br/>
        • Effective number of independent assets: {self._calculate_effective_n():.2f}<br/>
        <br/>
        <b>Methodology:</b><br/>
        • Correlation-based distance metric: d = √(2(1-ρ))<br/>
        • Ward's hierarchical clustering for minimum variance<br/>
        • Automatic cluster determination via elbow method<br/>
        • Equal allocation across clusters for maximum diversification
        """
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        
        story.append(PageBreak())
        
        # Asset Universe
        story.append(Paragraph("Asset Universe", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # Create asset table
        asset_data = [['Asset', 'Cluster', 'Annual Return', 'Annual Vol', 'Sharpe', 'Weight']]
        asset_to_cluster = {}
        for cluster_id, assets in self.framework.clusters.items():
            for asset in assets:
                asset_to_cluster[asset] = cluster_id
        
        for ticker in self.framework.tickers:
            mean_ret = self.framework.returns[ticker].mean() * 252
            std_ret = self.framework.returns[ticker].std() * np.sqrt(252)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            weight = self.framework.portfolio_weights[ticker]
            cluster = asset_to_cluster[ticker]
            
            asset_data.append([
                ticker,
                f"C{cluster}",
                f"{mean_ret:.2%}",
                f"{std_ret:.2%}",
                f"{sharpe:.2f}",
                f"{weight:.2%}"
            ])
        
        asset_table = Table(asset_data, colWidths=[1*inch, 0.8*inch, 1.2*inch, 1.2*inch, 0.8*inch, 1*inch])
        asset_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(asset_table)
        
        story.append(PageBreak())
        
        # Cluster Analysis
        story.append(Paragraph("Cluster Analysis", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        cluster_text = f"""
        The hierarchical clustering algorithm identified {len(self.framework.clusters)} distinct clusters 
        based on correlation patterns in asset returns. Each cluster represents a group of assets with 
        similar behavior patterns, providing natural diversification boundaries.
        <br/><br/>
        <b>Cluster Composition:</b>
        """
        story.append(Paragraph(cluster_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.1*inch))
        
        for cluster_id in sorted(self.framework.clusters.keys()):
            assets = self.framework.clusters[cluster_id]
            cluster_weight = sum(self.framework.portfolio_weights[a] for a in assets)
            
            cluster_desc = f"""
            <b>Cluster {cluster_id}</b> ({len(assets)} assets, {cluster_weight:.1%} portfolio weight):<br/>
            Assets: {', '.join(assets)}<br/>
            """
            story.append(Paragraph(cluster_desc, self.styles['CustomBody']))
            story.append(Spacer(1, 0.1*inch))
        
        story.append(PageBreak())
        
        # Add visualizations if available
        if self.visualizer is not None:
            story.append(Paragraph("Visualizations", self.styles['SectionHeader']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add dendrogram
            if os.path.exists('dendrogram.png'):
                story.append(Paragraph("<b>Hierarchical Clustering Dendrogram</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
                img = Image('dendrogram.png', width=6.5*inch, height=3.8*inch)
                story.append(img)
                story.append(PageBreak())
            
            # Add correlation heatmap
            if os.path.exists('correlation_heatmap.png'):
                story.append(Paragraph("<b>Correlation Matrix</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
                img = Image('correlation_heatmap.png', width=5.5*inch, height=4.5*inch)
                story.append(img)
                story.append(PageBreak())
            
            # Add portfolio composition
            if os.path.exists('portfolio_composition.png'):
                story.append(Paragraph("<b>Portfolio Composition</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
                img = Image('portfolio_composition.png', width=6.5*inch, height=3.8*inch)
                story.append(img)
                story.append(PageBreak())
            
            # Add performance comparison
            if os.path.exists('performance_comparison.png'):
                story.append(Paragraph("<b>Performance Analysis</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
                img = Image('performance_comparison.png', width=6.5*inch, height=4.8*inch)
                story.append(img)
                story.append(PageBreak())
        
        # Methodology
        story.append(Paragraph("Methodology", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        methodology_text = """
        <b>1. Data Collection:</b><br/>
        Historical price data retrieved from Yahoo Finance, a free and reliable data source. 
        Adjusted close prices account for stock splits and dividends.<br/><br/>
        
        <b>2. Return Calculation:</b><br/>
        Logarithmic returns computed for time-additivity and improved statistical properties: 
        r_t = ln(P_t / P_{t-1})<br/><br/>
        
        <b>3. Correlation Analysis:</b><br/>
        Pearson correlation matrix calculated to measure linear relationships between asset returns.<br/><br/>
        
        <b>4. Distance Metric:</b><br/>
        Correlation converted to distance using: d = √(2(1-ρ)), ensuring proper metric properties 
        for hierarchical clustering.<br/><br/>
        
        <b>5. Hierarchical Clustering:</b><br/>
        Ward's method applied to minimize within-cluster variance. Optimal cluster count determined 
        via elbow method analyzing merge distances.<br/><br/>
        
        <b>6. Portfolio Optimization:</b><br/>
        Equal allocation across clusters ensures diversification. Within each cluster, assets 
        receive equal weights, implementing a naive diversification strategy that maximizes 
        cluster-level diversification benefits.
        """
        story.append(Paragraph(methodology_text, self.styles['CustomBody']))
        
        story.append(PageBreak())
        
        # Conclusion
        story.append(Paragraph("Conclusion and Educational Notes", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        conclusion_text = """
        This framework demonstrates the practical application of hierarchical clustering for portfolio 
        construction and asset classification. The methodology provides several educational insights:
        <br/><br/>
        <b>Key Learnings:</b><br/>
        • Correlation-based clustering reveals natural asset class groupings<br/>
        • Diversification benefits arise from low inter-cluster correlations<br/>
        • Equal cluster weighting provides systematic diversification<br/>
        • Hierarchical methods offer interpretable, transparent classification<br/>
        <br/>
        <b>Practical Applications:</b><br/>
        • Asset allocation across diverse investment universes<br/>
        • Risk management through systematic diversification<br/>
        • Portfolio rebalancing based on changing correlations<br/>
        • Educational tool for understanding portfolio theory<br/>
        <br/>
        <b>Limitations and Considerations:</b><br/>
        • Historical correlations may not predict future relationships<br/>
        • Equal weighting ignores expected returns and risk preferences<br/>
        • Clustering stability should be monitored over time<br/>
        • Transaction costs and constraints not considered<br/>
        <br/>
        <b>Open Source and Educational Use:</b><br/>
        This framework is designed for educational purposes using only free, publicly available 
        data sources. The code is transparent, well-documented, and extensible for further research 
        and learning.
        """
        story.append(Paragraph(conclusion_text, self.styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        
        print(f"✓ PDF report generated: {output_path}")
        print("\n" + "="*60)
        print("PDF GENERATION COMPLETE")
        print("="*60)
        
        return output_path
    
    def _calculate_div_ratio(self):
        """Calculate diversification ratio."""
        weights_array = self.framework.portfolio_weights.values.reshape(-1, 1)
        cov_matrix = self.framework.returns.cov().values
        portfolio_variance = float((weights_array.T @ cov_matrix @ weights_array)[0, 0])
        portfolio_vol = np.sqrt(portfolio_variance * 252)
        
        individual_vols = self.framework.returns.std() * np.sqrt(252)
        weighted_avg_vol = (self.framework.portfolio_weights * individual_vols).sum()
        
        return weighted_avg_vol / portfolio_vol
    
    def _calculate_effective_n(self):
        """Calculate effective number of assets."""
        herfindahl = (self.framework.portfolio_weights ** 2).sum()
        return 1 / herfindahl


# Example usage
if __name__ == "__main__":
    from portfolio_clustering_framework import PortfolioClusteringFramework
    from visualization_module import PortfolioVisualizer
    
    # Run complete analysis
    print("Running complete portfolio analysis pipeline...")
    framework = PortfolioClusteringFramework()
    results = framework.run_full_analysis()
    
    # Generate visualizations
    visualizer = PortfolioVisualizer(framework)
    figures = visualizer.generate_all_visualizations()
    
    # Generate reports
    report_gen = PortfolioReportGenerator(framework, visualizer)
    
    # Export CSV files
    csv_files = report_gen.export_csv_files()
    
    # Generate PDF report
    pdf_path = report_gen.generate_pdf_report()
    
    print("\n" + "="*60)
    print("COMPLETE FRAMEWORK EXECUTION FINISHED")
    print("="*60)
    print("\nGenerated files:")
    print("  • PDF Report: portfolio_clustering_report.pdf")
    print("  • Visualizations: 5 PNG files")
    print("  • Data Exports: 6 CSV files")
    print("\n✓ All outputs ready for educational use!")