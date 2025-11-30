
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
Générateur de Rapports 
============================
Génère les rapports PDF et exports CSV dans le dossier de sortie.
"""

import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import os

class PortfolioReportGenerator:
    """
    Générateur de rapports pour le framework de clustering.
    Tous les fichiers sont sauvegardés dans le dossier de sortie.
    """
    
    def __init__(self, framework, visualizer=None):
        """
        Initialiser le générateur de rapports.
        
        Parameters:
        -----------
        framework : PortfolioClusteringFramework
            Instance du framework complétée
        visualizer : PortfolioVisualizer, optional
            Instance du visualiseur avec figures générées
        """
        self.framework = framework
        self.visualizer = visualizer
        self.data_dir = os.path.join(framework.output_dir, 'data')
        self.report_dir = os.path.join(framework.output_dir, 'reports')
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Configurer les styles personnalisés."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2e5c8a'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
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
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        ))
    
    def export_csv_files(self):
        """Exporter toutes les données en CSV."""
        print("\n" + "="*70)
        print("EXPORT DES FICHIERS CSV")
        print("="*70 + "\n")
        
        files = {}
        
        # Prix
        prices_path = os.path.join(self.data_dir, 'asset_prices.csv')
        self.framework.prices.to_csv(prices_path)
        files['prices'] = prices_path
        print(f"✓ Prix: {prices_path}")
        
        # Rendements
        returns_path = os.path.join(self.data_dir, 'asset_returns.csv')
        self.framework.returns.to_csv(returns_path)
        files['returns'] = returns_path
        print(f"✓ Rendements: {returns_path}")
        
        # Corrélation
        corr_path = os.path.join(self.data_dir, 'correlation_matrix.csv')
        self.framework.correlation_matrix.to_csv(corr_path)
        files['correlation'] = corr_path
        print(f"✓ Matrice de corrélation: {corr_path}")
        
        # Poids du portefeuille
        weights_path = os.path.join(self.data_dir, 'portfolio_weights.csv')
        asset_to_cluster = {}
        for cluster_id, assets in self.framework.clusters.items():
            for asset in assets:
                asset_to_cluster[asset] = cluster_id
        
        weights_df = pd.DataFrame({
            'Asset': self.framework.portfolio_weights.index,
            'Weight': self.framework.portfolio_weights.values,
            'Cluster': [asset_to_cluster[a] for a in self.framework.portfolio_weights.index]
        })
        weights_df.to_csv(weights_path, index=False)
        files['weights'] = weights_path
        print(f"✓ Poids du portefeuille: {weights_path}")
        
        # Clusters
        clusters_path = os.path.join(self.data_dir, 'cluster_assignments.csv')
        cluster_data = []
        for cluster_id, assets in self.framework.clusters.items():
            for asset in assets:
                cluster_data.append({'Asset': asset, 'Cluster': cluster_id})
        pd.DataFrame(cluster_data).to_csv(clusters_path, index=False)
        files['clusters'] = clusters_path
        print(f"✓ Assignations de clusters: {clusters_path}")
        
        # Statistiques
        stats_path = os.path.join(self.data_dir, 'summary_statistics.csv')
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
        print(f"✓ Statistiques récapitulatives: {stats_path}")
        
        print("\n" + "="*70)
        print(f"✓ Tous les CSV sauvegardés dans:")
        print(f"  {self.data_dir}/")
        print("="*70)
        
        return files
    
    def generate_pdf_report(self):
        """Générer le rapport PDF complet."""
        output_path = os.path.join(self.report_dir, 'portfolio_clustering_report.pdf')
        
        print("\n" + "="*70)
        print("GÉNÉRATION DU RAPPORT PDF")
        print("="*70 + "\n")
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Page de titre
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(
            "Rapport d'Analyse de Portefeuille",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(
            f"{self.framework.portfolio_name}<br/>Clustering Hiérarchique et Optimisation",
            self.styles['CustomSubtitle']
        ))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            f"<b>Date du rapport:</b> {datetime.now().strftime('%d %B %Y à %H:%M')}<br/>"
            f"<b>Période d'analyse:</b> {self.framework.start_date} au {self.framework.end_date}<br/>"
            f"<b>Nombre d'actifs:</b> {len(self.framework.tickers)}<br/>"
            f"<b>Nombre de clusters:</b> {len(self.framework.clusters)}<br/>"
            f"<b>Description:</b> {self.framework.portfolio_description}",
            self.styles['CustomBody']
        ))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            "<i>Framework Open-Source Éducatif</i><br/>"
            "<i>Source de données: Yahoo Finance (gratuit)</i>",
            self.styles['CustomBody']
        ))
        
        story.append(PageBreak())
        
        # Résumé exécutif
        story.append(Paragraph("Résumé Exécutif", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        summary_text = f"""
        Ce rapport présente une analyse complète de diversification de portefeuille utilisant 
        des techniques de clustering hiérarchique. Le framework a analysé {len(self.framework.tickers)} 
        actifs sur une période de {(pd.to_datetime(self.framework.end_date) - pd.to_datetime(self.framework.start_date)).days // 365} 
        ans, identifiant {len(self.framework.clusters)} clusters distincts basés sur les patterns de corrélation.
        <br/><br/>
        <b>Résultats Clés:</b><br/>
        • Clusters identifiés avec succès par l'algorithme de clustering hiérarchique<br/>
        • Ratio de diversification de {self._calculate_div_ratio():.2f} indiquant de forts bénéfices<br/>
        • Stratégie d'allocation équilibrée par cluster pour une exposition balancée<br/>
        • Nombre effectif d'actifs indépendants: {self._calculate_effective_n():.2f}<br/>
        <br/>
        <b>Méthodologie:</b><br/>
        • Métrique de distance basée sur la corrélation: d = √(2(1-ρ))<br/>
        • Clustering hiérarchique de Ward pour minimiser la variance<br/>
        • Détermination automatique du nombre de clusters (méthode du coude)<br/>
        • Allocation équilibrée entre clusters pour maximiser la diversification
        """
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        
        story.append(PageBreak())
        
        # Univers d'actifs
        story.append(Paragraph("Univers d'Actifs", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        asset_data = [['Actif', 'Cluster', 'Rend. Annuel', 'Vol. Annuelle', 'Sharpe', 'Poids']]
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
        
        # Analyse des clusters
        story.append(Paragraph("Analyse des Clusters", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        cluster_text = f"""
        L'algorithme de clustering hiérarchique a identifié {len(self.framework.clusters)} clusters 
        distincts basés sur les patterns de corrélation des rendements. Chaque cluster représente 
        un groupe d'actifs avec des comportements similaires, fournissant des frontières naturelles 
        de diversification.
        <br/><br/>
        <b>Composition des Clusters:</b>
        """
        story.append(Paragraph(cluster_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.1*inch))
        
        for cluster_id in sorted(self.framework.clusters.keys()):
            assets = self.framework.clusters[cluster_id]
            cluster_weight = sum(self.framework.portfolio_weights[a] for a in assets)
            
            cluster_desc = f"""
            <b>Cluster {cluster_id}</b> ({len(assets)} actifs, {cluster_weight:.1%} du portefeuille):<br/>
            Actifs: {', '.join(assets)}<br/>
            """
            story.append(Paragraph(cluster_desc, self.styles['CustomBody']))
            story.append(Spacer(1, 0.1*inch))
        
        story.append(PageBreak())
        
        # Visualisations
        if self.visualizer is not None:
            story.append(Paragraph("Visualisations", self.styles['SectionHeader']))
            story.append(Spacer(1, 0.2*inch))
            
            viz_dir = os.path.join(self.framework.output_dir, 'visualizations')
            
            # Dendrogramme
            dendro_path = os.path.join(viz_dir, 'dendrogram.png')
            if os.path.exists(dendro_path):
                story.append(Paragraph("<b>Dendrogramme de Clustering Hiérarchique</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
                img = Image(dendro_path, width=6.5*inch, height=3.8*inch)
                story.append(img)
                story.append(PageBreak())
            
            # Heatmap de corrélation
            corr_path = os.path.join(viz_dir, 'correlation_heatmap.png')
            if os.path.exists(corr_path):
                story.append(Paragraph("<b>Matrice de Corrélation</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
                img = Image(corr_path, width=5.5*inch, height=4.5*inch)
                story.append(img)
                story.append(PageBreak())
            
            # Composition du portefeuille
            comp_path = os.path.join(viz_dir, 'portfolio_composition.png')
            if os.path.exists(comp_path):
                story.append(Paragraph("<b>Composition du Portefeuille</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
                img = Image(comp_path, width=6.5*inch, height=3.8*inch)
                story.append(img)
                story.append(PageBreak())
            
            # Performance
            perf_path = os.path.join(viz_dir, 'performance_comparison.png')
            if os.path.exists(perf_path):
                story.append(Paragraph("<b>Analyse de Performance</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
                img = Image(perf_path, width=6.5*inch, height=4.8*inch)
                story.append(img)
                story.append(PageBreak())
        
        # Conclusion
        story.append(Paragraph("Conclusion", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        conclusion_text = """
        Ce framework démontre l'application pratique du clustering hiérarchique pour la construction 
        de portefeuille et la classification d'actifs. La méthodologie fournit plusieurs insights éducatifs 
        importants sur la diversification et l'allocation d'actifs.
        <br/><br/>
        <b>Points Clés:</b><br/>
        • Le clustering basé sur la corrélation révèle des groupements naturels de classes d'actifs<br/>
        • Les bénéfices de diversification proviennent de corrélations inter-clusters faibles<br/>
        • L'allocation équilibrée par cluster fournit une diversification systématique<br/>
        • Les méthodes hiérarchiques offrent une classification transparente et interprétable<br/>
        <br/>
        <b>Utilisation Éducative:</b><br/>
        Ce framework est conçu pour l'enseignement en utilisant uniquement des sources de données 
        gratuites. Le code est transparent, bien documenté et extensible pour la recherche et 
        l'apprentissage supplémentaires.
        """
        story.append(Paragraph(conclusion_text, self.styles['CustomBody']))
        
        # Construire le PDF
        doc.build(story)
        
        print(f"✓ Rapport PDF généré: {output_path}")
        print("\n" + "="*70)
        print(f"✓ Rapport sauvegardé dans:")
        print(f"  {self.report_dir}/")
        print("="*70)
        
        return output_path
    
    def _calculate_div_ratio(self):
        """Calculer le ratio de diversification."""
        weights_array = self.framework.portfolio_weights.values.reshape(-1, 1)
        cov_matrix = self.framework.returns.cov().values
        portfolio_variance = float((weights_array.T @ cov_matrix @ weights_array)[0, 0])
        portfolio_vol = np.sqrt(portfolio_variance * 252)
        
        individual_vols = self.framework.returns.std() * np.sqrt(252)
        weighted_avg_vol = (self.framework.portfolio_weights * individual_vols).sum()
        
        return weighted_avg_vol / portfolio_vol
    
    def _calculate_effective_n(self):
        """Calculer le nombre effectif d'actifs."""
        herfindahl = (self.framework.portfolio_weights ** 2).sum()
        return 1 / herfindahl


# Test du module
if __name__ == "__main__":
    from portfolio_clustering_framework import PortfolioClusteringFramework
    from visualization_module import PortfolioVisualizer
    
    print("Test du générateur de rapports \n")
    
    # Créer un framework
    framework = PortfolioClusteringFramework(preset='simple')
    framework.run_full_analysis()
    
    # Générer les visualisations
    visualizer = PortfolioVisualizer(framework)
    visualizer.generate_all_visualizations()
    
    # Générer les rapports
    report_gen = PortfolioReportGenerator(framework, visualizer)
    csv_files = report_gen.export_csv_files()
    pdf_path = report_gen.generate_pdf_report()
    
    print(f"\n✓ Test réussi!")
    print(f"✓ Tous les fichiers dans: {framework.output_dir}/")