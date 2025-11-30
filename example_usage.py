
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: example_usage.py
# execution: true

"""
Script d'Exemple Complet - Framework 
==========================================
Démontre l'utilisation complète du framework avec toutes les fonctionnalités.
"""

from portfolio_clustering_framework import PortfolioClusteringFramework
from visualization_module import PortfolioVisualizer
from report_generator import PortfolioReportGenerator
import os

def main():
    """Exécuter une analyse complète de portefeuille."""
    
    print("="*70)
    print("FRAMEWORK DE CLUSTERING HIÉRARCHIQUE ")
    print("Exemple d'Utilisation Complète")
    print("="*70)
    
    # ========================================================================
    # EXEMPLE 1: Utiliser un portfolio prédéfini
    # ========================================================================
    print("\n" + "="*70)
    print("EXEMPLE 1: Portfolio Prédéfini 'tech'")
    print("="*70)
    
    framework1 = PortfolioClusteringFramework(preset='tech')
    results1 = framework1.run_full_analysis()
    
    visualizer1 = PortfolioVisualizer(framework1)
    visualizer1.generate_all_visualizations()
    
    report_gen1 = PortfolioReportGenerator(framework1, visualizer1)
    report_gen1.export_csv_files()
    report_gen1.generate_pdf_report()
    
    print(f"\n✓ Exemple 1 terminé!")
    print(f"✓ Tous les fichiers dans: {framework1.output_dir}/")
    
    # ========================================================================
    # EXEMPLE 2: Liste d'actifs personnalisée
    # ========================================================================
    print("\n" + "="*70)
    print("EXEMPLE 2: Liste Personnalisée")
    print("="*70)
    
    # Définir votre propre liste d'actifs
    mes_actifs = ['AAPL', 'MSFT', 'GOOGL', 'GLD', 'TLT', 'VNQ', 'BND']
    
    framework2 = PortfolioClusteringFramework(
        tickers=mes_actifs,
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    results2 = framework2.run_full_analysis()
    
    visualizer2 = PortfolioVisualizer(framework2)
    visualizer2.generate_all_visualizations()
    
    report_gen2 = PortfolioReportGenerator(framework2, visualizer2)
    report_gen2.export_csv_files()
    report_gen2.generate_pdf_report()
    
    print(f"\n✓ Exemple 2 terminé!")
    print(f"✓ Tous les fichiers dans: {framework2.output_dir}/")
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print("\n" + "="*70)
    print("RÉSUMÉ DES ANALYSES")
    print("="*70)
    
    print(f"\n📁 Dossiers créés:")
    print(f"   1. {framework1.output_dir}/")
    print(f"      ├── visualizations/ (5 PNG)")
    print(f"      ├── data/ (6 CSV)")
    print(f"      └── reports/ (1 PDF)")
    print(f"\n   2. {framework2.output_dir}/")
    print(f"      ├── visualizations/ (5 PNG)")
    print(f"      ├── data/ (6 CSV)")
    print(f"      └── reports/ (1 PDF)")

    return framework1, framework2

if __name__ == "__main__":
    fw1, fw2 = main()
    
    print("\n" + "="*70)
    print("GUIDE D'UTILISATION RAPIDE")
    print("="*70)
    print("""
# Méthode 1: Utiliser un preset
framework = PortfolioClusteringFramework(preset='tech')

# Méthode 2: Liste personnalisée
framework = PortfolioClusteringFramework(
    tickers=['AAPL', 'MSFT', 'GLD', 'TLT']
)

# Méthode 3: Avec dates personnalisées
framework = PortfolioClusteringFramework(
    tickers=['AAPL', 'MSFT', 'GLD'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Exécuter l'analyse complète
results = framework.run_full_analysis()

# Générer visualisations et rapports
from visualization_module import PortfolioVisualizer
from report_generator import PortfolioReportGenerator

visualizer = PortfolioVisualizer(framework)
visualizer.generate_all_visualizations()

report_gen = PortfolioReportGenerator(framework, visualizer)
report_gen.export_csv_files()
report_gen.generate_pdf_report()

# Tous les fichiers sont dans: framework.output_dir/
    """)