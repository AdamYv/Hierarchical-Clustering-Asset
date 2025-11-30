# Hierarchical-Clustering-Asset
## Quick Start
```
git clone https://github.com/AdamYv/Hierarchical-Clustering-Asset/
```
Dependence :
```
pip install -r requirements.txt
```
or 

```
pipenv install 
pipenv shell
pipenv run pip install -r requirements.txt
```


### Méthode 1: Portfolio Prédéfini

```python
from portfolio_clustering_framework import PortfolioClusteringFramework
from visualization_module import PortfolioVisualizer
from report_generator import PortfolioReportGenerator

# Choisir: 'default', 'tech', 'conservative', 'global', 'simple'
framework = PortfolioClusteringFramework(preset='tech')
results = framework.run_full_analysis()

# Générer visualisations et rapports
visualizer = PortfolioVisualizer(framework)
visualizer.generate_all_visualizations()

report_gen = PortfolioReportGenerator(framework, visualizer)
report_gen.export_csv_files()
report_gen.generate_pdf_report()
```

### Méthode 2: Liste Personnalisée

```python
mes_actifs = ['AAPL', 'MSFT', 'GLD', 'TLT', 'VNQ']

framework = PortfolioClusteringFramework(tickers=mes_actifs)
results = framework.run_full_analysis()

# Puis générer visualisations et rapports
```

---

## 📊 Portfolios Prédéfinis

| Preset | Actifs | Description |
|--------|--------|-------------|
| **default** | 12 | Portfolio diversifié standard |
| **tech** | 8 | Grandes valeurs technologiques |
| **conservative** | 8 | Obligations et défensives |
| **global** | 8 | ETFs diversifiés mondialement |
| **simple** | 4 | Portfolio minimaliste |

---




# Contributing
This is an educational framework. Contributions welcome!
# Support
For educational purposes only. Not financial advice.
