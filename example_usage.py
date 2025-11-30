#!/usr/bin/env python3
"""
Example Usage Script for Portfolio Clustering Framework
========================================================
This script demonstrates the complete workflow.
"""

from portfolio_clustering_framework import PortfolioClusteringFramework
from visualization_module import PortfolioVisualizer
from report_generator import PortfolioReportGenerator

def main():
    print("="*60)
    print("PORTFOLIO CLUSTERING FRAMEWORK - EXAMPLE")
    print("="*60)
    
    # Step 1: Initialize framework
    print("\n1. Initializing framework...")
    framework = PortfolioClusteringFramework()
    
    # Step 2: Run analysis
    print("\n2. Running analysis...")
    results = framework.run_full_analysis()
    
    # Step 3: Generate visualizations
    print("\n3. Generating visualizations...")
    visualizer = PortfolioVisualizer(framework)
    figures = visualizer.generate_all_visualizations()
    
    # Step 4: Create reports
    print("\n4. Creating reports...")
    report_gen = PortfolioReportGenerator(framework, visualizer)
    csv_files = report_gen.export_csv_files()
    pdf_path = report_gen.generate_pdf_report()
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  • PDF Report: portfolio_clustering_report.pdf")
    print("  • Visualizations: 5 PNG files")
    print("  • Data Exports: 6 CSV files")
    print("\n✓ Framework execution successful!")
    
    return framework, visualizer, report_gen

if __name__ == "__main__":
    framework, visualizer, report_gen = main()
