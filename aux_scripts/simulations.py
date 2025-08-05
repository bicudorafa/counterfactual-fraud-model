#!/usr/bin/env python3
"""
Main simulation runner that executes all pipeline simulation analyses.

This script runs all available simulation analyses from the utils directory,
providing a comprehensive evaluation of different pipeline configurations
and parameters.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the main functions from each analysis script
from notebooks.utils.simulation_analysis import main as run_standard_simulation_analysis
from notebooks.utils.synthetic_simulation_analysis import main as run_synthetic_simulation_analysis
from notebooks.utils.retraining_simulation_analysis import run_analysis as run_retraining_simulation_analysis


def run_all_simulations():
    """
    Run all available simulation analyses in sequence.
    
    This function executes:
    1. Standard Off-Policy Evaluation Pipeline Analysis
    2. Synthetic Off-Policy Evaluation Pipeline Analysis  
    3. Synthetic Retraining Pipeline Analysis
    """
    
    print("=" * 80)
    print("COMPREHENSIVE SIMULATION ANALYSIS SUITE")
    print("=" * 80)
    print("\nThis suite will run three different simulation analyses:")
    print("1. Standard Off-Policy Evaluation Pipeline")
    print("2. Synthetic Off-Policy Evaluation Pipeline") 
    print("3. Synthetic Retraining Pipeline")
    print("\nEach analysis generates plots and summary tables.")
    print("=" * 80)
    
    try:
        # 1. Standard Pipeline Analysis
        print("\n" + "="*60)
        print("RUNNING ANALYSIS 1/3: STANDARD OFF-POLICY EVALUATION")
        print("="*60)
        run_standard_simulation_analysis()
        print("✓ Standard analysis completed successfully!")
        
        # 2. Synthetic Pipeline Analysis
        print("\n" + "="*60)
        print("RUNNING ANALYSIS 2/3: SYNTHETIC OFF-POLICY EVALUATION")
        print("="*60)
        run_synthetic_simulation_analysis()
        print("✓ Synthetic analysis completed successfully!")
        
        # 3. Retraining Pipeline Analysis
        print("\n" + "="*60)
        print("RUNNING ANALYSIS 3/3: SYNTHETIC RETRAINING PIPELINE")
        print("="*60)
        analyzer, results, reference_data = run_retraining_simulation_analysis()
        print("✓ Retraining analysis completed successfully!")
        
        # Final summary
        print("\n" + "="*80)
        print("ALL SIMULATION ANALYSES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nSummary of generated outputs:")
        print("• Multiple visualization plots displayed")
        print("• Summary tables printed to console")
        print("\nAll analyses provide insights into:")
        print("• Policy performance across different exploration rates")
        print("• Model comparison and calibration")
        print("• Counterfactual estimation accuracy")
        print("• Retraining strategy effectiveness")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during simulation analysis: {str(e)}")
        print("Please check the error details above and ensure all dependencies are available.")
        return False


def run_individual_analysis(analysis_type: str):
    """
    Run a specific individual analysis.
    
    Args:
        analysis_type: Type of analysis to run ('standard', 'synthetic', 'retraining')
    """
    
    if analysis_type.lower() == 'standard':
        print("Running Standard Off-Policy Evaluation Analysis...")
        run_standard_simulation_analysis()
        
    elif analysis_type.lower() == 'synthetic':
        print("Running Synthetic Off-Policy Evaluation Analysis...")
        run_synthetic_simulation_analysis()
        
    elif analysis_type.lower() == 'retraining':
        print("Running Synthetic Retraining Pipeline Analysis...")
        analyzer, results, reference_data = run_retraining_simulation_analysis()
        
    else:
        print(f"Unknown analysis type: {analysis_type}")
        print("Available types: 'standard', 'synthetic', 'retraining'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive simulation analyses for counterfactual fraud models"
    )
    parser.add_argument(
        '--type', 
        choices=['all', 'standard', 'synthetic', 'retraining'],
        default='all',
        help="Type of analysis to run (default: all)"
    )
    
    args = parser.parse_args()
    
    if args.type == 'all':
        success = run_all_simulations()
        sys.exit(0 if success else 1)
    else:
        run_individual_analysis(args.type)