import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Import the simulation function from temp.py
from temp import run_retraining_simulations
from counterfactual_fraud_model.config import RetrainingStrategy


class RetrainingSimulationAnalyzer:
    """
    Analyzer for retraining simulation results with comprehensive visualization capabilities.
    """
    
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize the analyzer with simulation results.
        
        Args:
            results: List of simulation results from run_retraining_simulations
        """
        self.results = results
        self.df = self._process_results()
        
    def _process_results(self) -> pd.DataFrame:
        """
        Process raw simulation results into a structured DataFrame.
        
        Returns:
            DataFrame with columns: exploration_rate, strategy, metric, mean, p025, p975
        """
        processed_data = []
        
        for result in self.results:
            exploration_rate = result['simulation_metadata']['exploration_rate']
            strategy = result['simulation_metadata']['strategy']
            ope_metrics = result['ope_metrics']
            
            # Extract each metric
            for metric_name, metric_data in ope_metrics.items():
                processed_data.append({
                    'exploration_rate': exploration_rate,
                    'strategy': strategy,
                    'metric': metric_name,
                    'mean': metric_data['mean'],
                    'p025': metric_data['p025'],
                    'p975': metric_data['p975'],
                    'ci_width': metric_data['p975'] - metric_data['p025']
                })
        
        return pd.DataFrame(processed_data)
    
    def create_line_plots_with_ci(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create line plots with confidence interval ribbons for each metric.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        metrics = self.df['metric'].unique()
        n_metrics = len(metrics)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        axes = axes.flatten()
        
        colors = {'filtering': '#2E86C1', 'weighting': '#E74C3C'}
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            metric_data = self.df[self.df['metric'] == metric]
            
            # Plot for each strategy
            for strategy in metric_data['strategy'].unique():
                strategy_data = metric_data[metric_data['strategy'] == strategy].sort_values('exploration_rate')
                
                x = strategy_data['exploration_rate']
                y = strategy_data['mean']
                y_lower = strategy_data['p025']
                y_upper = strategy_data['p975']
                
                # Plot line
                ax.plot(x, y, 'o-', color=colors[strategy], linewidth=2, 
                       markersize=6, label=f'{strategy.title()}', alpha=0.8)
                
                # Plot confidence interval
                ax.fill_between(x, y_lower, y_upper, color=colors[strategy], 
                              alpha=0.2, label=f'{strategy.title()} 95% CI')
            
            # Formatting
            ax.set_xlabel('Exploration Rate')
            ax.set_ylabel(f'{metric.replace("_", " ").title()}')
            ax.set_title(f'{metric.replace("_", " ").title()} vs Exploration Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to show all exploration rates
            ax.set_xticks(self.df['exploration_rate'].unique())
        
        # Remove empty subplot if odd number of metrics
        if n_metrics < len(axes):
            fig.delaxes(axes[-1])
            
        fig.suptitle('OPE Metrics Comparison: Filtering vs Weighting Strategies', 
                     fontsize=16, fontweight='bold')
        
        return fig
    
    def create_bar_plots_with_error_bars(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create bar plots with error bars for each metric.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        metrics = self.df['metric'].unique()
        n_metrics = len(metrics)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        axes = axes.flatten()
        
        colors = {'filtering': '#2E86C1', 'weighting': '#E74C3C'}
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            metric_data = self.df[self.df['metric'] == metric]
            
            # Prepare data for grouped bar plot
            exploration_rates = sorted(metric_data['exploration_rate'].unique())
            strategies = sorted(metric_data['strategy'].unique())
            
            x = np.arange(len(exploration_rates))
            width = 0.35
            
            for j, strategy in enumerate(strategies):
                strategy_data = metric_data[metric_data['strategy'] == strategy].sort_values('exploration_rate')
                
                means = strategy_data['mean'].values
                errors = [(strategy_data['mean'] - strategy_data['p025']).values,
                         (strategy_data['p975'] - strategy_data['mean']).values]
                
                ax.bar(x + j*width, means, width, 
                      color=colors[strategy], alpha=0.7, 
                      label=f'{strategy.title()}',
                      yerr=errors, capsize=5, error_kw={'alpha': 0.6})
            
            # Formatting
            ax.set_xlabel('Exploration Rate')
            ax.set_ylabel(f'{metric.replace("_", " ").title()}')
            ax.set_title(f'{metric.replace("_", " ").title()} vs Exploration Rate')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels([f'{rate:.2f}' for rate in exploration_rates])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove empty subplot if odd number of metrics
        if n_metrics < len(axes):
            fig.delaxes(axes[-1])
            
        fig.suptitle('OPE Metrics Comparison: Filtering vs Weighting Strategies (Bar Plot)', 
                     fontsize=16, fontweight='bold')
        
        return fig
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of the results.
        
        Returns:
            DataFrame with summary statistics
        """
        summary = self.df.pivot_table(
            index=['metric', 'exploration_rate'], 
            columns='strategy',
            values=['mean', 'ci_width'],
            aggfunc='first'
        ).round(4)
        
        return summary
    
    def print_insights(self):
        """
        Print key insights from the analysis.
        """
        print("=" * 60)
        print("RETRAINING SIMULATION ANALYSIS - KEY INSIGHTS")
        print("=" * 60)
        
        # Overall performance comparison
        mean_performance = self.df.groupby(['strategy', 'metric'])['mean'].mean()
        
        print("\n1. AVERAGE PERFORMANCE ACROSS ALL EXPLORATION RATES:")
        for strategy in ['filtering', 'weighting']:
            print(f"\n{strategy.upper()} Strategy:")
            for metric in self.df['metric'].unique():
                value = mean_performance[strategy, metric]
                print(f"  • {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Confidence interval width analysis
        print("\n2. UNCERTAINTY ANALYSIS (Average CI Width):")
        ci_analysis = self.df.groupby(['strategy', 'metric'])['ci_width'].mean()
        
        for strategy in ['filtering', 'weighting']:
            print(f"\n{strategy.upper()} Strategy:")
            for metric in self.df['metric'].unique():
                value = ci_analysis[strategy, metric]
                print(f"  • {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Exploration rate impact
        print("\n3. EXPLORATION RATE IMPACT:")
        for metric in self.df['metric'].unique():
            print(f"\n{metric.replace('_', ' ').title()}:")
            metric_data = self.df[self.df['metric'] == metric]
            
            for strategy in ['filtering', 'weighting']:
                strategy_data = metric_data[metric_data['strategy'] == strategy].sort_values('exploration_rate')
                trend = "increasing" if strategy_data['mean'].iloc[-1] > strategy_data['mean'].iloc[0] else "decreasing"
                change = abs(strategy_data['mean'].iloc[-1] - strategy_data['mean'].iloc[0])
                print(f"  • {strategy.title()}: {trend} trend (Δ{change:.4f})")


def run_analysis():
    """
    Run the complete analysis pipeline.
    """
    print("Starting retraining simulation analysis...")
    
    # Define simulation parameters
    exploration_rates = np.array([0.05, 0.1, 0.2])
    strategies = [RetrainingStrategy.FILTERING, RetrainingStrategy.WEIGHTING]
    
    print(f"Running simulations with:")
    print(f"  • Exploration rates: {exploration_rates}")
    print(f"  • Strategies: {[s.value for s in strategies]}")
    
    # Run simulations
    results, reference_data = run_retraining_simulations(
        exploration_rates=exploration_rates,
        strategies=strategies,
        sample_size=10_000,
        random_state=42
    )
    
    # Create analyzer
    analyzer = RetrainingSimulationAnalyzer(results)
    
    # Generate visualizations
    print("\nCreating visualizations...")
    
    # Line plots with confidence intervals
    fig_line = analyzer.create_line_plots_with_ci()
    fig_line.savefig('retraining_simulation_line_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Line plots with confidence intervals saved as 'retraining_simulation_line_plots.png'")
    
    # Bar plots with error bars
    fig_bar = analyzer.create_bar_plots_with_error_bars()
    fig_bar.savefig('retraining_simulation_bar_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Bar plots with error bars saved as 'retraining_simulation_bar_plots.png'")
    
    # Generate summary table
    summary_table = analyzer.create_summary_table()
    summary_table.to_csv('retraining_simulation_summary.csv')
    print("✓ Summary table saved as 'retraining_simulation_summary.csv'")
    
    # Print insights
    analyzer.print_insights()
    
    # Display plots
    plt.show()
    
    return analyzer, results, reference_data


if __name__ == "__main__":
    analyzer, results, reference_data = run_analysis()