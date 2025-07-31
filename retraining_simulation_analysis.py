import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from pathlib import Path

from counterfactual_fraud_model.config import (
    SyntheticRetrainingConfig,
    SyntheticOffPolicyEvaluationConfig,
    SyntheticDataConfig,
    ModelConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    RetrainingConfig,
    RetrainingModelConfig,
    ModelType,
    RetrainingStrategy
)
from counterfactual_fraud_model.pipelines import SyntheticRetrainingPipeline
# TODO: entender se o pre_processing está afetando evaluation tambem, porque nas metricas normais está melhor, mas no OPE está estranho


def run_retraining_simulations(
    exploration_rates: np.ndarray,
    strategies: List[RetrainingStrategy],
    cutoff: float = 0.05,
    sample_size: int = 100_000,
    classification_threshold: float = 0.1,
    random_state: int = 42
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Run multiple SyntheticRetrainingPipeline simulations with different exploration rates and strategies.
    
    Args:
        exploration_rates: Array of exploration rate values to test
        strategies: List of retraining strategies to compare
        cutoff: Fixed cutoff value for all simulations
        sample_size: Number of samples in synthetic dataset
        classification_threshold: Threshold for binary classification
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (list of simulation results, reference data for plots)
    """
    # Calculate total number of simulations
    total_sims = len(exploration_rates) * len(strategies)
    print(f"Running {total_sims} simulations...")
    print(f"  - Exploration rates: {len(exploration_rates)} values")
    print(f"  - Retraining strategies: {len(strategies)} strategies")
    
    results = []
    reference_data = None  # Will store base data for plotting

    
    # Run simulations for each combination
    for exploration_rate in exploration_rates:
        print(f"Running simulations for exploration_rate={exploration_rate:.3f}")

        # Base config for most simulations
        config = SyntheticRetrainingConfig(
            base_config=SyntheticOffPolicyEvaluationConfig(
                synthetic_data=SyntheticDataConfig(
                    n_samples=sample_size,
                    n_features=15,
                    n_informative=10,
                    n_redundant=3,
                    n_repeated=0,
                    random_state=random_state  # Same seed for fair comparison
                ),
                model=ModelConfig(
                    model_type=ModelType.LIGHTGBM,
                    random_state=random_state
                ),
                logging_policy=LoggingPolicyConfig(
                    cutoff=cutoff,
                    exploration_rate=exploration_rate,
                    random_state=random_state
                ),
                counterfactual_estimator=CounterfactualEstimatorConfig(
                    n_bootstrap=5000,  # Reasonable for analysis
                    random_state=random_state
                )
            )
        )
        
        # Initialize and run pipeline
        pipeline = SyntheticRetrainingPipeline(config)

        pipeline.generate_logging_policy_data()

        for strategy in strategies:
            print(f"Strategy={strategy.value} simulation")

            retraining_config = RetrainingConfig(
                retrain_test_size=0.3,
                retrain_model=RetrainingModelConfig(
                    base_model=ModelConfig(
                        model_type=ModelType.LIGHTGBM,
                        random_state=random_state
                    ),
                    strategy=strategy, # it doesn't matter which strategy we use, it'll change dinamically through the simulation
                    classification_threshold=classification_threshold
                )
            )
            
            result = pipeline.run_retrain_pipeline(retraining_config=retraining_config)  # Removed include_data parameter
        
            # Add metadata for easy tracking
            result['simulation_metadata'] = {
                'exploration_rate': exploration_rate,
                'strategy': strategy.value,
                'result': result
            }
            
            results.append(result)
            
            # Store reference data from first simulation for plotting
            if reference_data is None:
                # Get the base data for plotting (same across all simulations with same random_state)
                reference_data = pipeline.get_test_policy_data()
    
    return results, reference_data


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
    
    def create_dot_chart(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create dot charts with confidence intervals for each metric.
        
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
            
            # Prepare data for dot chart
            exploration_rates = sorted(metric_data['exploration_rate'].unique())
            strategies = sorted(metric_data['strategy'].unique())
            
            x_offset = 0.001  # Small offset for different strategies on x-axis
            
            for j, strategy in enumerate(strategies):
                strategy_data = metric_data[metric_data['strategy'] == strategy].sort_values('exploration_rate')
                
                # Use exploration rates as x positions (with small offset for strategies)
                x_positions = strategy_data['exploration_rate'].values + j * x_offset
                
                means = strategy_data['mean'].values
                errors_lower = (strategy_data['mean'] - strategy_data['p025']).values
                errors_upper = (strategy_data['p975'] - strategy_data['mean']).values
                
                # Plot dots
                ax.scatter(x_positions, means, color=colors[strategy], s=80, 
                          alpha=0.8, label=f'{strategy.title()}', zorder=3)
                
                # Plot vertical error bars (confidence intervals)
                ax.errorbar(x_positions, means, yerr=[errors_lower, errors_upper],
                           fmt='none', color=colors[strategy], alpha=0.6, 
                           linewidth=2, capsize=4, capthick=2, zorder=2)
            
            # Formatting
            ax.set_xlabel('Exploration Rate')
            ax.set_ylabel(f'{metric.replace("_", " ").title()}')
            ax.set_title(f'{metric.replace("_", " ").title()} vs Exploration Rate')
            
            # Set x-axis ticks to exploration rates
            ax.set_xticks(exploration_rates)
            ax.set_xticklabels([f'{rate:.3f}' for rate in exploration_rates])
            
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot if odd number of metrics
        if n_metrics < len(axes):
            fig.delaxes(axes[-1])
            
        fig.suptitle('OPE Metrics Comparison: Filtering vs Weighting Strategies (Dot Chart)', 
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
    exploration_rates = np.linspace(0.01, 0.1, 5)
    strategies = [RetrainingStrategy.FILTERING, RetrainingStrategy.WEIGHTING]
    
    print(f"Running simulations with:")
    print(f"  • Exploration rates: {exploration_rates}")
    print(f"  • Strategies: {[s.value for s in strategies]}")
    
    # Run simulations
    results, reference_data = run_retraining_simulations(
        exploration_rates=exploration_rates,
        strategies=strategies,
        sample_size=300_000,
        random_state=42
    )
    
    # Create analyzer
    analyzer = RetrainingSimulationAnalyzer(results)
    
    # Generate visualizations
    print("\nCreating visualizations...")
    
    # Dot chart with confidence intervals
    fig_dot = analyzer.create_dot_chart()
    fig_dot.savefig('retraining_simulation_dot_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Dot chart with confidence intervals saved as 'retraining_simulation_dot_chart.png'")
    
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