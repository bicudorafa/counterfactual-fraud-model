"""Off Policy Evaluation Simulator for Counterfactual Fraud Model Analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from .pipeline import OffPolicyEvaluationPipeline


class OffPolicyEvaluationSimulator:
    """
    Simulates off-policy evaluation across multiple exploration rates and generates plots.
    
    Provides comprehensive analysis of how exploration rates affect counterfactual
    evaluation quality and model performance estimation.
    """
    
    def __init__(
        self,
        # Data generator parameters
        alpha: float = 0.5,
        beta_param: float = 10.0,
        mean: float = 0.0,
        sd: float = 0.1,
        sample_size: int = 10_000,
        # Logging policy parameters
        cutoff: float = 0.05,
        propensity_type: str = "linear",
        # Counterfactual estimator parameters
        n_bootstrap: int = 1000,
        random_state: Optional[int] = None
    ):
        """
        Initialize OffPolicyEvaluationSimulator.
        
        Args:
            alpha: Alpha parameter for beta distribution
            beta_param: Beta parameter for beta distribution
            mean: Mean for normal error distribution
            sd: Standard deviation for normal error distribution
            sample_size: Number of samples to generate
            cutoff: Score threshold for blocking transactions
            propensity_type: Type of propensity function
            n_bootstrap: Number of bootstrap repetitions
            random_state: Random seed for reproducibility
        """
        self.base_params = {
            'alpha': alpha,
            'beta_param': beta_param,
            'mean': mean,
            'sd': sd,
            'sample_size': sample_size,
            'cutoff': cutoff,
            'propensity_type': propensity_type,
            'n_bootstrap': n_bootstrap,
            'random_state': random_state
        }
        
        # All metrics are now calculated by default
        self.metrics = ['precision', 'recall', 'f1', 'approval_rate', 'fraud_rate']
    
    def simulate_exploration_rates(
        self, 
        exploration_rates: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Simulate off-policy evaluation across multiple exploration rates.
        
        Args:
            exploration_rates: List of exploration rates to test
            
        Returns:
            Dictionary with simulation results and data
        """
        if exploration_rates is None:
            exploration_rates = np.linspace(0.01, 0.1, 10).tolist()
        
        results = {
            'exploration_rates': exploration_rates,
            'metrics_by_rate': {},
            'data': None,
            'baseline_data': None
        }
        
        # Run simulation for each exploration rate
        for rate in exploration_rates:
            pipeline = OffPolicyEvaluationPipeline(
                exploration_rate=rate,
                **{k: v for k, v in self.base_params.items() 
                   if k != 'exploration_rate'}
            )
            
            pipeline_results = pipeline.run_pipeline()
            
            # Store results
            results['metrics_by_rate'][rate] = {
                'metrics': pipeline_results['metrics'],
                'statistics': pipeline_results['statistics']
            }
            
            # Store data from first run for plotting
            if results['data'] is None:
                results['data'] = pipeline_results['data']
        
        # Generate baseline data (no exploration) for comparison
        baseline_pipeline = OffPolicyEvaluationPipeline(
            exploration_rate=0.0,
            **{k: v for k, v in self.base_params.items() 
               if k != 'exploration_rate'}
        )
        baseline_data = baseline_pipeline.data_generator.generate_data()
        results['baseline_data'] = baseline_data
        
        return results
    
    def plot_model_score_distribution(
        self, 
        data: pd.DataFrame, 
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot the distribution of model scores."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot histogram of model scores
        ax.hist(data['model_scores'], bins=50, alpha=0.7, 
                density=True, label='Model Scores')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Model Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_calibration(
        self, 
        data: pd.DataFrame, 
        n_bins: int = 10,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot calibration curve showing actual vs predicted fraud rates."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Create bins based on model scores
        data_copy = data.copy()
        data_copy['score_bin'] = pd.cut(
            data_copy['model_scores'], 
            bins=n_bins, 
            include_lowest=True
        )
        
        # Calculate actual fraud rate by bin
        bin_stats = data_copy.groupby('score_bin').agg({
            'model_scores': 'mean',
            'is_fraud': ['mean', 'count']
        }).round(3)
        
        bin_stats.columns = ['mean_score', 'fraud_rate', 'count']
        bin_stats = bin_stats.dropna()
        
        # Plot calibration
        ax.plot(bin_stats['mean_score'], bin_stats['fraud_rate'], 
                'o-', label='Actual')
        ax.plot([0, 1], [0, 1], '--', color='red', label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Score')
        ax.set_ylabel('Actual Fraud Rate')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_precision_recall_curve(
        self, 
        data: pd.DataFrame,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot precision-recall curve for different thresholds."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        thresholds = np.linspace(0, 1, 50)
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            y_pred = (data['model_scores'] > threshold).astype(int)
            y_true = data['is_fraud'].values
            
            # Calculate precision and recall
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        ax.plot(recalls, precisions, 'b-', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        
        # Add cutoff point
        cutoff = self.base_params['cutoff']
        y_pred_cutoff = (data['model_scores'] > cutoff).astype(int)
        y_true = data['is_fraud'].values
        
        tp = np.sum((y_true == 1) & (y_pred_cutoff == 1))
        fp = np.sum((y_true == 0) & (y_pred_cutoff == 1))
        fn = np.sum((y_true == 1) & (y_pred_cutoff == 0))
        
        precision_cutoff = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_cutoff = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        ax.plot(recall_cutoff, precision_cutoff, 'ro', markersize=8,
                label=f'Cutoff = {cutoff:.3f}')
        ax.legend()
        
        return ax
    
    def plot_metrics_by_exploration_rate(
        self, 
        simulation_results: Dict[str, Any],
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """Plot how metrics change with exploration rate."""
        exploration_rates = simulation_results['exploration_rates']
        metrics_by_rate = simulation_results['metrics_by_rate']
        
        n_metrics = len(self.metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(self.metrics):
            ax = axes[i]
            
            means = []
            lower_bounds = []
            upper_bounds = []
            
            for rate in exploration_rates:
                metric_data = metrics_by_rate[rate]['metrics'][metric_name]
                means.append(metric_data['mean'])
                lower_bounds.append(metric_data['p025'])
                upper_bounds.append(metric_data['p975'])
            
            # Plot mean with confidence intervals
            ax.plot(exploration_rates, means, 'b-o', label='Mean')
            ax.fill_between(exploration_rates, lower_bounds, upper_bounds,
                           alpha=0.3, label='95% CI')
            
            ax.set_xlabel('Exploration Rate')
            ax.set_ylabel(metric_name.title())
            ax.set_title(f'{metric_name.title()} vs Exploration Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def run_complete_simulation(
        self, 
        exploration_rates: Optional[List[float]] = None,
        save_plots: bool = False,
        plot_dir: str = "plots"
    ) -> Dict[str, Any]:
        """
        Run complete simulation and generate all plots.
        
        Args:
            exploration_rates: List of exploration rates to test
            save_plots: Whether to save plots to files
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary with all simulation results and plot objects
        """
        print("Running off-policy evaluation simulation...")
        
        # Run simulation
        results = self.simulate_exploration_rates(exploration_rates)
        
        # Create plots
        print("Generating plots...")
        
        # 1. Model score distribution
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        self.plot_model_score_distribution(results['baseline_data'], ax1)
        
        # 2. Calibration plot
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        self.plot_calibration(results['baseline_data'], ax=ax2)
        
        # 3. Precision-recall curve
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        self.plot_precision_recall_curve(results['baseline_data'], ax=ax3)
        
        # 4. Metrics by exploration rate
        fig4 = self.plot_metrics_by_exploration_rate(results)
        
        plots = {
            'score_distribution': (fig1, ax1),
            'calibration': (fig2, ax2),
            'precision_recall': (fig3, ax3),
            'metrics_by_exploration': fig4
        }
        
        # Save plots if requested
        if save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
            
            fig1.savefig(f"{plot_dir}/score_distribution.png", dpi=300, bbox_inches='tight')
            fig2.savefig(f"{plot_dir}/calibration.png", dpi=300, bbox_inches='tight')
            fig3.savefig(f"{plot_dir}/precision_recall.png", dpi=300, bbox_inches='tight')
            fig4.savefig(f"{plot_dir}/metrics_by_exploration.png", dpi=300, bbox_inches='tight')
            
            print(f"Plots saved to {plot_dir}/")
        
        # Show plots
        plt.show()
        
        # Add plots to results
        results['plots'] = plots
        results['parameters'] = self.base_params.copy()
        
        print("Simulation complete!")
        return results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of simulation results."""
        print("\n" + "="*50)
        print("SIMULATION SUMMARY")
        print("="*50)
        
        print(f"Data size: {len(results['baseline_data'])}")
        print(f"Fraud rate: {results['baseline_data']['is_fraud'].mean():.3f}")
        print(f"Cutoff threshold: {self.base_params['cutoff']:.3f}")
        
        print(f"\nExploration rates tested: {len(results['exploration_rates'])}")
        print(f"Range: {min(results['exploration_rates']):.3f} - {max(results['exploration_rates']):.3f}")
        
        # Show metrics for a few exploration rates
        rates_to_show = [results['exploration_rates'][0], 
                        results['exploration_rates'][len(results['exploration_rates'])//2],
                        results['exploration_rates'][-1]]
        
        print("\nMetrics by exploration rate:")
        for rate in rates_to_show:
            print(f"\nExploration rate: {rate:.3f}")
            metrics = results['metrics_by_rate'][rate]['metrics']
            for metric_name in self.metrics:
                metric_data = metrics[metric_name]
                print(f"  {metric_name}: {metric_data['mean']:.3f} "
                      f"({metric_data['p025']:.3f} - {metric_data['p975']:.3f})")
    
    def get_params(self) -> dict:
        """Return the current parameters."""
        return self.base_params.copy() 