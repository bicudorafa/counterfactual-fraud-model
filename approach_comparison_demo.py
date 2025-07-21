"""
Comparison Demo: Original Probabilistic vs New Synthetic Approach

This script demonstrates the differences between the original probabilistic
data generation approach and the new synthetic dataset approach.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.counterfactual_fraud_model import (
    OffPolicyEvaluationPipeline,
    SyntheticOffPolicyEvaluationPipeline
)


def compare_approaches():
    """Compare the original and synthetic approaches side by side."""
    print("=" * 80)
    print("COMPARISON: Original Probabilistic vs New Synthetic Approach")
    print("=" * 80)
    
    # Common parameters
    cutoff = 0.05
    exploration_rate = 0.05
    random_state = 42
    
    # 1. Original Probabilistic Approach
    print("\n1. ORIGINAL PROBABILISTIC APPROACH")
    print("-" * 50)
    
    original_pipeline = OffPolicyEvaluationPipeline(
        sample_size=20000,
        random_state=random_state
    )
    
    original_results = original_pipeline.run_pipeline(
        cutoff=cutoff,
        exploration_rate=exploration_rate,
        include_data=True
    )
    
    print("✓ Original approach completed")
    print(f"  Data shape: {original_results['data'].shape}")
    print(f"  Data columns: {list(original_results['data'].columns)}")
    print(f"  True fraud rate: {original_results['statistics']['true_fraud_rate']:.4f}")
    print(f"  Policy approval rate: {original_results['statistics']['policy_approval_rate']:.4f}")
    print(f"  Policy fraud rate: {original_results['statistics']['policy_fraud_rate']:.4f}")
    
    # 2. New Synthetic Approach
    print("\n2. NEW SYNTHETIC APPROACH")
    print("-" * 50)
    
    synthetic_pipeline = SyntheticOffPolicyEvaluationPipeline(
        n_samples=20000,
        model_type="lightgbm",
        random_state=random_state
    )
    
    synthetic_results = synthetic_pipeline.run_pipeline(
        cutoff=cutoff,
        exploration_rate=exploration_rate,
        include_data=True
    )
    
    print("✓ Synthetic approach completed")
    print(f"  Data shape: {synthetic_results['data'].shape}")
    print(f"  Data columns: {list(synthetic_results['data'].columns)}")
    print(f"  True fraud rate: {synthetic_results['statistics']['true_fraud_rate']:.4f}")
    print(f"  Policy approval rate: {synthetic_results['statistics']['policy_approval_rate']:.4f}")
    print(f"  Policy fraud rate: {synthetic_results['statistics']['policy_fraud_rate']:.4f}")
    print(f"  Model ROC AUC: {synthetic_results['model_performance']['roc_auc']:.4f}")
    print(f"  Model Precision: {synthetic_results['model_performance']['precision']:.4f}")
    
    # 3. Side-by-side comparison
    print("\n3. SIDE-BY-SIDE COMPARISON")
    print("-" * 50)
    
    comparison_data = {
        'Metric': [
            'Total Transactions',
            'True Fraud Rate', 
            'Policy Approval Rate',
            'Policy Fraud Rate',
            'Approval Rate Increase',
            'Data Generation Method',
            'Model Scores Source',
            'Additional Features'
        ],
        'Original Approach': [
            len(original_results['data']),
            f"{original_results['statistics']['true_fraud_rate']:.4f}",
            f"{original_results['statistics']['policy_approval_rate']:.4f}",
            f"{original_results['statistics']['policy_fraud_rate']:.4f}",
            f"{original_results['statistics']['approval_rate_increase']:.4f}",
            'Beta distribution',
            'Probabilistic generation',
            'None'
        ],
        'Synthetic Approach': [
            len(synthetic_results['data']),
            f"{synthetic_results['statistics']['true_fraud_rate']:.4f}",
            f"{synthetic_results['statistics']['policy_approval_rate']:.4f}",
            f"{synthetic_results['statistics']['policy_fraud_rate']:.4f}",
            f"{synthetic_results['statistics']['approval_rate_increase']:.4f}",
            'sklearn make_classification',
            'Trained ML model predictions',
            'Model performance metrics'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))
    
    return original_results, synthetic_results


def plot_score_distributions(original_results, synthetic_results):
    """Plot model score distributions for both approaches."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original approach
    original_data = original_results['data']
    axes[0].hist(original_data['model_scores'], bins=50, alpha=0.7, edgecolor='black', color='blue')
    axes[0].set_title('Original Approach\n(Beta Distribution)')
    axes[0].set_xlabel('Model Scores')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Synthetic approach
    synthetic_data = synthetic_results['data']
    axes[1].hist(synthetic_data['model_scores'], bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1].set_title('Synthetic Approach\n(LightGBM Predictions)')
    axes[1].set_xlabel('Model Scores')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Model Score Distributions Comparison', y=1.02, fontsize=14)
    plt.show()


def plot_fraud_rate_by_score_bins(original_results, synthetic_results):
    """Plot fraud rates by model score bins for both approaches."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (results, title, color) in enumerate([
        (original_results, 'Original Approach', 'blue'),
        (synthetic_results, 'Synthetic Approach', 'green')
    ]):
        data = results['data']
        
        # Create score bins
        score_bins = np.linspace(0, 1, 11)
        bin_centers = (score_bins[:-1] + score_bins[1:]) / 2
        
        # Calculate fraud rate in each bin
        fraud_rates = []
        for j in range(len(score_bins) - 1):
            mask = (data['model_scores'] >= score_bins[j]) & (data['model_scores'] < score_bins[j+1])
            if mask.sum() > 0:
                fraud_rate = data.loc[mask, 'is_fraud'].mean()
            else:
                fraud_rate = 0
            fraud_rates.append(fraud_rate)
        
        axes[i].bar(bin_centers, fraud_rates, width=0.08, alpha=0.7, color=color)
        axes[i].set_title(title)
        axes[i].set_xlabel('Model Score Bins')
        axes[i].set_ylabel('Fraud Rate')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, max(max(fraud_rates), 0.1))
    
    plt.tight_layout()
    plt.suptitle('Fraud Rate by Model Score Bins', y=1.02, fontsize=14)
    plt.show()


def main():
    """Run the complete comparison demo."""
    print("Approach Comparison Demo")
    print("This demo compares the original probabilistic and new synthetic approaches")
    
    # Run comparison
    original_results, synthetic_results = compare_approaches()
    
    # Create visualizations
    print("\n4. VISUALIZATIONS")
    print("-" * 50)
    
    print("Generating score distribution comparison...")
    plot_score_distributions(original_results, synthetic_results)
    
    print("Generating fraud rate by score bins comparison...")
    plot_fraud_rate_by_score_bins(original_results, synthetic_results)
    
    # Key insights
    print("\n5. KEY INSIGHTS")
    print("-" * 50)
    print("✓ Both approaches produce compatible data formats")
    print("✓ Both work seamlessly with LoggingPolicyGenerator")
    print("✓ Synthetic approach provides additional model performance metrics")
    print("✓ Synthetic approach allows testing different ML model types")
    print("✓ Original approach uses pure statistical distributions")
    print("✓ Synthetic approach uses realistic ML model predictions")
    print("✓ Both approaches support the same counterfactual evaluation workflow")
    
    print("\n" + "=" * 80)
    print("Comparison completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main() 