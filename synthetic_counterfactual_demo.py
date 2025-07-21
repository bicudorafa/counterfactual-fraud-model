"""Demo script showcasing the new synthetic counterfactual fraud model capabilities."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.counterfactual_fraud_model import (
    SyntheticDataGenerator,
    SyntheticOffPolicyEvaluationPipeline,
    LoggingPolicyGenerator
)


def demo_synthetic_data_generator():
    """Demonstrate the SyntheticDataGenerator functionality."""
    print("=" * 60)
    print("DEMO: Synthetic Data Generator")
    print("=" * 60)
    
    # Create synthetic data generator
    generator = SyntheticDataGenerator(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        model_type="lightgbm",
        random_state=42
    )
    
    # Generate data
    print("Generating synthetic dataset and training model...")
    data = generator.generate_data()
    
    # Display basic info
    print(f"\nDataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nFirst few rows:")
    print(data.head())
    
    # Get dataset info
    dataset_info = generator.get_dataset_info()
    print(f"\nDataset Info:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # Get model performance
    model_performance = generator.get_model_performance()
    print(f"\nModel Performance:")
    for key, value in model_performance.items():
        print(f"  {key}: {value:.4f}")
    
    return data, generator


def demo_logging_policy_with_synthetic_data(data):
    """Demonstrate LoggingPolicyGenerator with synthetic data."""
    print("\n" + "=" * 60)
    print("DEMO: Logging Policy with Synthetic Data")
    print("=" * 60)
    
    # Create logging policy generator
    policy_generator = LoggingPolicyGenerator(
        cutoff=0.05,
        exploration_rate=0.1,
        random_state=42
    )
    
    # Apply policy
    print("Applying logging policy...")
    policy_data = policy_generator.generate_policy(data.copy())
    
    print(f"\nPolicy data shape: {policy_data.shape}")
    print(f"Columns: {list(policy_data.columns)}")
    
    # Show action distribution
    print(f"\nModel Action Distribution:")
    print(policy_data['model_action'].value_counts(normalize=True))
    
    print(f"\nPolicy Action Distribution:")
    print(policy_data['policy_action'].value_counts(normalize=True))
    
    # Show propensity score statistics
    allowed_data = policy_data[policy_data['policy_action'] == 'allow']
    print(f"\nPropensity Score Statistics (for allowed transactions):")
    print(f"  Mean: {allowed_data['propensity_score'].mean():.4f}")
    print(f"  Min: {allowed_data['propensity_score'].min():.4f}")
    print(f"  Max: {allowed_data['propensity_score'].max():.4f}")
    
    return policy_data


def demo_synthetic_pipeline():
    """Demonstrate the complete SyntheticOffPolicyEvaluationPipeline."""
    print("\n" + "=" * 60)
    print("DEMO: Synthetic Off-Policy Evaluation Pipeline")
    print("=" * 60)
    
    # Create pipeline
    pipeline = SyntheticOffPolicyEvaluationPipeline(
        n_samples=20000,
        n_features=25,
        n_informative=12,
        model_type="lightgbm",
        model_params={'n_estimators': 100, 'learning_rate': 0.1},
        random_state=42
    )
    
    # Run pipeline with different cutoffs
    cutoffs = [0.03, 0.05, 0.1]
    exploration_rate = 0.05
    
    results = {}
    
    for cutoff in cutoffs:
        print(f"\nRunning pipeline with cutoff={cutoff}, exploration_rate={exploration_rate}")
        result = pipeline.run_pipeline(
            cutoff=cutoff,
            exploration_rate=exploration_rate,
            include_data=False  # Don't include data to save memory
        )
        results[cutoff] = result
        
        # Display key metrics
        stats = result['statistics']
        print(f"  Original approval rate: {stats['original_approval_rate']:.4f}")
        print(f"  Policy approval rate: {stats['policy_approval_rate']:.4f}")
        print(f"  Original fraud rate: {stats['original_fraud_rate']:.4f}")
        print(f"  Policy fraud rate: {stats['policy_fraud_rate']:.4f}")
        print(f"  Approval rate increase: {stats['approval_rate_increase']:.4f}")
    
    # Show model performance
    model_perf = pipeline.get_model_performance()
    print(f"\nModel Performance:")
    for key, value in model_perf.items():
        print(f"  {key}: {value:.4f}")
    
    # Show dataset info
    dataset_info = pipeline.get_dataset_info()
    print(f"\nDataset Info:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    return results


def demo_model_comparison():
    """Demonstrate comparison between different models."""
    print("\n" + "=" * 60)
    print("DEMO: Model Comparison")
    print("=" * 60)
    
    models = ["lightgbm", "random_forest", "logistic"]
    results = {}
    
    for model_type in models:
        print(f"\nTesting {model_type} model...")
        
        pipeline = SyntheticOffPolicyEvaluationPipeline(
            n_samples=15000,
            n_features=20,
            n_informative=10,
            model_type=model_type,
            random_state=42
        )
        
        result = pipeline.run_pipeline(
            cutoff=0.05,
            exploration_rate=0.05,
            include_data=False
        )
        
        results[model_type] = result
        
        # Display model performance
        model_perf = result['model_performance']
        print(f"  ROC AUC: {model_perf['roc_auc']:.4f}")
        print(f"  Precision: {model_perf['precision']:.4f}")
        print(f"  Recall: {model_perf['recall']:.4f}")
        print(f"  F1 Score: {model_perf['f1_score']:.4f}")
        
        # Display policy metrics
        stats = result['statistics']
        print(f"  Policy approval rate: {stats['policy_approval_rate']:.4f}")
        print(f"  Policy fraud rate: {stats['policy_fraud_rate']:.4f}")
    
    return results


def create_comparison_plots(results):
    """Create comparison plots for different models or cutoffs."""
    print("\n" + "=" * 60)
    print("Creating comparison plots...")
    print("=" * 60)
    
    # Extract metrics for plotting
    metrics_data = []
    for key, result in results.items():
        if 'model_performance' in result:
            # Model comparison data
            model_perf = result['model_performance']
            stats = result['statistics']
            metrics_data.append({
                'Model': key,
                'ROC_AUC': model_perf['roc_auc'],
                'Precision': model_perf['precision'],
                'Recall': model_perf['recall'],
                'F1_Score': model_perf['f1_score'],
                'Policy_Approval_Rate': stats['policy_approval_rate'],
                'Policy_Fraud_Rate': stats['policy_fraud_rate']
            })
        else:
            # Cutoff comparison data
            stats = result['statistics']
            metrics_data.append({
                'Cutoff': key,
                'Policy_Approval_Rate': stats['policy_approval_rate'],
                'Policy_Fraud_Rate': stats['policy_fraud_rate'],
                'Approval_Rate_Increase': stats['approval_rate_increase']
            })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    if 'Model' in df_metrics.columns:
        # Model comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ROC AUC comparison
        axes[0, 0].bar(df_metrics['Model'], df_metrics['ROC_AUC'])
        axes[0, 0].set_title('ROC AUC by Model')
        axes[0, 0].set_ylabel('ROC AUC')
        
        # Precision vs Recall
        axes[0, 1].scatter(df_metrics['Recall'], df_metrics['Precision'])
        for i, model in enumerate(df_metrics['Model']):
            axes[0, 1].annotate(model, (df_metrics['Recall'].iloc[i], df_metrics['Precision'].iloc[i]))
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall')
        
        # Policy approval rate
        axes[1, 0].bar(df_metrics['Model'], df_metrics['Policy_Approval_Rate'])
        axes[1, 0].set_title('Policy Approval Rate by Model')
        axes[1, 0].set_ylabel('Approval Rate')
        
        # Policy fraud rate
        axes[1, 1].bar(df_metrics['Model'], df_metrics['Policy_Fraud_Rate'])
        axes[1, 1].set_title('Policy Fraud Rate by Model')
        axes[1, 1].set_ylabel('Fraud Rate')
        
        plt.tight_layout()
        plt.suptitle('Model Comparison Results', y=1.02)
        plt.show()
    
    elif 'Cutoff' in df_metrics.columns:
        # Cutoff comparison plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Approval rate vs Cutoff
        axes[0].plot(df_metrics['Cutoff'], df_metrics['Policy_Approval_Rate'], 'o-')
        axes[0].set_xlabel('Cutoff')
        axes[0].set_ylabel('Policy Approval Rate')
        axes[0].set_title('Approval Rate vs Cutoff')
        
        # Fraud rate vs Cutoff
        axes[1].plot(df_metrics['Cutoff'], df_metrics['Policy_Fraud_Rate'], 'o-')
        axes[1].set_xlabel('Cutoff')
        axes[1].set_ylabel('Policy Fraud Rate')
        axes[1].set_title('Fraud Rate vs Cutoff')
        
        plt.tight_layout()
        plt.show()


def main():
    """Run the complete demo."""
    print("Synthetic Counterfactual Fraud Model Demo")
    print("This demo showcases the new synthetic dataset capabilities")
    
    # Demo 1: Basic synthetic data generation
    data, generator = demo_synthetic_data_generator()
    
    # Demo 2: Logging policy with synthetic data
    policy_data = demo_logging_policy_with_synthetic_data(data)
    
    # Demo 3: Complete pipeline
    pipeline_results = demo_synthetic_pipeline()
    
    # Demo 4: Model comparison
    model_results = demo_model_comparison()
    
    # Create visualization plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(model_results)
    create_comparison_plots(pipeline_results)
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main() 