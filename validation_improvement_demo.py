"""
Validation Improvement Demo: From ValidationUtils to Pure Pydantic

This demo shows how we improved the validation approach by:
1. Eliminating ValidationUtils redundancy 
2. Using Pydantic consistently for ALL validation
3. Better error messages and type safety
4. JSON serialization for execution parameters too
"""

import sys
from pathlib import Path
from pydantic import ValidationError

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from counterfactual_fraud_model import (
    ProbabilisticPipelineConfig,
    SyntheticPipelineConfig,
    PipelineExecutionConfig,
    PipelineFactory
)


def demonstrate_old_vs_new_approach():
    """Show the difference between old ValidationUtils and new Pydantic approach."""
    print("=" * 60)
    print("VALIDATION IMPROVEMENT: ValidationUtils → Pure Pydantic")
    print("=" * 60)
    
    print("\n❌ OLD APPROACH (ValidationUtils):")
    print("  - Separate validation functions for different parameter types")
    print("  - Duplicated validation logic between pipeline classes")
    print("  - Runtime parameters validated separately from config parameters")
    print("  - Inconsistent error handling")
    print("  - No type coercion or JSON serialization")
    
    print("\n✅ NEW APPROACH (Pure Pydantic):")
    print("  - Consistent Pydantic validation for ALL parameters")
    print("  - Configuration parameters + execution parameters")
    print("  - Automatic type coercion and JSON serialization")
    print("  - Clear, descriptive error messages")
    print("  - Single validation system throughout the codebase")


def demonstrate_execution_config_validation():
    """Show how execution parameters are now validated with Pydantic."""
    print("\n" + "=" * 60)
    print("EXECUTION PARAMETER VALIDATION")
    print("=" * 60)
    
    print("\n✅ Valid execution parameters:")
    valid_exec = PipelineExecutionConfig(
        cutoff=0.05,
        exploration_rate=0.1,
        include_data=True
    )
    print(f"  - Cutoff: {valid_exec.cutoff}")
    print(f"  - Exploration rate: {valid_exec.exploration_rate}")
    print(f"  - Include data: {valid_exec.include_data}")
    
    print("\n✅ Type coercion works:")
    coerced_exec = PipelineExecutionConfig(
        cutoff="0.03",  # String → float
        exploration_rate="0.08",  # String → float
        include_data="true"  # String → bool
    )
    print(f"  - Cutoff coerced: {coerced_exec.cutoff} ({type(coerced_exec.cutoff).__name__})")
    print(f"  - Exploration rate coerced: {coerced_exec.exploration_rate} ({type(coerced_exec.exploration_rate).__name__})")
    print(f"  - Include data coerced: {coerced_exec.include_data} ({type(coerced_exec.include_data).__name__})")
    
    print("\n❌ Invalid parameters caught:")
    try:
        invalid_exec = PipelineExecutionConfig(
            cutoff=1.5,  # > 1.0
            exploration_rate=-0.1  # < 0.0
        )
    except ValidationError as e:
        print(f"  - Validation error: {str(e)[:100]}...")
    
    print("\n✅ JSON serialization for execution config:")
    exec_json = valid_exec.model_dump_json(indent=2)
    print("  - Execution config as JSON:")
    print("    " + exec_json.replace("\n", "\n    "))


def demonstrate_unified_pipeline_validation():
    """Show how pipelines now use consistent Pydantic validation."""
    print("\n" + "=" * 60)
    print("UNIFIED PIPELINE VALIDATION")
    print("=" * 60)
    
    print("\n✅ Creating pipelines with Pydantic configs:")
    
    # Configuration validation
    prob_config = ProbabilisticPipelineConfig(
        alpha=0.2,
        sample_size=5000,
        random_state=42
    )
    
    pipeline = PipelineFactory.create_probabilistic_pipeline(
        alpha=prob_config.alpha,
        sample_size=prob_config.sample_size,
        random_state=prob_config.random_state
    )
    
    print(f"  - Created {type(pipeline).__name__}")
    print(f"  - Config validation: ✅ Passed")
    
    print("\n✅ Execution parameter validation happens inside run_pipeline:")
    print("  - No need for separate ValidationUtils calls")
    print("  - Pydantic validates cutoff, exploration_rate automatically")
    print("  - Consistent error handling throughout")
    
    # This will work - valid parameters
    try:
        result = pipeline.run_pipeline(cutoff=0.05, exploration_rate=0.03, include_data=False)
        print(f"  - Pipeline executed successfully with {result['statistics']['total_transactions']} transactions")
    except Exception as e:
        print(f"  - Execution error: {e}")
    
    # This will fail - invalid parameters
    print("\n❌ Invalid execution parameters caught automatically:")
    try:
        result = pipeline.run_pipeline(cutoff=1.5, exploration_rate=-0.1)
    except ValidationError as e:
        print(f"  - Pydantic validation error: {str(e)[:150]}...")


def demonstrate_benefits():
    """Show the benefits of the improved approach."""
    print("\n" + "=" * 60)
    print("BENEFITS OF PURE PYDANTIC APPROACH")
    print("=" * 60)
    
    print("\n🎯 Code Quality Improvements:")
    print("  ✅ Eliminated ValidationUtils redundancy")
    print("  ✅ Consistent validation approach throughout codebase")
    print("  ✅ Better separation of concerns (no mixed validation)")
    print("  ✅ Reduced code duplication")
    
    print("\n🔧 Technical Benefits:")
    print("  ✅ Automatic type coercion for all parameters")
    print("  ✅ JSON serialization for execution parameters")
    print("  ✅ Better error messages with field-specific context")
    print("  ✅ Schema generation for API documentation")
    
    print("\n🚀 Maintainability Benefits:")
    print("  ✅ Single source of truth for validation logic")
    print("  ✅ Easy to add new validation rules")
    print("  ✅ Consistent error handling patterns")
    print("  ✅ Better testability through unified approach")
    
    print("\n📊 User Experience Benefits:")
    print("  ✅ Clear, descriptive error messages")
    print("  ✅ Automatic parameter conversion")
    print("  ✅ Configuration sharing via JSON")
    print("  ✅ Better IDE support with type hints")


def main():
    """Run the complete validation improvement demonstration."""
    print("VALIDATION IMPROVEMENT DEMONSTRATION")
    print("Transitioning from ValidationUtils to Pure Pydantic")
    print("\nThis demo shows how we eliminated the ValidationUtils redundancy")
    print("and moved to a consistent Pydantic-based validation approach.")
    
    demonstrate_old_vs_new_approach()
    demonstrate_execution_config_validation()
    demonstrate_unified_pipeline_validation()
    demonstrate_benefits()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n🎯 You were absolutely right to question ValidationUtils!")
    print("\n✅ Improvements Made:")
    print("  - Removed redundant ValidationUtils class")
    print("  - Created PipelineExecutionConfig for runtime parameters")
    print("  - Updated base pipeline to use Pydantic validation")
    print("  - Consistent validation approach throughout codebase")
    print("  - Better error messages and type safety")
    
    print("\n🚀 Result:")
    print("  - Pure Pydantic validation for ALL parameters")
    print("  - No more mixed validation approaches")
    print("  - Better maintainability and consistency")
    print("  - Enhanced user experience with better errors")
    
    print(f"\n✨ Validation improvement demonstration completed!")


if __name__ == "__main__":
    main() 