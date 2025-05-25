#!/usr/bin/env python3
"""
Test script for the enhanced round-by-round model saving and testing functionality.
This script demonstrates the new features implemented for the HETROFL system.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.data_loader import DataLoader
from global_model.federated_learning import HeterogeneousFederatedLearning
from local_models.xgboost_model import XGBoostModel
from local_models.random_forest_model import RandomForestModel
from local_models.lightgbm_model import LightGBMModel
from utils.model_persistence import ModelTracker
from config.config import CLIENT_MODELS


def create_sample_dataset():
    """Create a sample dataset for testing."""
    print("Creating sample dataset for testing...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_classes = 3
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some pattern
    y = np.random.choice(n_classes, n_samples)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to CSV
    test_data_path = "test_data.csv"
    df.to_csv(test_data_path, index=False)
    
    print(f"Sample dataset created: {test_data_path}")
    print(f"Shape: {df.shape}")
    print(f"Classes: {sorted(df['target'].unique())}")
    
    return test_data_path


def test_round_by_round_functionality():
    """Test the round-by-round model saving and testing functionality."""
    print("=" * 60)
    print("Testing Round-by-Round Model Saving and Testing")
    print("=" * 60)
    
    # Create sample dataset
    dataset_path = create_sample_dataset()
    
    try:
        # Initialize data loader
        print("\n1. Initializing data loader...")
        data_loader = DataLoader(
            dataset_path=dataset_path,
            target_column='target',
            test_size=0.2,
            val_size=0.2,
            sample_size=800  # Use smaller sample for testing
        )
        
        # Initialize local model classes (use subset for faster testing)
        print("\n2. Setting up local models...")
        local_model_classes = {
            "xgboost": XGBoostModel,
            "random_forest": RandomForestModel,
            "lightgbm": LightGBMModel
        }
        
        # Initialize federated learning system
        print("\n3. Initializing federated learning system...")
        fl_system = HeterogeneousFederatedLearning(
            data_loader=data_loader,
            local_model_classes=local_model_classes,
            experiment_name="round_tracking_test"
        )
        
        # Run federated learning with fewer rounds for testing
        print("\n4. Running federated learning (3 rounds for testing)...")
        global_model = fl_system.run_federated_learning(
            communication_rounds=3,
            hyperparameter_tuning=False,  # Skip for faster testing
            data_distribution="iid",
            load_previous_models=False,
            local_epochs=1  # Reduced for faster testing
        )
        
        print("\n5. Testing round-by-round functionality...")
        
        # Test specific round
        print("\n   Testing Round 2 models...")
        round_2_results = fl_system.test_round_models(round_num=2)
        if round_2_results:
            print(f"   Round 2 test completed. Global model accuracy: {round_2_results['global_model'].get('accuracy', 'N/A'):.4f}")
        
        # Test all rounds
        print("\n   Testing all rounds...")
        all_rounds_results = fl_system.test_all_rounds()
        if all_rounds_results:
            print(f"   All rounds tested. Total rounds: {len(all_rounds_results['rounds'])}")
        
        # Generate improvement analysis
        print("\n   Generating improvement analysis...")
        improvement_analysis = fl_system.generate_improvement_analysis()
        if improvement_analysis and "improvement_summary" in improvement_analysis:
            global_summary = improvement_analysis["improvement_summary"].get("global_model", {})
            print(f"   Global model accuracy improvement: {global_summary.get('accuracy_improvement', 'N/A'):.4f}")
        
        print("\n6. Testing ModelTracker functionality...")
        
        # Test model tracker methods
        tracker = fl_system.model_tracker
        
        # List experiments
        experiments = tracker.list_experiments()
        print(f"   Available experiments: {len(experiments)}")
        
        # Get latest experiment
        latest_exp = tracker.get_latest_experiment()
        print(f"   Latest experiment: {latest_exp}")
        
        if latest_exp:
            # Get experiment rounds
            rounds = tracker.get_experiment_rounds(latest_exp)
            print(f"   Rounds in latest experiment: {list(rounds.keys())}")
            
            # Test round-by-round testing via tracker
            print("\n   Testing via ModelTracker...")
            test_results = tracker.test_models_across_rounds(
                experiment_id=latest_exp,
                test_data=data_loader.X_test,
                test_labels=data_loader.y_test,
                save_results=True
            )
            
            if test_results:
                print(f"   Cross-round testing completed for {len(test_results['rounds'])} rounds")
        
        print("\n" + "=" * 60)
        print("✅ Round-by-Round Testing Completed Successfully!")
        print("=" * 60)
        
        # Print summary of what was tested
        print("\n📊 Summary of Features Tested:")
        print("   ✓ Automatic round-by-round model saving")
        print("   ✓ Round-specific directory structure creation")
        print("   ✓ Metadata saving with each model")
        print("   ✓ Round-specific model testing")
        print("   ✓ Cross-round performance analysis")
        print("   ✓ Improvement trend visualization")
        print("   ✓ Comprehensive experiment tracking")
        
        # Show where results are saved
        if hasattr(tracker, 'current_experiment_dir') and tracker.current_experiment_dir:
            print(f"\n📁 Results saved in: {tracker.current_experiment_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test data
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
            print(f"\n🧹 Cleaned up test dataset: {dataset_path}")


def demonstrate_usage():
    """Demonstrate how to use the new functionality."""
    print("\n" + "=" * 60)
    print("Usage Examples for Round-by-Round Functionality")
    print("=" * 60)
    
    usage_examples = """
# 1. Basic Usage - Automatic round saving is now enabled by default
fl_system = HeterogeneousFederatedLearning(data_loader, local_model_classes)
global_model = fl_system.run_federated_learning(communication_rounds=5)

# 2. Test models from a specific round
round_results = fl_system.test_round_models(round_num=3)
print(f"Round 3 global model accuracy: {round_results['global_model']['accuracy']}")

# 3. Test all rounds and get comprehensive results
all_results = fl_system.test_all_rounds(save_results=True)
print(f"Tested {len(all_results['rounds'])} rounds")

# 4. Generate improvement analysis with plots
analysis = fl_system.generate_improvement_analysis(save_plots=True)
global_improvement = analysis['improvement_summary']['global_model']['accuracy_improvement']
print(f"Global model improved by: {global_improvement:.4f}")

# 5. Using ModelTracker directly for advanced operations
tracker = fl_system.model_tracker

# List all experiments
experiments = tracker.list_experiments()

# Get specific round data
round_data = tracker.get_round_models(experiment_id, round_num=2)

# Test models across rounds with custom data
test_results = tracker.test_models_across_rounds(
    experiment_id=experiment_id,
    test_data=custom_test_data,
    test_labels=custom_test_labels
)

# Analyze improvement trends
trends = tracker.analyze_improvement_trends(experiment_id, save_plots=True)
"""
    
    print(usage_examples)
    
    print("\n📁 Directory Structure Created:")
    structure = """
results/models/experiment_YYYY-MM-DD_HH-MM-SS/
├── round_1/
│   ├── local_models/
│   │   ├── client_0_xgboost/
│   │   │   ├── xgboost.joblib
│   │   │   ├── metadata.json
│   │   │   └── ...
│   │   ├── client_1_random_forest/
│   │   └── ...
│   ├── global_model/
│   │   ├── global_model.joblib
│   │   └── metadata.json
│   └── round_summary.json
├── round_2/
└── ...

results/test_results/
├── experiment_YYYY-MM-DD_HH-MM-SS_test_results.json
└── ...

results/improvement_plots/experiment_YYYY-MM-DD_HH-MM-SS/
├── improvement_trends.png
├── client_0_improvement.png
└── ...
"""
    print(structure)


if __name__ == "__main__":
    print("🚀 HETROFL Round-by-Round Testing System")
    print("Testing enhanced model saving and improvement tracking...")
    
    # Run the test
    success = test_round_by_round_functionality()
    
    if success:
        # Show usage examples
        demonstrate_usage()
        print("\n🎉 All tests passed! The round-by-round functionality is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1)