#!/usr/bin/env python3
"""
Test core federated learning functionality affected by my changes.
"""

import sys
import os
import tempfile
import numpy as np
from sklearn.datasets import make_classification

# Add current directory to path
sys.path.insert(0, '.')

def test_federated_learning_initialization():
    """Test that federated learning system can be initialized."""
    print("🧪 Testing Federated Learning Initialization...")
    
    try:
        from data.data_loader import DataLoader
        from global_model.federated_learning import HeterogeneousFederatedLearning
        from local_models.xgboost_model import XGBoostModel
        from local_models.random_forest_model import RandomForestModel
        
        # Create mock data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        
        # Create temporary file
        import pandas as pd
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame(X)
        df['target'] = y
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        # Test data loader
        data_loader = DataLoader(temp_file.name)
        print("✅ DataLoader created successfully")
        
        # Test model classes
        local_model_classes = {
            'xgboost': XGBoostModel,
            'random_forest': RandomForestModel
        }
        
        # Test FL system initialization
        fl_system = HeterogeneousFederatedLearning(
            data_loader=data_loader,
            local_model_classes=local_model_classes,
            experiment_name="test_validation"
        )
        print("✅ HeterogeneousFederatedLearning created successfully")
        
        # Clean up
        os.unlink(temp_file.name)
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_tracker_round_functionality():
    """Test model tracker round functionality that my changes use."""
    print("\n🧪 Testing Model Tracker Round Functionality...")
    
    try:
        # Test the specific functionality my changes depend on
        from utils.model_persistence import ModelTracker
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Mock config
        from config import config
        original_results_dir = config.RESULTS_DIR
        config.RESULTS_DIR = temp_dir
        
        tracker = ModelTracker()
        
        # Test experiment creation
        exp_id, exp_dir = tracker.start_experiment("validation_test")
        print(f"✅ Experiment created: {exp_id}")
        
        # Test round registration
        tracker.register_round_models(
            round_num=1,
            global_model_path="/mock/path",
            local_model_paths={"0": "/mock/local"},
            global_metrics={"accuracy": 0.8},
            local_metrics={"0": {"accuracy": 0.75}}
        )
        print("✅ Round registration successful")
        
        # Test round retrieval
        rounds = tracker.get_experiment_rounds(exp_id)
        print(f"✅ Round retrieval successful: {len(rounds)} rounds")
        
        # Test latest experiment
        latest = tracker.get_latest_experiment()
        print(f"✅ Latest experiment: {latest}")
        
        # Restore config
        config.RESULTS_DIR = original_results_dir
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_access():
    """Test that configuration can be accessed."""
    print("\n🧪 Testing Configuration Access...")
    
    try:
        from config.config import CLIENT_MODELS, DATA_DISTRIBUTIONS, MODEL_SAVE_DIR
        print(f"✅ CLIENT_MODELS: {CLIENT_MODELS}")
        print(f"✅ DATA_DISTRIBUTIONS: {DATA_DISTRIBUTIONS}")
        print(f"✅ MODEL_SAVE_DIR: {MODEL_SAVE_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run core functionality tests."""
    print("🚀 CORE FUNCTIONALITY VALIDATION")
    print("=" * 50)
    print("Testing that my changes don't break core federated learning")
    print()
    
    tests = [
        test_config_access,
        test_model_tracker_round_functionality,
        test_federated_learning_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"📊 RESULTS: {passed}/{total} core tests passed")
    
    if passed == total:
        print("🎉 CORE FUNCTIONALITY INTACT!")
        print("✅ My changes don't break existing functionality")
        return 0
    else:
        print("❌ Some core functionality may be affected")
        return 1

if __name__ == "__main__":
    exit(main())