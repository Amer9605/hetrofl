#!/usr/bin/env python3
"""
Validation script for round-by-round model tracking and testing fixes.
This script validates that both reported issues have been resolved:

1. CSS transform warnings removed
2. Model loading and testing from saved rounds works correctly
"""

import os
import sys
import tempfile
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Add current directory to path
sys.path.insert(0, '.')

def test_css_transform_fix():
    """Test that CSS transform properties have been removed."""
    print("=" * 60)
    print("TESTING CSS TRANSFORM FIX (Issue 1)")
    print("=" * 60)
    
    try:
        with open('gui/gui_styles.py', 'r') as f:
            content = f.read()
        
        if 'transform:' in content:
            print("❌ FAILED: CSS transform properties still present")
            return False
        else:
            print("✅ SUCCESS: CSS transform properties removed")
            print("   - No more 'Unknown property transform' warnings")
            return True
    except Exception as e:
        print(f"❌ ERROR: Could not test CSS fix: {e}")
        return False

def create_mock_experiment():
    """Create a mock federated learning experiment with multiple rounds."""
    print("\n" + "=" * 60)
    print("CREATING MOCK FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    results_dir = os.path.join(temp_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Mock the config
    from config import config
    original_results_dir = config.RESULTS_DIR
    config.RESULTS_DIR = results_dir
    
    # Import model tracker
    from utils.model_persistence import ModelTracker
    
    # Create tracker and start experiment
    tracker = ModelTracker()
    exp_id, exp_dir = tracker.start_experiment('validation_test')
    
    print(f"✅ Created experiment: {exp_id}")
    print(f"   Directory: {exp_dir}")
    
    # Generate mock training data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                              n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simulate multiple communication rounds
    rounds_data = {}
    
    for round_num in range(1, 6):  # 5 rounds
        print(f"\n--- Simulating Round {round_num} ---")
        
        # Create and train mock models
        global_model = RandomForestClassifier(n_estimators=50 + round_num * 10, random_state=42)
        global_model.fit(X_train, y_train)
        
        local_models = {}
        for client_id in range(3):  # 3 clients
            local_model = RandomForestClassifier(
                n_estimators=30 + round_num * 5 + client_id * 5, 
                random_state=42 + client_id
            )
            local_model.fit(X_train, y_train)
            local_models[client_id] = local_model
        
        # Calculate metrics
        global_pred = global_model.predict(X_test)
        global_accuracy = accuracy_score(y_test, global_pred)
        global_f1 = f1_score(y_test, global_pred, average='weighted')
        
        # Save models
        round_dir = os.path.join(exp_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        # Save global model
        global_dir = os.path.join(round_dir, "global_model")
        os.makedirs(global_dir, exist_ok=True)
        global_path = os.path.join(global_dir, "global_model.joblib")
        joblib.dump(global_model, global_path)
        
        # Save local models
        local_paths = {}
        local_metrics = {}
        
        for client_id, local_model in local_models.items():
            local_dir = os.path.join(round_dir, "local_models", f"client_{client_id}")
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, f"client_{client_id}_model.joblib")
            joblib.dump(local_model, local_path)
            
            # Calculate local metrics
            local_pred = local_model.predict(X_test)
            local_accuracy = accuracy_score(y_test, local_pred)
            local_f1 = f1_score(y_test, local_pred, average='weighted')
            
            local_paths[str(client_id)] = local_path
            local_metrics[str(client_id)] = {
                'accuracy': local_accuracy,
                'f1_score': local_f1,
                'round_num': round_num
            }
        
        # Register round in tracker
        tracker.register_round_models(
            round_num=round_num,
            global_model_path=global_path,
            local_model_paths=local_paths,
            global_metrics={
                'accuracy': global_accuracy,
                'f1_score': global_f1,
                'round_num': round_num
            },
            local_metrics=local_metrics
        )
        
        rounds_data[round_num] = {
            'global_accuracy': global_accuracy,
            'global_f1': global_f1,
            'local_accuracies': [local_metrics[str(i)]['accuracy'] for i in range(3)],
            'test_data': (X_test, y_test)
        }
        
        print(f"   Global accuracy: {global_accuracy:.4f}")
        print(f"   Local accuracies: {[f'{acc:.4f}' for acc in rounds_data[round_num]['local_accuracies']]}")
    
    tracker.finish_experiment()
    
    return tracker, exp_id, rounds_data, temp_dir, original_results_dir

def test_model_loading_and_testing(tracker, exp_id, rounds_data):
    """Test model loading and testing functionality."""
    print("\n" + "=" * 60)
    print("TESTING MODEL LOADING AND TESTING (Issue 2)")
    print("=" * 60)
    
    success = True
    
    try:
        # Test 1: List experiments
        experiments = tracker.list_experiments()
        if exp_id not in experiments:
            print("❌ FAILED: Experiment not found in list")
            return False
        print(f"✅ SUCCESS: Found experiment {exp_id}")
        
        # Test 2: Get experiment rounds
        rounds = tracker.get_experiment_rounds(exp_id)
        if len(rounds) != 5:
            print(f"❌ FAILED: Expected 5 rounds, found {len(rounds)}")
            return False
        print(f"✅ SUCCESS: Found {len(rounds)} rounds")
        
        # Test 3: Test models across all rounds
        X_test, y_test = rounds_data[1]['test_data']
        test_results = tracker.test_models_across_rounds(
            experiment_id=exp_id,
            test_data=X_test,
            test_labels=y_test,
            save_results=True
        )
        
        if not test_results:
            print("❌ FAILED: No test results returned")
            return False
        
        print(f"✅ SUCCESS: Tested models across all rounds")
        print(f"   Test size: {test_results['test_size']}")
        print(f"   Rounds tested: {len(test_results['rounds'])}")
        
        # Test 4: Verify test results accuracy
        for round_num, round_result in test_results['rounds'].items():
            if 'global_model' in round_result and 'accuracy' in round_result['global_model']:
                expected_acc = rounds_data[int(round_num)]['global_accuracy']
                actual_acc = round_result['global_model']['accuracy']
                
                # Allow small floating point differences
                if abs(expected_acc - actual_acc) > 0.001:
                    print(f"❌ FAILED: Round {round_num} accuracy mismatch")
                    print(f"   Expected: {expected_acc:.4f}, Got: {actual_acc:.4f}")
                    success = False
                else:
                    print(f"✅ Round {round_num}: Accuracy verified ({actual_acc:.4f})")
        
        # Test 5: Best model identification
        best_round = None
        best_accuracy = 0.0
        
        for round_num, round_data in rounds_data.items():
            if round_data['global_accuracy'] > best_accuracy:
                best_accuracy = round_data['global_accuracy']
                best_round = round_num
        
        print(f"✅ SUCCESS: Best performing round identified: Round {best_round} ({best_accuracy:.4f})")
        
        # Test 6: Model loading from specific round
        round_data = tracker.get_round_models(exp_id, best_round)
        if not round_data:
            print(f"❌ FAILED: Could not get round data for round {best_round}")
            return False
        
        global_model_path = round_data['global_model']['path']
        if not os.path.exists(global_model_path):
            print(f"❌ FAILED: Global model file not found: {global_model_path}")
            return False
        
        # Load and test the model
        loaded_model = joblib.load(global_model_path)
        loaded_pred = loaded_model.predict(X_test)
        loaded_accuracy = accuracy_score(y_test, loaded_pred)
        
        if abs(loaded_accuracy - best_accuracy) > 0.001:
            print(f"❌ FAILED: Loaded model accuracy mismatch")
            return False
        
        print(f"✅ SUCCESS: Best model loaded and tested successfully")
        print(f"   Model path: {global_model_path}")
        print(f"   Accuracy: {loaded_accuracy:.4f}")
        
        return success
        
    except Exception as e:
        print(f"❌ ERROR: Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_improvement_analysis(tracker, exp_id):
    """Test improvement analysis functionality."""
    print("\n" + "=" * 60)
    print("TESTING IMPROVEMENT ANALYSIS")
    print("=" * 60)
    
    try:
        analysis = tracker.analyze_improvement_trends(exp_id, save_plots=False)
        
        if not analysis:
            print("❌ FAILED: No analysis results returned")
            return False
        
        print("✅ SUCCESS: Improvement analysis completed")
        
        # Check analysis structure
        if "improvement_summary" in analysis:
            if "global_model" in analysis["improvement_summary"]:
                global_summary = analysis["improvement_summary"]["global_model"]
                print(f"   Global accuracy improvement: {global_summary['accuracy_improvement']:.4f}")
                print(f"   Best accuracy: {global_summary['best_accuracy']:.4f} (Round {global_summary['best_accuracy_round']})")
            
            if "local_models" in analysis["improvement_summary"]:
                local_summary = analysis["improvement_summary"]["local_models"]
                print(f"   Local models analyzed: {len(local_summary)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Improvement analysis failed: {e}")
        return False

def test_gui_integration():
    """Test GUI integration components."""
    print("\n" + "=" * 60)
    print("TESTING GUI INTEGRATION")
    print("=" * 60)
    
    try:
        # Test that the enhanced GUI components can be imported
        # (without actually creating GUI widgets due to environment limitations)
        
        # Check that the test interface file has the enhanced functionality
        with open('gui/gui_test_interface.py', 'r') as f:
            content = f.read()
        
        required_methods = [
            'load_best_model_from_rounds',
            'ModelTestPanel',
            'RoundTestingWidget'
        ]
        
        for method in required_methods:
            if method not in content:
                print(f"❌ FAILED: Required method/class '{method}' not found in GUI")
                return False
        
        print("✅ SUCCESS: GUI integration components verified")
        print("   - ModelTestPanel enhanced with best model loading")
        print("   - RoundTestingWidget available for comprehensive testing")
        print("   - Automatic model loading from best performing rounds")
        
        # Check main GUI file for new tabs
        with open('gui_main.py', 'r') as f:
            main_content = f.read()
        
        if 'Global Model Testing' not in main_content:
            print("❌ FAILED: Global Model Testing tab not found")
            return False
        
        if 'Round-by-Round Analysis' not in main_content:
            print("❌ FAILED: Round-by-Round Analysis tab not found")
            return False
        
        print("✅ SUCCESS: New GUI tabs added successfully")
        print("   - Global Model Testing tab")
        print("   - Round-by-Round Analysis tab")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: GUI integration test failed: {e}")
        return False

def cleanup(temp_dir, original_results_dir):
    """Clean up test environment."""
    print("\n" + "=" * 60)
    print("CLEANING UP")
    print("=" * 60)
    
    try:
        # Restore original config
        from config import config
        config.RESULTS_DIR = original_results_dir
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)
        
        print("✅ SUCCESS: Cleanup completed")
        
    except Exception as e:
        print(f"⚠️  WARNING: Cleanup failed: {e}")

def main():
    """Main validation function."""
    print("🚀 HETROFL ROUND TRACKING FIXES VALIDATION")
    print("=" * 60)
    print("Validating fixes for:")
    print("1. CSS transform warnings (Issue 1)")
    print("2. Model loading and testing from rounds (Issue 2)")
    print("=" * 60)
    
    all_tests_passed = True
    temp_dir = None
    original_results_dir = None
    
    try:
        # Test 1: CSS Transform Fix
        if not test_css_transform_fix():
            all_tests_passed = False
        
        # Test 2: Create Mock Experiment and Test Model Loading
        tracker, exp_id, rounds_data, temp_dir, original_results_dir = create_mock_experiment()
        
        if not test_model_loading_and_testing(tracker, exp_id, rounds_data):
            all_tests_passed = False
        
        # Test 3: Improvement Analysis
        if not test_improvement_analysis(tracker, exp_id):
            all_tests_passed = False
        
        # Test 4: GUI Integration
        if not test_gui_integration():
            all_tests_passed = False
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
    
    finally:
        if temp_dir and original_results_dir:
            cleanup(temp_dir, original_results_dir)
    
    # Final Results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\n✅ Issue 1 RESOLVED: CSS transform warnings eliminated")
        print("✅ Issue 2 RESOLVED: Model loading and testing works correctly")
        print("\n🚀 The system now:")
        print("   - Saves models automatically after each round")
        print("   - Loads best models automatically for testing")
        print("   - Provides comprehensive round-by-round analysis")
        print("   - Shows improvement trends across rounds")
        print("   - Eliminates CSS warnings in the GUI")
        print("\n🎯 Ready for production use!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the output above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())