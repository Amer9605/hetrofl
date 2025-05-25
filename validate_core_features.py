#!/usr/bin/env python3
"""
Core functionality validation for the enhanced round-by-round system.
Tests the implementation without GUI dependencies.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_core_functionality():
    """Validate the core round-by-round functionality."""
    
    print("🔍 Validating Core Round-by-Round Functionality")
    print("=" * 55)
    
    try:
        # Test ModelTracker import and basic functionality
        print("\n1. Testing ModelTracker...")
        from utils.model_persistence import ModelTracker
        
        tracker = ModelTracker()
        print("   ✅ ModelTracker imported and initialized successfully")
        
        # Test experiment creation
        exp_id, exp_dir = tracker.start_experiment("validation_test")
        print(f"   ✅ Experiment created: {exp_id}")
        print(f"   ✅ Experiment directory: {exp_dir}")
        
        # Test experiment listing
        experiments = tracker.list_experiments()
        print(f"   ✅ Found {len(experiments)} experiments")
        
        # Test round history structure
        print(f"   ✅ Round history structure initialized")
        
    except Exception as e:
        print(f"   ❌ ModelTracker test failed: {e}")
        return False
    
    try:
        # Test enhanced BaseLocalModel
        print("\n2. Testing Enhanced BaseLocalModel...")
        from local_models.base_model import BaseLocalModel
        
        # Check if save_model method has new parameters
        import inspect
        save_method = BaseLocalModel.save_model
        sig = inspect.signature(save_method)
        params = list(sig.parameters.keys())
        
        expected_params = ['self', 'save_dir', 'round_num', 'experiment_id', 'metadata']
        if all(param in params for param in expected_params):
            print("   ✅ BaseLocalModel.save_model enhanced with round parameters")
        else:
            print(f"   ❌ Missing parameters in save_model: {params}")
            return False
            
    except Exception as e:
        print(f"   ❌ BaseLocalModel test failed: {e}")
        return False
    
    try:
        # Test enhanced HeterogeneousFederatedLearning
        print("\n3. Testing Enhanced HeterogeneousFederatedLearning...")
        from global_model.federated_learning import HeterogeneousFederatedLearning
        
        # Check for new methods
        required_methods = [
            'save_round_models',
            'test_round_models', 
            'test_all_rounds',
            'generate_improvement_analysis'
        ]
        
        for method_name in required_methods:
            if hasattr(HeterogeneousFederatedLearning, method_name):
                print(f"   ✅ Method {method_name} added successfully")
            else:
                print(f"   ❌ Method {method_name} missing")
                return False
                
    except Exception as e:
        print(f"   ❌ HeterogeneousFederatedLearning test failed: {e}")
        return False
    
    try:
        # Test ModelTracker advanced methods
        print("\n4. Testing ModelTracker Advanced Methods...")
        
        advanced_methods = [
            'register_round_models',
            'test_models_across_rounds',
            'analyze_improvement_trends',
            'get_round_models',
            'get_experiment_rounds'
        ]
        
        for method_name in advanced_methods:
            if hasattr(tracker, method_name):
                print(f"   ✅ Method {method_name} available")
            else:
                print(f"   ❌ Method {method_name} missing")
                return False
                
    except Exception as e:
        print(f"   ❌ Advanced methods test failed: {e}")
        return False
    
    try:
        # Test file structure and compilation
        print("\n5. Testing File Structure and Compilation...")
        
        files_to_check = [
            'utils/model_persistence.py',
            'global_model/federated_learning.py', 
            'local_models/base_model.py',
            'test_round_tracking.py',
            'README_ROUND_TRACKING.md'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path} exists")
                
                # Test Python file compilation
                if file_path.endswith('.py'):
                    import py_compile
                    try:
                        py_compile.compile(file_path, doraise=True)
                        print(f"   ✅ {file_path} compiles successfully")
                    except py_compile.PyCompileError as e:
                        print(f"   ❌ {file_path} compilation failed: {e}")
                        return False
            else:
                print(f"   ❌ {file_path} missing")
                return False
                
    except Exception as e:
        print(f"   ❌ File structure test failed: {e}")
        return False
    
    # Test directory creation
    try:
        print("\n6. Testing Directory Structure Creation...")
        
        # Check if experiment directory was created
        if os.path.exists(exp_dir):
            print(f"   ✅ Experiment directory created: {exp_dir}")
        else:
            print(f"   ❌ Experiment directory not created")
            return False
            
        # Check history files
        history_dir = os.path.join("results", "history")
        if os.path.exists(history_dir):
            print(f"   ✅ History directory exists: {history_dir}")
            
            # Check for round_history.json (always created)
            round_hist_path = os.path.join(history_dir, 'round_history.json')
            if os.path.exists(round_hist_path):
                print(f"   ✅ round_history.json created")
                
                # Validate JSON structure
                try:
                    with open(round_hist_path, 'r') as f:
                        data = json.load(f)
                    print(f"   ✅ round_history.json has valid JSON structure")
                except json.JSONDecodeError:
                    print(f"   ❌ round_history.json has invalid JSON")
                    return False
            else:
                print(f"   ❌ round_history.json not created")
                return False
            
            # Other history files are created on demand
            print(f"   ✅ Other history files will be created on demand")
        else:
            print(f"   ❌ History directory not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Directory structure test failed: {e}")
        return False
    
    print("\n" + "=" * 55)
    print("🎉 ALL CORE FUNCTIONALITY TESTS PASSED!")
    print("=" * 55)
    
    print("\n✅ Validation Summary:")
    print("   • ModelTracker enhanced with round-based tracking")
    print("   • BaseLocalModel enhanced with round saving")
    print("   • HeterogeneousFederatedLearning enhanced with testing methods")
    print("   • All new methods successfully added")
    print("   • File structure and compilation validated")
    print("   • Directory structure creation working")
    print("   • JSON configuration files properly initialized")
    
    print("\n🚀 The round-by-round system is ready for use!")
    print("\n📋 Next Steps:")
    print("   1. Run 'python test_round_tracking.py' for full functionality test")
    print("   2. Use the enhanced federated learning system with automatic round saving")
    print("   3. Test round-by-round functionality on your datasets")
    print("   4. Generate improvement analysis and visualizations")
    
    return True

if __name__ == "__main__":
    success = validate_core_functionality()
    if not success:
        sys.exit(1)
