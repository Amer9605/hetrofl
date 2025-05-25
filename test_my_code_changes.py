#!/usr/bin/env python3
"""
Validation test for the specific code changes made to resolve the reported issues.
Tests the changes without requiring GUI libraries.
"""

import sys
import os

def test_css_transform_fix():
    """Test that CSS transform properties have been removed."""
    print("🧪 Testing CSS Transform Fix...")
    
    try:
        with open('gui/gui_styles.py', 'r') as f:
            content = f.read()
        
        if 'transform:' in content:
            print("❌ FAILED: CSS transform properties still present")
            return False
        else:
            print("✅ SUCCESS: CSS transform properties removed")
            return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_gui_file_syntax():
    """Test that GUI files have valid Python syntax."""
    print("\n🧪 Testing GUI File Syntax...")
    
    gui_files = [
        'gui/gui_styles.py',
        'gui/gui_test_interface.py',
        'gui_main.py'
    ]
    
    success = True
    for file_path in gui_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Compile to check syntax
            compile(content, file_path, 'exec')
            print(f"✅ {file_path}: Valid Python syntax")
        except SyntaxError as e:
            print(f"❌ {file_path}: Syntax error - {e}")
            success = False
        except Exception as e:
            print(f"❌ {file_path}: Error - {e}")
            success = False
    
    return success

def test_model_tracker_enhancements():
    """Test that model tracker functionality works."""
    print("\n🧪 Testing Model Tracker Enhancements...")
    
    try:
        from utils.model_persistence import ModelTracker
        
        # Test basic functionality
        tracker = ModelTracker()
        print("✅ ModelTracker initialized successfully")
        
        # Test method existence
        required_methods = [
            'start_experiment',
            'register_round_models',
            'get_round_models',
            'get_experiment_rounds',
            'test_models_across_rounds',
            'analyze_improvement_trends'
        ]
        
        for method in required_methods:
            if hasattr(tracker, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"❌ Method '{method}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_gui_test_interface_enhancements():
    """Test GUI test interface enhancements without creating widgets."""
    print("\n🧪 Testing GUI Test Interface Enhancements...")
    
    try:
        # Check that the file has the required enhancements
        with open('gui/gui_test_interface.py', 'r') as f:
            content = f.read()
        
        required_elements = [
            'load_best_model_from_rounds',
            'ModelTestPanel',
            'RoundTestingWidget',
            'model_tracker = ModelTracker()'
        ]
        
        success = True
        for element in required_elements:
            if element in content:
                print(f"✅ Found: {element}")
            else:
                print(f"❌ Missing: {element}")
                success = False
        
        return success
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_main_gui_enhancements():
    """Test main GUI file enhancements."""
    print("\n🧪 Testing Main GUI Enhancements...")
    
    try:
        with open('gui_main.py', 'r') as f:
            content = f.read()
        
        required_elements = [
            'Global Model Testing',
            'Round-by-Round Analysis',
            'global_test_panel',
            'round_test_widget',
            'client_{self.client_id}_{model_name}'
        ]
        
        success = True
        for element in required_elements:
            if element in content:
                print(f"✅ Found: {element}")
            else:
                print(f"❌ Missing: {element}")
                success = False
        
        return success
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_import_compatibility():
    """Test that all imports work correctly."""
    print("\n🧪 Testing Import Compatibility...")
    
    try:
        # Test core functionality imports (without GUI)
        from utils.model_persistence import ModelTracker
        print("✅ ModelTracker import successful")
        
        from config.config import MODEL_SAVE_DIR, RESULTS_DIR
        print("✅ Config imports successful")
        
        # Test that our validation script works
        import validate_round_tracking_fixes
        print("✅ Validation script import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("🚀 VALIDATION TESTS FOR CODE CHANGES")
    print("=" * 50)
    
    tests = [
        ("CSS Transform Fix", test_css_transform_fix),
        ("GUI File Syntax", test_gui_file_syntax),
        ("Model Tracker Enhancements", test_model_tracker_enhancements),
        ("GUI Test Interface Enhancements", test_gui_test_interface_enhancements),
        ("Main GUI Enhancements", test_main_gui_enhancements),
        ("Import Compatibility", test_import_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Code changes are validated and ready")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())