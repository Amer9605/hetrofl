#!/usr/bin/env python
"""
Test script for HETROFL GUI enhancements.
Validates that all new components work correctly.
"""

import sys
import os

def test_imports():
    """Test that all new GUI components can be imported."""
    print("🧪 Testing GUI component imports...")
    
    try:
        from gui.gui_themes import ThemeManager
        print("✅ ThemeManager imported successfully")
        
        from gui.gui_styles import ModernStyles
        print("✅ ModernStyles imported successfully")
        
        from gui.gui_test_interface import ModelTestPanel, DatasetSelector, TestResultsWidget
        print("✅ Test interface components imported successfully")
        
        from gui.gui_dataset_manager import DatasetManager, DatasetAnalysisWidget
        print("✅ Dataset manager components imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_theme_manager():
    """Test the theme manager functionality."""
    print("\n🎨 Testing ThemeManager...")
    
    try:
        from gui.gui_themes import ThemeManager
        
        # Create theme manager
        theme_manager = ThemeManager()
        print(f"✅ ThemeManager created with default theme: {theme_manager.current_theme}")
        
        # Test available themes
        themes = theme_manager.get_available_themes()
        print(f"✅ Available themes: {themes}")
        
        # Test theme colors
        colors = theme_manager.get_theme_colors('light')
        print(f"✅ Light theme colors loaded: {len(colors)} colors")
        
        colors = theme_manager.get_theme_colors('dark')
        print(f"✅ Dark theme colors loaded: {len(colors)} colors")
        
        return True
    except Exception as e:
        print(f"❌ ThemeManager error: {e}")
        return False

def test_styles():
    """Test the modern styles functionality."""
    print("\n🎨 Testing ModernStyles...")
    
    try:
        from gui.gui_styles import ModernStyles
        from gui.gui_themes import ThemeManager
        
        styles = ModernStyles()
        theme_manager = ThemeManager()
        
        # Test stylesheet generation
        colors = theme_manager.get_theme_colors('light')
        stylesheet = styles.get_complete_stylesheet('light', colors)
        print(f"✅ Light theme stylesheet generated: {len(stylesheet)} characters")
        
        colors = theme_manager.get_theme_colors('dark')
        stylesheet = styles.get_complete_stylesheet('dark', colors)
        print(f"✅ Dark theme stylesheet generated: {len(stylesheet)} characters")
        
        # Test specific button styles
        button_style = styles.get_button_style('primary', colors)
        print(f"✅ Button style generated: {len(button_style)} characters")
        
        return True
    except Exception as e:
        print(f"❌ ModernStyles error: {e}")
        return False

def test_component_creation():
    """Test creating GUI components without Qt application."""
    print("\n🧩 Testing component creation...")
    
    try:
        # Test that classes can be imported and have expected methods
        from gui.gui_test_interface import ModelTestPanel, DatasetSelector
        from gui.gui_dataset_manager import DatasetManager
        
        # Check if classes have expected methods
        expected_methods = {
            'ModelTestPanel': ['set_model', 'run_test'],
            'DatasetSelector': ['browse_dataset', 'load_dataset'],
            'DatasetManager': ['browse_dataset', 'load_dataset', 'export_dataset']
        }
        
        for class_name, methods in expected_methods.items():
            cls = locals()[class_name]
            for method in methods:
                if hasattr(cls, method):
                    print(f"✅ {class_name}.{method} exists")
                else:
                    print(f"❌ {class_name}.{method} missing")
                    return False
        
        return True
    except Exception as e:
        print(f"❌ Component creation error: {e}")
        return False

def test_configuration():
    """Test that configuration is properly accessible."""
    print("\n⚙️ Testing configuration access...")
    
    try:
        from config.config import CLIENT_MODELS, DATA_DISTRIBUTIONS
        print(f"✅ CLIENT_MODELS: {CLIENT_MODELS}")
        print(f"✅ DATA_DISTRIBUTIONS: {DATA_DISTRIBUTIONS}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 HETROFL GUI Enhancement Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_theme_manager,
        test_styles,
        test_component_creation,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GUI enhancements are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())