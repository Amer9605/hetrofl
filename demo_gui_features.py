#!/usr/bin/env python
"""
Demo script showcasing the new HETROFL GUI features.
Run this to see the enhanced interface in action.
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PySide6.QtCore import Qt

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.gui_themes import ThemeManager
from gui.gui_test_interface import ModelTestPanel
from gui.gui_dataset_manager import DatasetManager


class DemoWindow(QMainWindow):
    """Demo window showcasing new features."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HETROFL GUI Enhancements Demo")
        self.setMinimumSize(800, 600)
        
        # Initialize theme manager
        self.theme_manager = ThemeManager()
        
        # Setup UI
        self.setup_ui()
        
        # Apply theme
        self.apply_theme()
    
    def setup_ui(self):
        """Setup the demo UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("HETROFL GUI Enhancements Demo")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("This demo showcases the new Material Design interface, "
                     "theme system, and enhanced components.")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Theme buttons
        theme_layout = QVBoxLayout()
        
        light_btn = QPushButton("Switch to Light Theme")
        light_btn.clicked.connect(lambda: self.change_theme('light'))
        theme_layout.addWidget(light_btn)
        
        dark_btn = QPushButton("Switch to Dark Theme")
        dark_btn.clicked.connect(lambda: self.change_theme('dark'))
        theme_layout.addWidget(dark_btn)
        
        layout.addLayout(theme_layout)
        
        # Feature buttons
        feature_layout = QVBoxLayout()
        
        dataset_btn = QPushButton("Open Dataset Manager")
        dataset_btn.clicked.connect(self.show_dataset_manager)
        feature_layout.addWidget(dataset_btn)
        
        test_btn = QPushButton("Show Model Test Interface")
        test_btn.clicked.connect(self.show_test_interface)
        feature_layout.addWidget(test_btn)
        
        layout.addLayout(feature_layout)
        
        # Info
        info = QLabel("✨ Features demonstrated:\n"
                     "• Modern Material Design styling\n"
                     "• Light/Dark theme switching\n"
                     "• Enhanced dataset management\n"
                     "• Comprehensive model testing\n"
                     "• Professional UI components")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
    
    def apply_theme(self):
        """Apply the current theme."""
        app = QApplication.instance()
        if app:
            self.theme_manager.apply_theme(app)
    
    def change_theme(self, theme_name):
        """Change to a specific theme."""
        app = QApplication.instance()
        if app:
            self.theme_manager.apply_theme(app, theme_name)
    
    def show_dataset_manager(self):
        """Show the dataset manager."""
        if not hasattr(self, 'dataset_manager'):
            self.dataset_manager = DatasetManager()
            self.dataset_manager.setWindowTitle("Dataset Manager Demo")
        
        self.dataset_manager.show()
        self.dataset_manager.activateWindow()
    
    def show_test_interface(self):
        """Show the test interface."""
        if not hasattr(self, 'test_window'):
            self.test_window = QMainWindow()
            self.test_window.setWindowTitle("Model Test Interface Demo")
            self.test_window.setMinimumSize(1000, 700)
            
            test_panel = ModelTestPanel(model=None, model_name="Demo Model")
            self.test_window.setCentralWidget(test_panel)
        
        self.test_window.show()
        self.test_window.activateWindow()


def main():
    """Run the demo."""
    app = QApplication(sys.argv)
    app.setApplicationName("HETROFL Demo")
    
    # Create and show demo window
    demo = DemoWindow()
    demo.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())