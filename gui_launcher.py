#!/usr/bin/env python
"""
Launcher script for the HETROFL GUI application.
This script starts the main application and initializes all GUI components.
"""

import os
import sys
import argparse
from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox, QStyleFactory
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QIcon, QFont, QColor

# Import GUI components
from gui_main import GlobalModelWindow


def show_splash_screen(app, message="Loading application..."):
    """Show a modern splash screen while the application loads."""
    # Create an enhanced splash screen
    splash_pixmap = QPixmap(700, 400)
    splash_pixmap.fill(QColor("#2196F3"))
    
    splash = QSplashScreen(splash_pixmap)
    splash.showMessage(
        "HETROFL v2.0\nHeterogeneous Federated Learning System\nEnhanced with Modern UI & Testing Capabilities\n\nLoading...",
        alignment=Qt.AlignCenter | Qt.AlignBottom,
        color=Qt.white
    )
    
    # Show the splash screen
    splash.show()
    app.processEvents()
    
    return splash


def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import numpy
        import pandas
        import sklearn
        import xgboost
        import lightgbm
        import matplotlib
        import torch
        import mlflow
        import pyqtgraph
        import PySide6
        return True
    except ImportError as e:
        return str(e)


def main():
    """Main function to launch the HETROFL GUI."""
    parser = argparse.ArgumentParser(description="HETROFL GUI Enhanced v2.0")
    parser.add_argument('--style', default='Fusion', help='Qt style to use')
    parser.add_argument('--theme', default='light', choices=['light', 'dark', 'auto'], help='Theme to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency checks')
    args = parser.parse_args()
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Set application name and style
    app.setApplicationName("HETROFL")
    
    # Set style if available
    if args.style in QStyleFactory.keys():
        app.setStyle(args.style)
    else:
        print(f"Warning: Style '{args.style}' not available. Using default.")
        print(f"Available styles: {QStyleFactory.keys()}")
    
    # Show splash screen
    splash = show_splash_screen(app)
    
    if not args.skip_checks:
        # Check dependencies
        splash.showMessage("Checking dependencies...")
        app.processEvents()
        
        dependency_check = check_dependencies()
        if dependency_check is not True:
            QMessageBox.critical(
                None, 
                "Dependency Error",
                f"Missing dependency: {dependency_check}\n\n"
                f"Please install all required dependencies."
            )
            return 1
    
    try:
        # Create main window
        splash.showMessage("Initializing application...")
        app.processEvents()
        
        main_window = GlobalModelWindow()
        
        # Apply theme if specified
        if args.theme:
            main_window.theme_manager.apply_theme(app, args.theme)
        
        # Close splash and show main window
        splash.finish(main_window)
        main_window.show()
        
        # Execute application
        return app.exec()
    
    except Exception as e:
        if args.debug:
            import traceback
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        else:
            error_msg = f"Error: {str(e)}"
            
        QMessageBox.critical(None, "Application Error", error_msg)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
