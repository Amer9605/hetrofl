"""
Modern theme management system for HETROFL GUI.
Provides Material Design-inspired themes with light/dark mode support.
"""

import os
import json
from typing import Dict, Optional
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings, QObject, Signal
from PySide6.QtGui import QPalette, QColor

from .gui_styles import ModernStyles


class ThemeManager(QObject):
    """
    Manages application themes with Material Design principles.
    Supports light/dark themes with smooth transitions and persistence.
    """
    
    theme_changed = Signal(str)  # Emitted when theme changes
    
    # Material Design Color Palettes
    MATERIAL_COLORS = {
        'light': {
            'primary': '#2196F3',
            'primary_dark': '#1976D2', 
            'primary_light': '#BBDEFB',
            'secondary': '#FFC107',
            'secondary_dark': '#F57C00',
            'secondary_light': '#FFF8E1',
            'background': '#FAFAFA',
            'surface': '#FFFFFF',
            'error': '#F44336',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'info': '#2196F3',
            'text_primary': '#212121',
            'text_secondary': '#757575',
            'text_disabled': '#BDBDBD',
            'divider': '#E0E0E0',
            'border': '#E0E0E0',
            'hover': '#F5F5F5',
            'selected': '#E3F2FD',
            'shadow': 'rgba(0, 0, 0, 0.1)'
        },
        'dark': {
            'primary': '#2196F3',
            'primary_dark': '#1565C0',
            'primary_light': '#42A5F5',
            'secondary': '#FFC107',
            'secondary_dark': '#F57C00',
            'secondary_light': '#FFECB3',
            'background': '#121212',
            'surface': '#1E1E1E',
            'error': '#CF6679',
            'success': '#81C784',
            'warning': '#FFB74D',
            'info': '#64B5F6',
            'text_primary': '#FFFFFF',
            'text_secondary': '#B3B3B3',
            'text_disabled': '#666666',
            'divider': '#333333',
            'border': '#333333',
            'hover': '#2C2C2C',
            'selected': '#1E3A8A',
            'shadow': 'rgba(0, 0, 0, 0.3)'
        }
    }
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings('HETROFL', 'GUI')
        self.current_theme = self.settings.value('theme', 'light')
        self.styles = ModernStyles()
        
    def get_available_themes(self) -> list:
        """Get list of available theme names."""
        return ['light', 'dark', 'auto']
    
    def get_current_theme(self) -> str:
        """Get the currently active theme name."""
        return self.current_theme
    
    def get_theme_colors(self, theme_name: Optional[str] = None) -> Dict[str, str]:
        """Get color palette for specified theme."""
        if theme_name is None:
            theme_name = self.current_theme
            
        if theme_name == 'auto':
            # Detect system theme (simplified - could be enhanced)
            theme_name = 'dark'  # Default to dark for now
            
        return self.MATERIAL_COLORS.get(theme_name, self.MATERIAL_COLORS['light'])
    
    def apply_theme(self, app: QApplication, theme_name: str = None):
        """
        Apply theme to the application.
        
        Args:
            app: QApplication instance
            theme_name: Theme to apply ('light', 'dark', 'auto')
        """
        if theme_name is None:
            theme_name = self.current_theme
            
        if theme_name not in self.get_available_themes():
            theme_name = 'light'
            
        # Handle auto theme
        if theme_name == 'auto':
            theme_name = self._detect_system_theme()
            
        self.current_theme = theme_name
        self.settings.setValue('theme', theme_name)
        
        # Get theme colors
        colors = self.get_theme_colors(theme_name)
        
        # Apply stylesheet
        stylesheet = self.styles.get_complete_stylesheet(theme_name, colors)
        app.setStyleSheet(stylesheet)
        
        # Apply palette for native widgets
        self._apply_palette(app, colors)
        
        # Emit theme changed signal
        self.theme_changed.emit(theme_name)
    
    def toggle_theme(self, app: QApplication):
        """Toggle between light and dark themes."""
        new_theme = 'dark' if self.current_theme == 'light' else 'light'
        self.apply_theme(app, new_theme)
    
    def _detect_system_theme(self) -> str:
        """
        Detect system theme preference.
        This is a simplified implementation - could be enhanced with platform-specific detection.
        """
        # For now, return dark as default
        # In a full implementation, this would check system settings
        return 'dark'
    
    def _apply_palette(self, app: QApplication, colors: Dict[str, str]):
        """Apply color palette to application for native widget styling."""
        palette = QPalette()
        
        # Convert hex colors to QColor
        def hex_to_qcolor(hex_color: str) -> QColor:
            return QColor(hex_color)
        
        # Set palette colors
        palette.setColor(QPalette.Window, hex_to_qcolor(colors['background']))
        palette.setColor(QPalette.WindowText, hex_to_qcolor(colors['text_primary']))
        palette.setColor(QPalette.Base, hex_to_qcolor(colors['surface']))
        palette.setColor(QPalette.AlternateBase, hex_to_qcolor(colors['hover']))
        palette.setColor(QPalette.ToolTipBase, hex_to_qcolor(colors['surface']))
        palette.setColor(QPalette.ToolTipText, hex_to_qcolor(colors['text_primary']))
        palette.setColor(QPalette.Text, hex_to_qcolor(colors['text_primary']))
        palette.setColor(QPalette.Button, hex_to_qcolor(colors['surface']))
        palette.setColor(QPalette.ButtonText, hex_to_qcolor(colors['text_primary']))
        palette.setColor(QPalette.BrightText, hex_to_qcolor(colors['error']))
        palette.setColor(QPalette.Link, hex_to_qcolor(colors['primary']))
        palette.setColor(QPalette.Highlight, hex_to_qcolor(colors['primary']))
        palette.setColor(QPalette.HighlightedText, hex_to_qcolor('#FFFFFF'))
        
        app.setPalette(palette)
    
    def get_icon_color(self, theme_name: Optional[str] = None) -> str:
        """Get appropriate icon color for the theme."""
        colors = self.get_theme_colors(theme_name)
        return colors['text_primary']
    
    def save_theme_preference(self, theme_name: str):
        """Save theme preference to settings."""
        self.settings.setValue('theme', theme_name)
        self.current_theme = theme_name