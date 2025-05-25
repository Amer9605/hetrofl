"""
Modern stylesheet definitions for HETROFL GUI.
Implements Material Design principles with professional styling.
"""

from typing import Dict


class ModernStyles:
    """
    Provides modern, Material Design-inspired stylesheets for the HETROFL GUI.
    """
    
    def __init__(self):
        self.base_font_size = 10
        self.border_radius = 8
        self.transition_duration = 200
        
    def get_complete_stylesheet(self, theme_name: str, colors: Dict[str, str]) -> str:
        """
        Get complete stylesheet for the specified theme.
        
        Args:
            theme_name: Theme name ('light' or 'dark')
            colors: Color palette dictionary
            
        Returns:
            Complete CSS stylesheet string
        """
        return f"""
        /* ===== GLOBAL STYLES ===== */
        QApplication {{
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            font-size: {self.base_font_size}pt;
            color: {colors['text_primary']};
        }}
        
        /* ===== MAIN WINDOW ===== */
        QMainWindow {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
        }}
        
        QWidget {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
        }}
        
        /* ===== BUTTONS ===== */
        QPushButton {{
            background-color: {colors['primary']};
            border: none;
            border-radius: {self.border_radius}px;
            padding: 12px 24px;
            color: white;
            font-weight: 600;
            font-size: {self.base_font_size + 1}pt;
            min-height: 20px;
            min-width: 80px;
        }}
        
        QPushButton:hover {{
            background-color: {colors['primary_dark']};
            transform: translateY(-1px);
        }}
        
        QPushButton:pressed {{
            background-color: {colors['primary_dark']};
            transform: translateY(0px);
        }}
        
        QPushButton:disabled {{
            background-color: {colors['text_disabled']};
            color: {colors['text_secondary']};
        }}
        
        /* Secondary Button Style */
        QPushButton[class="secondary"] {{
            background-color: {colors['surface']};
            border: 2px solid {colors['primary']};
            color: {colors['primary']};
        }}
        
        QPushButton[class="secondary"]:hover {{
            background-color: {colors['primary_light']};
            color: {colors['primary_dark']};
        }}
        
        /* Danger Button Style */
        QPushButton[class="danger"] {{
            background-color: {colors['error']};
        }}
        
        QPushButton[class="danger"]:hover {{
            background-color: #D32F2F;
        }}
        
        /* Success Button Style */
        QPushButton[class="success"] {{
            background-color: {colors['success']};
        }}
        
        QPushButton[class="success"]:hover {{
            background-color: #388E3C;
        }}
        
        /* ===== GROUP BOXES ===== */
        QGroupBox {{
            font-weight: 600;
            font-size: {self.base_font_size + 1}pt;
            border: 2px solid {colors['border']};
            border-radius: {self.border_radius}px;
            margin-top: 1ex;
            padding-top: 15px;
            background-color: {colors['surface']};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
            color: {colors['text_primary']};
            background-color: {colors['surface']};
        }}
        
        /* ===== TABS ===== */
        QTabWidget::pane {{
            border: 1px solid {colors['border']};
            border-radius: {self.border_radius}px;
            background-color: {colors['surface']};
            margin-top: -1px;
        }}
        
        QTabBar::tab {{
            background-color: {colors['background']};
            border: 1px solid {colors['border']};
            border-bottom: none;
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: {self.border_radius}px;
            border-top-right-radius: {self.border_radius}px;
            color: {colors['text_secondary']};
            font-weight: 500;
        }}
        
        QTabBar::tab:selected {{
            background-color: {colors['surface']};
            color: {colors['primary']};
            border-bottom: 2px solid {colors['primary']};
            font-weight: 600;
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {colors['hover']};
            color: {colors['text_primary']};
        }}
        
        /* ===== FORM CONTROLS ===== */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            border: 2px solid {colors['border']};
            border-radius: {self.border_radius}px;
            padding: 8px 12px;
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            font-size: {self.base_font_size}pt;
            min-height: 20px;
        }}
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {colors['primary']};
            outline: none;
        }}
        
        QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
            border-color: {colors['primary_light']};
        }}
        
        /* ===== COMBO BOX ===== */
        QComboBox::drop-down {{
            border: none;
            width: 30px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {colors['text_secondary']};
            margin-right: 10px;
        }}
        
        QComboBox QAbstractItemView {{
            border: 1px solid {colors['border']};
            border-radius: {self.border_radius}px;
            background-color: {colors['surface']};
            selection-background-color: {colors['primary_light']};
            selection-color: {colors['text_primary']};
            padding: 4px;
        }}
        
        /* ===== CHECKBOXES ===== */
        QCheckBox {{
            spacing: 8px;
            color: {colors['text_primary']};
            font-size: {self.base_font_size}pt;
        }}
        
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {colors['border']};
            border-radius: 4px;
            background-color: {colors['surface']};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {colors['primary']};
            border-color: {colors['primary']};
            image: none;
        }}
        
        QCheckBox::indicator:checked::after {{
            content: "✓";
            color: white;
            font-weight: bold;
        }}
        
        QCheckBox::indicator:hover {{
            border-color: {colors['primary']};
        }}
        
        /* ===== PROGRESS BAR ===== */
        QProgressBar {{
            border: none;
            border-radius: {self.border_radius}px;
            background-color: {colors['hover']};
            text-align: center;
            color: {colors['text_primary']};
            font-weight: 600;
            height: 24px;
        }}
        
        QProgressBar::chunk {{
            background-color: {colors['primary']};
            border-radius: {self.border_radius}px;
        }}
        
        /* ===== LABELS ===== */
        QLabel {{
            color: {colors['text_primary']};
            font-size: {self.base_font_size}pt;
        }}
        
        QLabel[class="title"] {{
            font-size: {self.base_font_size + 4}pt;
            font-weight: 700;
            color: {colors['text_primary']};
        }}
        
        QLabel[class="subtitle"] {{
            font-size: {self.base_font_size + 2}pt;
            font-weight: 600;
            color: {colors['text_secondary']};
        }}
        
        QLabel[class="caption"] {{
            font-size: {self.base_font_size - 1}pt;
            color: {colors['text_secondary']};
        }}
        
        /* ===== STATUS BAR ===== */
        QStatusBar {{
            background-color: {colors['surface']};
            border-top: 1px solid {colors['border']};
            color: {colors['text_secondary']};
            padding: 4px;
        }}
        
        /* ===== SPLITTER ===== */
        QSplitter::handle {{
            background-color: {colors['border']};
            width: 2px;
            height: 2px;
        }}
        
        QSplitter::handle:hover {{
            background-color: {colors['primary']};
        }}
        
        /* ===== TEXT EDIT ===== */
        QTextEdit, QPlainTextEdit {{
            border: 2px solid {colors['border']};
            border-radius: {self.border_radius}px;
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            padding: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: {self.base_font_size - 1}pt;
        }}
        
        QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {colors['primary']};
        }}
        
        /* ===== TABLE VIEW ===== */
        QTableView {{
            border: 1px solid {colors['border']};
            border-radius: {self.border_radius}px;
            background-color: {colors['surface']};
            alternate-background-color: {colors['hover']};
            selection-background-color: {colors['primary_light']};
            gridline-color: {colors['border']};
        }}
        
        QTableView::item {{
            padding: 8px;
            border: none;
        }}
        
        QTableView::item:selected {{
            background-color: {colors['primary_light']};
            color: {colors['text_primary']};
        }}
        
        QHeaderView::section {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
            padding: 8px;
            border: 1px solid {colors['border']};
            font-weight: 600;
        }}
        
        /* ===== SCROLL BARS ===== */
        QScrollBar:vertical {{
            background-color: {colors['background']};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {colors['text_disabled']};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {colors['text_secondary']};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            background-color: {colors['background']};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {colors['text_disabled']};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {colors['text_secondary']};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        
        /* ===== MENU BAR ===== */
        QMenuBar {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border-bottom: 1px solid {colors['border']};
            padding: 4px;
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 8px 12px;
            border-radius: 4px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {colors['hover']};
        }}
        
        QMenu {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {self.border_radius}px;
            padding: 4px;
        }}
        
        QMenu::item {{
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        QMenu::item:selected {{
            background-color: {colors['primary_light']};
        }}
        
        /* ===== TOOL TIP ===== */
        QToolTip {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            border-radius: 4px;
            padding: 8px;
            font-size: {self.base_font_size - 1}pt;
        }}
        
        /* ===== CUSTOM CLASSES ===== */
        .card {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {self.border_radius}px;
            padding: 16px;
        }}
        
        .elevated {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {self.border_radius}px;
        }}
        
        .metric-value {{
            font-size: {self.base_font_size + 3}pt;
            font-weight: 700;
            color: {colors['primary']};
        }}
        
        .metric-label {{
            font-size: {self.base_font_size - 1}pt;
            color: {colors['text_secondary']};
            font-weight: 500;
        }}
        """
    
    def get_button_style(self, button_type: str, colors: Dict[str, str]) -> str:
        """Get specific button style."""
        styles = {
            'primary': f"""
                background-color: {colors['primary']};
                color: white;
                border: none;
                border-radius: {self.border_radius}px;
                padding: 12px 24px;
                font-weight: 600;
            """,
            'secondary': f"""
                background-color: {colors['surface']};
                color: {colors['primary']};
                border: 2px solid {colors['primary']};
                border-radius: {self.border_radius}px;
                padding: 10px 22px;
                font-weight: 600;
            """,
            'danger': f"""
                background-color: {colors['error']};
                color: white;
                border: none;
                border-radius: {self.border_radius}px;
                padding: 12px 24px;
                font-weight: 600;
            """,
            'success': f"""
                background-color: {colors['success']};
                color: white;
                border: none;
                border-radius: {self.border_radius}px;
                padding: 12px 24px;
                font-weight: 600;
            """
        }
        return styles.get(button_type, styles['primary'])
    
    def get_card_style(self, colors: Dict[str, str]) -> str:
        """Get card container style."""
        return f"""
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {self.border_radius}px;
            padding: 16px;
            margin: 8px;
        """