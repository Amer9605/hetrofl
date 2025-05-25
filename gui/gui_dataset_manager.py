"""
Dataset management interface for HETROFL GUI.
Provides comprehensive dataset exploration and management capabilities.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QTextEdit,
    QTableWidget, QTableWidgetItem, QTabWidget, QSplitter,
    QMessageBox, QSpinBox, QCheckBox, QLineEdit, QProgressBar,
    QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetTableModel(QAbstractTableModel):
    """Table model for displaying dataset contents."""
    
    def __init__(self, data=None):
        super().__init__()
        self._data = data if data is not None else pd.DataFrame()
        
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
        
    def columnCount(self, parent=QModelIndex()):
        return len(self._data.columns) if not self._data.empty else 0
        
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._data.empty:
            return None
            
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if pd.isna(value):
                return "NaN"
            elif isinstance(value, (float, np.float32, np.float64)):
                return f"{value:.4f}"
            return str(value)
            
        return None
        
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section]) if not self._data.empty else ""
            else:
                return str(section)
        return None
        
    def update_data(self, data):
        """Update the model with new data."""
        self.beginResetModel()
        self._data = data
        self.endResetModel()


class DatasetAnalysisWidget(QWidget):
    """Widget for dataset analysis and visualization."""
    
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the analysis widget UI."""
        layout = QVBoxLayout(self)
        
        # Create tabs for different analysis views
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Overview tab
        self.overview_tab = self.create_overview_tab()
        self.tabs.addTab(self.overview_tab, "Overview")
        
        # Statistics tab
        self.stats_tab = self.create_statistics_tab()
        self.tabs.addTab(self.stats_tab, "Statistics")
        
        # Visualizations tab
        self.viz_tab = self.create_visualizations_tab()
        self.tabs.addTab(self.viz_tab, "Visualizations")
        
        # Data Quality tab
        self.quality_tab = self.create_quality_tab()
        self.tabs.addTab(self.quality_tab, "Data Quality")
        
    def create_overview_tab(self):
        """Create the overview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Basic info
        info_group = QGroupBox("Dataset Information")
        info_layout = QFormLayout(info_group)
        
        self.shape_label = QLabel("--")
        self.size_label = QLabel("--")
        self.memory_label = QLabel("--")
        self.dtypes_label = QLabel("--")
        
        info_layout.addRow("Shape:", self.shape_label)
        info_layout.addRow("Size:", self.size_label)
        info_layout.addRow("Memory Usage:", self.memory_label)
        info_layout.addRow("Data Types:", self.dtypes_label)
        
        layout.addWidget(info_group)
        
        # Column information
        columns_group = QGroupBox("Column Information")
        columns_layout = QVBoxLayout(columns_group)
        
        self.columns_table = QTableWidget()
        self.columns_table.setColumnCount(4)
        self.columns_table.setHorizontalHeaderLabels(["Column", "Type", "Non-Null", "Unique"])
        self.columns_table.horizontalHeader().setStretchLastSection(True)
        columns_layout.addWidget(self.columns_table)
        
        layout.addWidget(columns_group)
        
        return widget
        
    def create_statistics_tab(self):
        """Create the statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Descriptive statistics
        stats_group = QGroupBox("Descriptive Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_table = QTableWidget()
        stats_layout.addWidget(self.stats_table)
        
        layout.addWidget(stats_group)
        
        # Correlation matrix
        corr_group = QGroupBox("Correlation Matrix")
        corr_layout = QVBoxLayout(corr_group)
        
        self.correlation_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        corr_layout.addWidget(self.correlation_canvas)
        
        layout.addWidget(corr_group)
        
        return widget
        
    def create_visualizations_tab(self):
        """Create the visualizations tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Distribution plots
        dist_group = QGroupBox("Feature Distributions")
        dist_layout = QVBoxLayout(dist_group)
        
        # Column selector for distribution
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Column:"))
        self.dist_column_combo = QComboBox()
        self.dist_column_combo.currentTextChanged.connect(self.update_distribution_plot)
        selector_layout.addWidget(self.dist_column_combo)
        selector_layout.addStretch()
        dist_layout.addLayout(selector_layout)
        
        self.distribution_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        dist_layout.addWidget(self.distribution_canvas)
        
        layout.addWidget(dist_group)
        
        # Target distribution (if applicable)
        target_group = QGroupBox("Target Distribution")
        target_layout = QVBoxLayout(target_group)
        
        self.target_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        target_layout.addWidget(self.target_canvas)
        
        layout.addWidget(target_group)
        
        return widget
        
    def create_quality_tab(self):
        """Create the data quality tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Missing values
        missing_group = QGroupBox("Missing Values")
        missing_layout = QVBoxLayout(missing_group)
        
        self.missing_table = QTableWidget()
        self.missing_table.setColumnCount(3)
        self.missing_table.setHorizontalHeaderLabels(["Column", "Missing Count", "Missing %"])
        missing_layout.addWidget(self.missing_table)
        
        layout.addWidget(missing_group)
        
        # Duplicates
        duplicates_group = QGroupBox("Duplicate Analysis")
        duplicates_layout = QFormLayout(duplicates_group)
        
        self.duplicates_label = QLabel("--")
        self.duplicate_percent_label = QLabel("--")
        
        duplicates_layout.addRow("Duplicate Rows:", self.duplicates_label)
        duplicates_layout.addRow("Duplicate %:", self.duplicate_percent_label)
        
        layout.addWidget(duplicates_group)
        
        # Outliers (for numeric columns)
        outliers_group = QGroupBox("Outlier Detection")
        outliers_layout = QVBoxLayout(outliers_group)
        
        self.outliers_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        outliers_layout.addWidget(self.outliers_canvas)
        
        layout.addWidget(outliers_group)
        
        return widget
        
    def analyze_dataset(self, data: pd.DataFrame, target_column: str = None):
        """Analyze the dataset and update all tabs."""
        self.current_data = data
        self.target_column = target_column
        
        # Update overview
        self.update_overview()
        
        # Update statistics
        self.update_statistics()
        
        # Update visualizations
        self.update_visualizations()
        
        # Update quality analysis
        self.update_quality_analysis()
        
    def update_overview(self):
        """Update the overview tab."""
        if self.current_data is None:
            return
            
        df = self.current_data
        
        # Basic information
        self.shape_label.setText(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        self.size_label.setText(f"{df.size:,} cells")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        self.memory_label.setText(f"{memory_mb:.2f} MB")
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        dtype_str = ", ".join([f"{count} {dtype}" for dtype, count in dtype_counts.items()])
        self.dtypes_label.setText(dtype_str)
        
        # Column information table
        self.columns_table.setRowCount(len(df.columns))
        
        for i, col in enumerate(df.columns):
            self.columns_table.setItem(i, 0, QTableWidgetItem(col))
            self.columns_table.setItem(i, 1, QTableWidgetItem(str(df[col].dtype)))
            self.columns_table.setItem(i, 2, QTableWidgetItem(str(df[col].notna().sum())))
            self.columns_table.setItem(i, 3, QTableWidgetItem(str(df[col].nunique())))
            
        self.columns_table.resizeColumnsToContents()
        
    def update_statistics(self):
        """Update the statistics tab."""
        if self.current_data is None:
            return
            
        df = self.current_data
        
        # Descriptive statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe()
            
            self.stats_table.setRowCount(len(stats.index))
            self.stats_table.setColumnCount(len(stats.columns))
            self.stats_table.setHorizontalHeaderLabels(stats.columns.tolist())
            self.stats_table.setVerticalHeaderLabels(stats.index.tolist())
            
            for i, row in enumerate(stats.index):
                for j, col in enumerate(stats.columns):
                    value = stats.loc[row, col]
                    self.stats_table.setItem(i, j, QTableWidgetItem(f"{value:.4f}"))
                    
            self.stats_table.resizeColumnsToContents()
            
        # Correlation matrix
        if len(numeric_cols) > 1:
            self.correlation_canvas.figure.clear()
            ax = self.correlation_canvas.figure.add_subplot(111)
            
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, fmt='.2f')
            ax.set_title('Feature Correlation Matrix')
            
            self.correlation_canvas.draw()
            
    def update_visualizations(self):
        """Update the visualizations tab."""
        if self.current_data is None:
            return
            
        df = self.current_data
        
        # Update column selector
        self.dist_column_combo.clear()
        self.dist_column_combo.addItems(df.columns.tolist())
        
        # Update distribution plot for first column
        if len(df.columns) > 0:
            self.update_distribution_plot(df.columns[0])
            
        # Update target distribution
        if self.target_column and self.target_column in df.columns:
            self.update_target_distribution()
            
    def update_distribution_plot(self, column_name: str):
        """Update the distribution plot for a specific column."""
        if self.current_data is None or not column_name:
            return
            
        df = self.current_data
        if column_name not in df.columns:
            return
            
        self.distribution_canvas.figure.clear()
        ax = self.distribution_canvas.figure.add_subplot(111)
        
        try:
            if df[column_name].dtype in ['object', 'category']:
                # Categorical data - bar plot
                value_counts = df[column_name].value_counts().head(20)
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Distribution of {column_name}')
                ax.set_xlabel(column_name)
                ax.set_ylabel('Count')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                # Numeric data - histogram
                df[column_name].hist(bins=30, ax=ax, alpha=0.7)
                ax.set_title(f'Distribution of {column_name}')
                ax.set_xlabel(column_name)
                ax.set_ylabel('Frequency')
                
            self.distribution_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting distribution for {column_name}: {e}")
            
    def update_target_distribution(self):
        """Update the target distribution plot."""
        if self.current_data is None or not self.target_column:
            return
            
        df = self.current_data
        if self.target_column not in df.columns:
            return
            
        self.target_canvas.figure.clear()
        ax = self.target_canvas.figure.add_subplot(111)
        
        try:
            target_counts = df[self.target_column].value_counts()
            target_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'Target Distribution - {self.target_column}')
            ax.set_xlabel(self.target_column)
            ax.set_ylabel('Count')
            
            # Add percentage labels
            total = len(df)
            for i, v in enumerate(target_counts.values):
                ax.text(i, v + total * 0.01, f'{v/total*100:.1f}%', 
                       ha='center', va='bottom')
                       
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            self.target_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting target distribution: {e}")
            
    def update_quality_analysis(self):
        """Update the data quality analysis."""
        if self.current_data is None:
            return
            
        df = self.current_data
        
        # Missing values analysis
        missing_info = []
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_percent = (missing_count / len(df)) * 100
            missing_info.append((col, missing_count, missing_percent))
            
        self.missing_table.setRowCount(len(missing_info))
        for i, (col, count, percent) in enumerate(missing_info):
            self.missing_table.setItem(i, 0, QTableWidgetItem(col))
            self.missing_table.setItem(i, 1, QTableWidgetItem(str(count)))
            self.missing_table.setItem(i, 2, QTableWidgetItem(f"{percent:.2f}%"))
            
        self.missing_table.resizeColumnsToContents()
        
        # Duplicates analysis
        duplicate_count = df.duplicated().sum()
        duplicate_percent = (duplicate_count / len(df)) * 100
        self.duplicates_label.setText(f"{duplicate_count:,}")
        self.duplicate_percent_label.setText(f"{duplicate_percent:.2f}%")
        
        # Outliers visualization (box plots for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.outliers_canvas.figure.clear()
            
            # Show up to 6 numeric columns
            cols_to_plot = numeric_cols[:6]
            n_cols = len(cols_to_plot)
            
            for i, col in enumerate(cols_to_plot):
                ax = self.outliers_canvas.figure.add_subplot(2, 3, i + 1)
                df[col].plot(kind='box', ax=ax)
                ax.set_title(f'{col}')
                ax.set_ylabel('Value')
                
            self.outliers_canvas.figure.tight_layout()
            self.outliers_canvas.draw()


class DatasetManager(QWidget):
    """Complete dataset management interface."""
    
    dataset_loaded = Signal(pd.DataFrame, str)  # data, target_column
    
    def __init__(self):
        super().__init__()
        self.current_dataset = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dataset manager UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Dataset Manager")
        title_label.setProperty("class", "title")
        layout.addWidget(title_label)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(main_splitter)
        
        # Top section: Dataset loading and preview
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # Dataset loading controls
        loading_group = QGroupBox("Dataset Loading")
        loading_layout = QFormLayout(loading_group)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a dataset file...")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_dataset)
        
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_button)
        loading_layout.addRow("Dataset File:", file_layout)
        
        # Loading options
        options_layout = QHBoxLayout()
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Auto-detect", "CSV", "Parquet", "JSON"])
        options_layout.addWidget(QLabel("Format:"))
        options_layout.addWidget(self.format_combo)
        
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(-1, 1000000)
        self.sample_size_spin.setValue(-1)
        self.sample_size_spin.setSpecialValueText("All")
        options_layout.addWidget(QLabel("Sample Size:"))
        options_layout.addWidget(self.sample_size_spin)
        
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        options_layout.addWidget(QLabel("Target Column:"))
        options_layout.addWidget(self.target_combo)
        
        loading_layout.addRow("Options:", options_layout)
        
        # Load button
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        self.load_button.setEnabled(False)
        
        self.export_button = QPushButton("Export Dataset")
        self.export_button.clicked.connect(self.export_dataset)
        self.export_button.setEnabled(False)
        
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        loading_layout.addRow("Actions:", button_layout)
        
        top_layout.addWidget(loading_group)
        
        # Dataset preview
        preview_group = QGroupBox("Dataset Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Info bar
        self.info_label = QLabel("No dataset loaded")
        preview_layout.addWidget(self.info_label)
        
        # Preview table
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preview_table.setMaximumHeight(300)
        preview_layout.addWidget(self.preview_table)
        
        top_layout.addWidget(preview_group)
        main_splitter.addWidget(top_widget)
        
        # Bottom section: Dataset analysis
        self.analysis_widget = DatasetAnalysisWidget()
        main_splitter.addWidget(self.analysis_widget)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 600])
        
        # Connect signals
        self.file_path_edit.textChanged.connect(self.on_path_changed)
        
    def browse_dataset(self):
        """Open file dialog to select dataset."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset File",
            "",
            "Data Files (*.csv *.parquet *.json);;CSV Files (*.csv);;Parquet Files (*.parquet);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
            
    def on_path_changed(self, path):
        """Handle path change."""
        self.load_button.setEnabled(bool(path and os.path.exists(path)))
        
        if path and os.path.exists(path):
            self.preview_file(path)
            
    def preview_file(self, path):
        """Preview the selected file."""
        try:
            # Load a small sample for preview
            if path.endswith('.csv'):
                df = pd.read_csv(path, nrows=100)
            elif path.endswith('.parquet'):
                df = pd.read_parquet(path)
                df = df.head(100)
            elif path.endswith('.json'):
                df = pd.read_json(path, lines=True, nrows=100)
            else:
                return
                
            # Update target column options
            self.target_combo.clear()
            self.target_combo.addItems(df.columns.tolist())
            
            # Update preview table
            self.update_preview_table(df)
            
        except Exception as e:
            self.info_label.setText(f"Error loading preview: {str(e)}")
            
    def update_preview_table(self, df):
        """Update the preview table with data."""
        self.preview_table.setRowCount(min(20, len(df)))
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i in range(min(20, len(df))):
            for j, col in enumerate(df.columns):
                value = df.iloc[i, j]
                if pd.isna(value):
                    item_text = "NaN"
                elif isinstance(value, (float, np.float32, np.float64)):
                    item_text = f"{value:.4f}"
                else:
                    item_text = str(value)
                    
                item = QTableWidgetItem(item_text)
                self.preview_table.setItem(i, j, item)
                
        self.preview_table.resizeColumnsToContents()
        
        # Update info
        self.info_label.setText(f"Preview: {len(df)} rows × {len(df.columns)} columns")
        
    def load_dataset(self):
        """Load the complete dataset."""
        file_path = self.file_path_edit.text()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Warning", "Please select a valid dataset file.")
            return
            
        try:
            # Load dataset based on format
            format_type = self.format_combo.currentText()
            
            if format_type == 'Auto-detect' or format_type == 'CSV':
                df = pd.read_csv(file_path)
            elif format_type == 'Parquet':
                df = pd.read_parquet(file_path)
            elif format_type == 'JSON':
                df = pd.read_json(file_path, lines=True)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
            # Apply sampling if specified
            sample_size = self.sample_size_spin.value()
            if sample_size > 0 and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                
            # Store dataset
            self.current_dataset = df
            target_column = self.target_combo.currentText()
            
            # Update preview
            self.update_preview_table(df.head(20))
            self.info_label.setText(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")
            
            # Update analysis
            self.analysis_widget.analyze_dataset(df, target_column)
            
            # Enable export
            self.export_button.setEnabled(True)
            
            # Emit signal
            self.dataset_loaded.emit(df, target_column)
            
            QMessageBox.information(self, "Success", f"Dataset loaded successfully!\n{len(df):,} rows × {len(df.columns)} columns")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
            
    def export_dataset(self):
        """Export the current dataset."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "Warning", "No dataset to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Dataset", "", "CSV Files (*.csv);;Parquet Files (*.parquet)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.current_dataset.to_csv(file_path, index=False)
                elif file_path.endswith('.parquet'):
                    self.current_dataset.to_parquet(file_path, index=False)
                else:
                    # Default to CSV
                    self.current_dataset.to_csv(file_path, index=False)
                    
                QMessageBox.information(self, "Success", f"Dataset exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export dataset: {str(e)}")
                
    def get_current_dataset(self):
        """Get the currently loaded dataset."""
        return self.current_dataset
        
    def get_target_column(self):
        """Get the selected target column."""
        return self.target_combo.currentText()
