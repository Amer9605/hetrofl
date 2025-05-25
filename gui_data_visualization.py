#!/usr/bin/env python
"""
Module for data visualization components in the HETROFL GUI.
Provides interactive tools for exploring dataset characteristics
and visualizing the data used by local models.
"""

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QSplitter, QTabWidget,
    QTableView, QHeaderView, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QFont

import pyqtgraph as pg
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DataTableModel(QAbstractTableModel):
    """Table model for displaying dataset samples."""
    def __init__(self, data=None):
        super().__init__()
        self.data = data if data is not None else pd.DataFrame()
        self.header_labels = self.data.columns
    
    def rowCount(self, parent=QModelIndex()):
        return len(self.data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.data.columns)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        if role == Qt.DisplayRole:
            value = self.data.iloc[index.row(), index.column()]
            if isinstance(value, (float, np.float32, np.float64)):
                return f"{value:.4f}"
            return str(value)
            
        return None
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section < len(self.header_labels):
                    return str(self.header_labels[section])
            else:
                return str(section)
        return None
    
    def update_data(self, data):
        """Update the table with new data."""
        self.beginResetModel()
        self.data = data
        self.header_labels = data.columns
        self.endResetModel()


class DistributionPlot(FigureCanvas):
    """Matplotlib canvas for plotting feature distributions."""
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def plot_distribution(self, data, feature, by_class=None):
        """Plot distribution of a feature, optionally grouped by class."""
        self.axes.clear()
        
        if by_class is not None and by_class in data.columns:
            # Plot distribution grouped by class
            for class_value in data[by_class].unique():
                subset = data[data[by_class] == class_value]
                self.axes.hist(subset[feature], alpha=0.5, label=f'Class {class_value}')
            self.axes.legend()
        else:
            # Plot overall distribution
            self.axes.hist(data[feature], bins=30)
        
        self.axes.set_title(f'Distribution of {feature}')
        self.axes.set_xlabel(feature)
        self.axes.set_ylabel('Frequency')
        self.draw()


class ScatterPlot(pg.PlotWidget):
    """Interactive scatter plot for exploring feature relationships."""
    def __init__(self, title="Feature Relationships"):
        super().__init__(title=title)
        self.setLabel('left', 'Feature 2')
        self.setLabel('bottom', 'Feature 1')
        self.setBackground('w')
        self.addLegend()
        
        # Color map for different classes
        self.colors = [
            (255, 0, 0, 150),      # Red
            (0, 255, 0, 150),      # Green
            (0, 0, 255, 150),      # Blue
            (255, 255, 0, 150),    # Yellow
            (0, 255, 255, 150),    # Cyan
            (255, 0, 255, 150),    # Magenta
            (128, 128, 128, 150)   # Gray
        ]
        
        self.scatter_items = []
    
    def update_scatter(self, data, x_feature, y_feature, class_column=None):
        """Update the scatter plot with new data."""
        # Clear previous scatter items
        for item in self.scatter_items:
            self.removeItem(item)
        self.scatter_items = []
        
        # If no class column is specified, plot all points in one color
        if class_column is None or class_column not in data.columns:
            scatter = pg.ScatterPlotItem(
                pen=pg.mkPen(None), 
                brush=pg.mkBrush(30, 100, 200, 150),
                size=10
            )
            
            scatter.setData(
                x=data[x_feature].values,
                y=data[y_feature].values
            )
            
            self.addItem(scatter)
            self.scatter_items.append(scatter)
            
        else:
            # Plot points by class
            for i, class_value in enumerate(sorted(data[class_column].unique())):
                color_idx = i % len(self.colors)
                color = self.colors[color_idx]
                
                subset = data[data[class_column] == class_value]
                scatter = pg.ScatterPlotItem(
                    pen=pg.mkPen(None), 
                    brush=pg.mkBrush(*color),
                    size=10,
                    name=f'Class {class_value}'
                )
                
                scatter.setData(
                    x=subset[x_feature].values,
                    y=subset[y_feature].values
                )
                
                self.addItem(scatter)
                self.scatter_items.append(scatter)
        
        self.setLabel('left', y_feature)
        self.setLabel('bottom', x_feature)


class ClassBalancePlot(FigureCanvas):
    """Matplotlib canvas for visualizing class balance."""
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def update_plot(self, data, class_column):
        """Update the class balance visualization."""
        self.axes.clear()
        
        # Count samples per class
        if class_column in data.columns:
            class_counts = data[class_column].value_counts().sort_index()
            
            # Plot as bar chart
            self.axes.bar(class_counts.index.astype(str), class_counts.values)
            
            # Add percentages on top of the bars
            total = class_counts.sum()
            for i, count in enumerate(class_counts.values):
                percentage = (count / total) * 100
                self.axes.text(i, count + (total * 0.01), 
                              f"{percentage:.1f}%", 
                              ha='center')
            
            self.axes.set_title('Class Distribution')
            self.axes.set_xlabel('Class')
            self.axes.set_ylabel('Count')
            
            # Add value labels on top of bars
            for i, v in enumerate(class_counts.values):
                self.axes.text(i, v + 5, str(v), ha='center')
            
        self.draw()


class FeatureCorrelationMatrix(FigureCanvas):
    """Matplotlib canvas for visualizing feature correlations."""
    def __init__(self, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def update_correlation(self, data, target_column=None, max_features=20):
        """Update the correlation matrix."""
        self.axes.clear()
        import seaborn as sns
        
        # Select numerical columns only
        numerical_data = data.select_dtypes(include=['int64', 'float64'])
        
        # If we have too many features, select the most correlated ones
        if target_column and target_column in numerical_data.columns:
            if len(numerical_data.columns) > max_features:
                correlations = numerical_data.corr()[target_column].abs()
                top_features = correlations.nlargest(max_features).index.tolist()
                
                # Make sure target column is included
                if target_column not in top_features:
                    top_features.append(target_column)
                
                numerical_data = numerical_data[top_features]
        
        # Calculate the correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Create heatmap
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            ax=self.axes,
            fmt='.2f',
            linewidths=0.5
        )
        
        self.axes.set_title('Feature Correlation Matrix')
        self.draw()


class PartitionComparisonPlot(pg.PlotWidget):
    """Interactive plot for comparing distributions across local model partitions."""
    def __init__(self, title="Partition Comparison"):
        super().__init__(title=title)
        self.setBackground('w')
        self.setLabel('left', 'Value')
        self.setLabel('bottom', 'Model Partition')
        self.addLegend()
        
        self.partition_data = {}
        self.box_items = []
    
    def update_partition_data(self, partition_id, data, feature=None):
        """Update data for a specific partition."""
        if feature and feature in data.columns:
            self.partition_data[partition_id] = data[feature].values
            self.update_plot(feature)
    
    def update_plot(self, feature_name=None):
        """Update the partition comparison visualization."""
        # Clear previous box items
        for item in self.box_items:
            self.removeItem(item)
        self.box_items = []
        
        # Sort partitions by ID
        partition_ids = sorted(self.partition_data.keys())
        
        # Set up data for box plots
        pos = []
        data = []
        
        for i, p_id in enumerate(partition_ids):
            pos.append(i)
            data.append(self.partition_data[p_id])
        
        # Create box plot for each partition
        for i in range(len(pos)):
            box = pg.PlotDataItem(
                x=np.ones(len(data[i])) * pos[i], 
                y=data[i], 
                pen=pg.mkPen(None),
                symbolBrush=(100, 100, 200),
                symbolPen=(100, 100, 200),
                symbolSize=5
            )
            self.addItem(box)
            self.box_items.append(box)
        
        self.setXRange(-0.5, len(pos) - 0.5)
        self.getAxis('bottom').setTicks([list(zip(pos, [f"Model {p_id+1}" for p_id in partition_ids]))])
        
        if feature_name:
            self.setLabel('left', feature_name)


class DataVisualizationWidget(QWidget):
    """Main widget for all data visualization components."""
    def __init__(self):
        super().__init__()
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create tabs for different visualizations
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Tab 1: Overview
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        
        # Top section with dataset statistics
        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QFormLayout(stats_group)
        
        self.sample_count_label = QLabel("0")
        self.feature_count_label = QLabel("0")
        self.class_count_label = QLabel("0")
        self.missing_values_label = QLabel("0")
        
        stats_layout.addRow("Samples:", self.sample_count_label)
        stats_layout.addRow("Features:", self.feature_count_label)
        stats_layout.addRow("Classes:", self.class_count_label)
        stats_layout.addRow("Missing Values:", self.missing_values_label)
        
        overview_layout.addWidget(stats_group)
        
        # Class balance visualization
        self.class_balance_plot = ClassBalancePlot(width=6, height=3)
        overview_layout.addWidget(self.class_balance_plot)
        
        # Add overview tab
        self.tabs.addTab(overview_tab, "Dataset Overview")
        
        # Tab 2: Feature exploration
        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)
        
        # Feature selection controls
        controls_layout = QHBoxLayout()
        
        self.x_feature_combo = QComboBox()
        self.y_feature_combo = QComboBox()
        self.class_column_combo = QComboBox()
        
        controls_layout.addWidget(QLabel("X Feature:"))
        controls_layout.addWidget(self.x_feature_combo)
        controls_layout.addWidget(QLabel("Y Feature:"))
        controls_layout.addWidget(self.y_feature_combo)
        controls_layout.addWidget(QLabel("Class:"))
        controls_layout.addWidget(self.class_column_combo)
        
        features_layout.addLayout(controls_layout)
        
        # Scatter plot
        self.scatter_plot = ScatterPlot(title="Feature Relationships")
        features_layout.addWidget(self.scatter_plot)
        
        # Distribution plot
        self.distribution_plot = DistributionPlot(width=6, height=3)
        features_layout.addWidget(self.distribution_plot)
        
        # Add features tab
        self.tabs.addTab(features_tab, "Feature Exploration")
        
        # Tab 3: Partition comparison
        partition_tab = QWidget()
        partition_layout = QVBoxLayout(partition_tab)
        
        # Controls
        partition_controls_layout = QHBoxLayout()
        self.partition_feature_combo = QComboBox()
        
        partition_controls_layout.addWidget(QLabel("Feature:"))
        partition_controls_layout.addWidget(self.partition_feature_combo)
        
        partition_layout.addLayout(partition_controls_layout)
        
        # Partition comparison plot
        self.partition_plot = PartitionComparisonPlot(title="Feature Distribution Across Partitions")
        partition_layout.addWidget(self.partition_plot)
        
        # Add partition tab
        self.tabs.addTab(partition_tab, "Partition Comparison")
        
        # Tab 4: Correlations
        corr_tab = QWidget()
        corr_layout = QVBoxLayout(corr_tab)
        
        # Feature correlation matrix
        self.correlation_matrix = FeatureCorrelationMatrix(width=8, height=6)
        corr_layout.addWidget(self.correlation_matrix)
        
        # Add correlation tab
        self.tabs.addTab(corr_tab, "Feature Correlations")
        
        # Tab 5: Data samples
        samples_tab = QWidget()
        samples_layout = QVBoxLayout(samples_tab)
        
        # Data table view
        self.table_view = QTableView()
        self.table_model = DataTableModel()
        self.table_view.setModel(self.table_model)
        
        # Configure the table
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table_view.verticalHeader().setVisible(True)
        
        samples_layout.addWidget(self.table_view)
        
        # Add samples tab
        self.tabs.addTab(samples_tab, "Data Samples")
        
        # Connect signals
        self.x_feature_combo.currentTextChanged.connect(self.update_scatter)
        self.y_feature_combo.currentTextChanged.connect(self.update_scatter)
        self.class_column_combo.currentTextChanged.connect(self.update_scatter)
        self.partition_feature_combo.currentTextChanged.connect(self.update_partition_plot)
    
    def set_dataset(self, data, target_column=None):
        """Update all visualizations with a new dataset."""
        # Update dataset statistics
        self.sample_count_label.setText(str(len(data)))
        self.feature_count_label.setText(str(len(data.columns)))
        
        if target_column and target_column in data.columns:
            self.class_count_label.setText(str(data[target_column].nunique()))
        else:
            self.class_count_label.setText("N/A")
            
        missing = data.isnull().sum().sum()
        self.missing_values_label.setText(str(missing))
        
        # Update class balance plot
        if target_column and target_column in data.columns:
            self.class_balance_plot.update_plot(data, target_column)
        
        # Update feature combos
        self.update_feature_combos(data.columns, target_column)
        
        # Update correlation matrix
        self.correlation_matrix.update_correlation(data, target_column)
        
        # Update data table
        sample_data = data.head(100)  # Show first 100 rows
        self.table_model.update_data(sample_data)
        
        # Initial scatter plot
        if len(data.columns) >= 2:
            x_col = data.columns[0]
            y_col = data.columns[1]
            self.update_scatter(x_col, y_col, target_column)
            
            # Initial distribution plot
            self.distribution_plot.plot_distribution(data, x_col, target_column)
    
    def update_feature_combos(self, columns, target_column=None):
        """Update the feature selection dropdown menus."""
        # Block signals to prevent triggering updates during setup
        self.x_feature_combo.blockSignals(True)
        self.y_feature_combo.blockSignals(True)
        self.class_column_combo.blockSignals(True)
        self.partition_feature_combo.blockSignals(True)
        
        # Clear existing items
        self.x_feature_combo.clear()
        self.y_feature_combo.clear()
        self.class_column_combo.clear()
        self.partition_feature_combo.clear()
        
        # Add special "None" option for class selection
        self.class_column_combo.addItem("None")
        
        # Add columns
        for column in columns:
            self.x_feature_combo.addItem(column)
            self.y_feature_combo.addItem(column)
            self.class_column_combo.addItem(column)
            self.partition_feature_combo.addItem(column)
        
        # Set defaults
        if len(columns) >= 2:
            self.x_feature_combo.setCurrentText(columns[0])
            self.y_feature_combo.setCurrentText(columns[1])
            
        if target_column and target_column in columns:
            self.class_column_combo.setCurrentText(target_column)
        
        if len(columns) >= 1:
            self.partition_feature_combo.setCurrentText(columns[0])
        
        # Unblock signals
        self.x_feature_combo.blockSignals(False)
        self.y_feature_combo.blockSignals(False)
        self.class_column_combo.blockSignals(False)
        self.partition_feature_combo.blockSignals(False)
    
    def update_scatter(self, *args):
        """Update the scatter plot with currently selected features."""
        # This method will be called when any of the combo boxes change
        # args are not used directly - we get the values from the combos
        
        x_feature = self.x_feature_combo.currentText()
        y_feature = self.y_feature_combo.currentText()
        class_column = self.class_column_combo.currentText()
        
        if class_column == "None":
            class_column = None
        
        # Update scatter plot if we have data
        if hasattr(self.table_model, 'data') and len(self.table_model.data) > 0:
            if x_feature in self.table_model.data.columns and y_feature in self.table_model.data.columns:
                self.scatter_plot.update_scatter(
                    self.table_model.data, 
                    x_feature, 
                    y_feature, 
                    class_column
                )
                
                # Also update distribution plot for X feature
                self.distribution_plot.plot_distribution(
                    self.table_model.data, 
                    x_feature, 
                    class_column
                )
    
    def update_partition_plot(self, feature=None):
        """Update the partition comparison plot."""
        if feature is None:
            feature = self.partition_feature_combo.currentText()
            
        if feature:
            self.partition_plot.update_plot(feature)
    
    def add_partition_data(self, partition_id, data):
        """Add data for a specific model partition."""
        feature = self.partition_feature_combo.currentText()
        if feature and feature in data.columns:
            self.partition_plot.update_partition_data(partition_id, data, feature) 