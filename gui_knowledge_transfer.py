#!/usr/bin/env python
"""
Module for visualizing knowledge transfer in heterogeneous federated learning.
Provides interactive components for displaying knowledge flow between models.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

import pyqtgraph as pg
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class KnowledgeTransferHeatmap(FigureCanvas):
    """
    A heatmap visualization showing knowledge transfer between models.
    Uses matplotlib for rendering.
    """
    def __init__(self, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.model_names = ["XGBoost", "RandomForest", "LightGBM", "CNN", "Autoencoder"]
        self.initialize_heatmap()
    
    def initialize_heatmap(self):
        """Initialize the heatmap with zeros."""
        # Create a zero matrix for the initial heatmap
        n_models = len(self.model_names)
        self.transfer_matrix = np.zeros((n_models, n_models))
        self.update_heatmap()
    
    def update_heatmap(self, matrix=None):
        """Update the heatmap with new transfer values."""
        if matrix is not None:
            self.transfer_matrix = matrix
        
        self.axes.clear()
        import seaborn as sns
        
        # Create the heatmap
        sns.heatmap(
            self.transfer_matrix,
            annot=True,
            cmap='viridis',
            xticklabels=self.model_names,
            yticklabels=self.model_names,
            ax=self.axes
        )
        
        self.axes.set_title('Knowledge Transfer Between Models')
        self.axes.set_xlabel('Receiving Model')
        self.axes.set_ylabel('Sending Model')
        self.draw()
    
    def update_transfer_value(self, sender_idx, receiver_idx, value):
        """Update a single transfer value in the matrix."""
        if 0 <= sender_idx < len(self.model_names) and 0 <= receiver_idx < len(self.model_names):
            self.transfer_matrix[sender_idx, receiver_idx] = value
            self.update_heatmap()


class KnowledgeFlowNetwork(pg.GraphItem):
    """
    A network graph visualizing knowledge flow between models.
    Uses PyQtGraph for interactive network visualization.
    """
    def __init__(self):
        self.picture = None
        self.scatter = None
        pg.GraphItem.__init__(self)
        
        # Model positions in a circle
        self.n_models = 5
        self.model_names = ["XGBoost", "RandomForest", "LightGBM", "CNN", "Autoencoder"]
        
        # Initialize graph
        self.setup_graph()
    
    def setup_graph(self):
        """Initialize the graph structure."""
        # Calculate node positions in a circle
        theta = np.linspace(0, 2*np.pi, self.n_models, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        self.node_positions = np.column_stack([x, y])
        
        # Create node data
        self.node_data = []
        for i in range(self.n_models):
            self.node_data.append({
                'pos': self.node_positions[i],
                'size': 0.2,
                'label': self.model_names[i],
                'symbolBrush': pg.mkBrush(30, 100, 200, 200)
            })
        
        # Initialize with no edges
        self.edges = np.array([], dtype=np.int32).reshape(0, 2)
        
        # Update graph
        self.update_graph()
    
    def update_graph(self, transfer_matrix=None):
        """Update the graph with new transfer values."""
        if transfer_matrix is not None:
            # Create edges based on transfer matrix
            edges = []
            for i in range(self.n_models):
                for j in range(self.n_models):
                    if i != j and transfer_matrix[i, j] > 0.01:  # Only show significant transfers
                        edges.append([i, j])
            
            if edges:
                self.edges = np.array(edges)
                
                # Adjust node sizes based on received knowledge
                for i in range(self.n_models):
                    received = np.sum(transfer_matrix[:, i])
                    self.node_data[i]['size'] = 0.15 + received * 0.1  # Scale size by received knowledge
        
        # Create graph data
        node_positions = np.array([item['pos'] for item in self.node_data])
        node_sizes = np.array([item['size'] for item in self.node_data])
        
        # Set data
        self.setData(
            pos=node_positions,
            size=node_sizes,
            adj=self.edges,
            symbolBrush=[item['symbolBrush'] for item in self.node_data],
            pxMode=False
        )


class KnowledgeTransferWidget(QWidget):
    """
    Widget for visualizing knowledge transfer between models.
    Includes both heatmap and network graph visualizations.
    """
    def __init__(self):
        super().__init__()
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create a split layout
        h_layout = QHBoxLayout()
        
        # Add the heatmap on the left
        heatmap_group = QGroupBox("Knowledge Transfer Heatmap")
        heatmap_layout = QVBoxLayout(heatmap_group)
        self.heatmap = KnowledgeTransferHeatmap(width=5, height=4)
        heatmap_layout.addWidget(self.heatmap)
        h_layout.addWidget(heatmap_group)
        
        # Add the network graph on the right
        network_group = QGroupBox("Knowledge Flow Network")
        network_layout = QVBoxLayout(network_group)
        network_widget = pg.PlotWidget(title="Model Knowledge Flow")
        
        # Set up the graph
        self.knowledge_graph = KnowledgeFlowNetwork()
        network_widget.addItem(self.knowledge_graph)
        
        # Configure the plot
        network_widget.setAspectLocked(True)
        network_widget.hideAxis('left')
        network_widget.hideAxis('bottom')
        
        network_layout.addWidget(network_widget)
        h_layout.addWidget(network_group)
        
        layout.addLayout(h_layout)
        
        # Add statistics at the bottom
        stats_group = QGroupBox("Knowledge Transfer Statistics")
        stats_layout = QFormLayout(stats_group)
        
        self.total_transfers_label = QLabel("0")
        self.avg_improvement_label = QLabel("0.00%")
        self.most_improved_label = QLabel("--")
        self.knowledge_ratio_label = QLabel("0.00")
        
        stats_layout.addRow("Total Knowledge Transfers:", self.total_transfers_label)
        stats_layout.addRow("Average Performance Improvement:", self.avg_improvement_label)
        stats_layout.addRow("Most Improved Model:", self.most_improved_label)
        stats_layout.addRow("Knowledge Utilization Ratio:", self.knowledge_ratio_label)
        
        layout.addWidget(stats_group)
    
    def update_transfer_data(self, transfer_matrix, improvement_data=None):
        """
        Update the visualization with new transfer data.
        
        Args:
            transfer_matrix: NxN matrix where [i,j] represents 
                             knowledge transferred from model i to model j
            improvement_data: Dictionary with improvement metrics (optional)
        """
        # Update the heatmap
        self.heatmap.update_heatmap(transfer_matrix)
        
        # Update the network graph
        self.knowledge_graph.update_graph(transfer_matrix)
        
        # Update statistics if available
        if improvement_data:
            self.total_transfers_label.setText(str(improvement_data.get('total_transfers', 0)))
            self.avg_improvement_label.setText(f"{improvement_data.get('avg_improvement', 0.0):.2f}%")
            self.most_improved_label.setText(improvement_data.get('most_improved', '--'))
            self.knowledge_ratio_label.setText(f"{improvement_data.get('utilization_ratio', 0.0):.2f}")


class BeforeAfterComparisonWidget(QWidget):
    """Widget for visualizing before/after performance comparison."""
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget(title="Before vs After Knowledge Transfer")
        self.plot_widget.setLabel('left', 'Performance (%)')
        self.plot_widget.setLabel('bottom', 'Model')
        self.plot_widget.addLegend()
        
        # Set up bar chart
        self.x_axis = np.arange(5)  # 5 models
        self.before_bars = pg.BarGraphItem(x=self.x_axis-0.15, width=0.3, brush='b', name='Before')
        self.after_bars = pg.BarGraphItem(x=self.x_axis+0.15, width=0.3, brush='g', name='After')
        
        self.plot_widget.addItem(self.before_bars)
        self.plot_widget.addItem(self.after_bars)
        
        # Set axis
        axis = self.plot_widget.getAxis('bottom')
        axis.setTicks([list(zip(range(5), ["XGBoost", "RF", "LGBM", "CNN", "AE"]))])
        
        layout.addWidget(self.plot_widget)
        
        # Add details widget
        details_group = QGroupBox("Transfer Details")
        details_layout = QFormLayout(details_group)
        
        self.improvement_percent_label = QLabel("--")
        self.convergence_rounds_label = QLabel("--")
        self.transfer_efficiency_label = QLabel("--")
        
        details_layout.addRow("Average Improvement:", self.improvement_percent_label)
        details_layout.addRow("Rounds to Converge:", self.convergence_rounds_label)
        details_layout.addRow("Transfer Efficiency:", self.transfer_efficiency_label)
        
        layout.addWidget(details_group)
    
    def update_comparison(self, before_values, after_values, details=None):
        """
        Update the comparison chart with new values.
        
        Args:
            before_values: List of performance values before knowledge transfer
            after_values: List of performance values after knowledge transfer
            details: Dictionary with additional details (optional)
        """
        # Update bar heights
        self.before_bars.setOpts(height=before_values)
        self.after_bars.setOpts(height=after_values)
        
        # Update details if available
        if details:
            self.improvement_percent_label.setText(f"{details.get('improvement_pct', 0.0):.2f}%")
            self.convergence_rounds_label.setText(str(details.get('convergence_rounds', '--')))
            self.transfer_efficiency_label.setText(f"{details.get('transfer_efficiency', 0.0):.2f}") 