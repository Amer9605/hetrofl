#!/usr/bin/env python
"""
GUI for the HETROFL (Heterogeneous Federated Learning) system.
This module creates a desktop application that visualizes the
federated learning process with separate windows for global
and local models.

Enhanced with modern Material Design styling and comprehensive testing capabilities.
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplashScreen, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox, QCheckBox,
    QTabWidget, QGroupBox, QFormLayout, QMessageBox, QFileDialog,
    QSplitter, QStatusBar, QProgressBar, QLineEdit, QScrollArea, QTextEdit,
    QSizePolicy, QMenuBar, QMenu, QToolBar
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QObject, QSize
from PySide6.QtGui import QPixmap, QIcon, QFont, QAction, QColor, QKeySequence

import pyqtgraph as pg
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data.data_loader import DataLoader
from local_models.xgboost_model import XGBoostModel
from local_models.random_forest_model import RandomForestModel
from local_models.lightgbm_model import LightGBMModel
from local_models.cnn_model import CNNModel
from local_models.autoencoder_model import AutoencoderModel
from global_model.federated_learning import HeterogeneousFederatedLearning
from visualization.learning_visualizer import LearningVisualizer
from config.config import (
    CLIENT_MODELS, 
    DATA_DISTRIBUTIONS, 
    CUMULATIVE_LEARNING,
    LOCAL_EPOCHS,
    DATASET_SAMPLE_SIZE
)

# Import new GUI components
from gui.gui_themes import ThemeManager
from gui.gui_test_interface import ModelTestPanel
from gui.gui_dataset_manager import DatasetManager

# Set up PyQtGraph configuration
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    started = Signal()
    finished = Signal()
    error = Signal(str)
    progress = Signal(int, str)
    result = Signal(object)
    status = Signal(str)
    metrics_update = Signal(dict)
    local_update = Signal(int, dict)  # client_id, metrics
    global_update = Signal(dict)  # global metrics


class FederatedLearningWorker(QThread):
    """
    Worker thread for running the federated learning process.
    """
    def __init__(self, fl_system, params):
        super().__init__()
        self.fl_system = fl_system
        self.params = params
        self.signals = WorkerSignals()
        self.stop_requested = False
    
    def run(self):
        """Main execution method for the federated learning thread."""
        try:
            self.signals.started.emit()
            self.signals.status.emit("Initializing federated learning system...")
            
            # Initialize system
            data_distribution = self.params.get("data_distribution", "iid")
            load_previous = self.params.get("load_previous_models", False)
            
            self.fl_system.initialize_system(
                data_distribution=data_distribution,
                load_previous_models=load_previous
            )
            self.signals.status.emit("System initialized successfully.")
            
            # Evaluate initial models
            self.signals.status.emit("Evaluating initial models...")
            initial_metrics = self.fl_system.evaluate_local_models(stage_name="initial")
            self.signals.metrics_update.emit({"initial": initial_metrics})
            
            # Emit initial metrics for each local model
            for client_id, metrics in initial_metrics.items():
                self.signals.local_update.emit(client_id, {"initial": metrics})
            
            # Run federated learning rounds
            rounds = self.params.get("communication_rounds", 5)
            hp_tuning = self.params.get("hyperparameter_tuning", False)
            local_epochs = self.params.get("local_epochs", LOCAL_EPOCHS)
            
            for round_num in range(1, rounds + 1):
                if self.stop_requested:
                    self.signals.status.emit("Federated learning process stopped.")
                    break
                
                self.signals.status.emit(f"Running federated learning round {round_num}/{rounds}...")
                self.signals.progress.emit(round_num * 100 // rounds, f"Round {round_num}/{rounds}")
                
                # Run a single FL round
                round_params = self.fl_system.run_federated_learning_round(
                    round_num=round_num,
                    hyperparameter_tuning=hp_tuning,
                    data_distribution=data_distribution,
                    local_epochs=local_epochs
                )
                
                # Get metrics for this round
                round_metrics = self.fl_system.evaluate_round(round_num)
                
                # Update metrics for each local model
                for client_id, client_metrics in round_metrics["local_metrics"].items():
                    self.signals.local_update.emit(client_id, {f"round_{round_num}": client_metrics})
                
                # Update global model metrics
                self.signals.global_update.emit({f"round_{round_num}": round_metrics["global_metrics"]})
                
                # Send overall metrics update
                self.signals.metrics_update.emit({f"round_{round_num}": round_metrics})
            
            # Final evaluation
            if not self.stop_requested:
                self.signals.status.emit("Performing final evaluation...")
                final_metrics = self.fl_system.evaluate_local_models(stage_name="final")
                self.signals.metrics_update.emit({"final": final_metrics})
                
                # Emit final metrics for each local model
                for client_id, metrics in final_metrics.items():
                    self.signals.local_update.emit(client_id, {"final": metrics})
                
                # Save models if requested
                if self.params.get("save_models", False):
                    self.signals.status.emit("Saving trained models...")
                    self.fl_system.save_models()
                
                self.signals.status.emit("Federated learning process completed successfully.")
            
            self.signals.finished.emit()
            
        except Exception as e:
            import traceback
            self.signals.error.emit(f"Error in federated learning process: {str(e)}\n{traceback.format_exc()}")
    
    def stop(self):
        """Request the worker to stop processing."""
        self.stop_requested = True
        self.signals.status.emit("Stopping federated learning process...")


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in Qt."""
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class LocalModelWindow(QMainWindow):
    """Window for visualizing a local model's performance and training."""
    def __init__(self, client_id, model_name, theme_manager=None):
        super().__init__()
        self.client_id = client_id
        self.model_name = model_name
        self.metrics_history = {}
        self.theme_manager = theme_manager
        self.model_instance = None  # Will store the actual model for testing
        
        self.setWindowTitle(f"Local Model {client_id+1}: {model_name}")
        self.setMinimumSize(1000, 700)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Tab 1: Overview and metrics
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        
        # Model info section
        model_info_group = QGroupBox("Model Information")
        model_info_layout = QFormLayout(model_info_group)
        self.model_name_label = QLabel(model_name)
        self.model_type_label = QLabel("")
        self.training_status_label = QLabel("Idle")
        model_info_layout.addRow("Model Name:", self.model_name_label)
        model_info_layout.addRow("Model Type:", self.model_type_label)
        model_info_layout.addRow("Status:", self.training_status_label)
        overview_layout.addWidget(model_info_group)
        
        # Performance metrics section
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QFormLayout(metrics_group)
        self.accuracy_label = QLabel("--")
        self.f1_score_label = QLabel("--")
        self.precision_label = QLabel("--")
        self.recall_label = QLabel("--")
        metrics_layout.addRow("Accuracy:", self.accuracy_label)
        metrics_layout.addRow("F1 Score:", self.f1_score_label)
        metrics_layout.addRow("Precision:", self.precision_label)
        metrics_layout.addRow("Recall:", self.recall_label)
        overview_layout.addWidget(metrics_group)
        
        # Knowledge transfer section
        transfer_group = QGroupBox("Knowledge Transfer Impact")
        transfer_layout = QFormLayout(transfer_group)
        self.pre_transfer_label = QLabel("--")
        self.post_transfer_label = QLabel("--")
        self.improvement_label = QLabel("--")
        transfer_layout.addRow("Performance before transfer:", self.pre_transfer_label)
        transfer_layout.addRow("Performance after transfer:", self.post_transfer_label)
        transfer_layout.addRow("Improvement:", self.improvement_label)
        overview_layout.addWidget(transfer_group)
        
        # Actions section
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)
        
        self.train_button = QPushButton("Train Model")
        self.export_button = QPushButton("Export Results")
        actions_layout.addWidget(self.train_button)
        actions_layout.addWidget(self.export_button)
        overview_layout.addWidget(actions_group)
        
        # Tab 2: Real-time plots
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        
        # Create plots
        self.accuracy_plot = pg.PlotWidget(title="Accuracy Over Time")
        self.accuracy_plot.showGrid(x=True, y=True)
        self.accuracy_plot.addLegend()
        self.accuracy_curve = self.accuracy_plot.plot(name="Accuracy", pen=pg.mkPen('b', width=2), symbol='o', symbolSize=6, symbolBrush=pg.mkBrush('b'))
        
        self.loss_plot = pg.PlotWidget(title="Loss Over Time")
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.setLabel('bottom', 'Round')
        self.loss_plot.showGrid(x=True, y=True)
        self.loss_plot.addLegend()
        self.loss_curve = self.loss_plot.plot(name="Loss", pen=pg.mkPen('r', width=2), symbol='x', symbolSize=6, symbolBrush=pg.mkBrush('r'))
        
        # Matplotlib canvas for confusion matrix
        self.confusion_matrix_canvas = MatplotlibCanvas(width=5, height=4)
        
        plot_layout.addWidget(self.accuracy_plot)
        plot_layout.addWidget(self.loss_plot)
        plot_layout.addWidget(self.confusion_matrix_canvas)
        
        # Tab 3: Feature importance and model details
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        
        # Feature importance plot (Matplotlib)
        self.feature_importance_canvas = MatplotlibCanvas(width=6, height=4)
        details_layout.addWidget(QLabel("Feature Importance"))
        details_layout.addWidget(self.feature_importance_canvas)
        
        # Model parameters section
        params_group = QGroupBox("Model Parameters")
        params_layout = QVBoxLayout(params_group)
        self.params_label = QLabel("No parameters available yet")
        params_layout.addWidget(self.params_label)
        details_layout.addWidget(params_group)
        
        # Tab 4: Model Testing
        test_tab = QWidget()
        test_layout = QVBoxLayout(test_tab)
        
        # Create test panel
        self.test_panel = ModelTestPanel(model=None, model_name=model_name)
        test_layout.addWidget(self.test_panel)
        
        # Add tabs to tab widget
        self.tabs.addTab(overview_tab, "Overview")
        self.tabs.addTab(plot_tab, "Real-time Plots")
        self.tabs.addTab(details_tab, "Model Details")
        self.tabs.addTab(test_tab, "Model Testing")
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
    
    def set_model_instance(self, model):
        """Set the model instance for testing."""
        self.model_instance = model
        if hasattr(self, 'test_panel'):
            self.test_panel.set_model(model, self.model_name)
    
    @Slot(dict)
    def update_metrics(self, stage_metrics):
        """Update the model metrics from training."""
        stage = list(stage_metrics.keys())[0]
        metrics = stage_metrics[stage]
        
        # Store metrics history
        self.metrics_history[stage] = metrics
        
        # Update UI elements with the latest metrics
        self.accuracy_label.setText(f"{metrics.get('accuracy', 0):.4f}")
        # Use weighted F1 if available
        f1_val = metrics.get('f1_score', metrics.get('f1_weighted', 0))
        self.f1_score_label.setText(f"{f1_val:.4f}")
        prec = metrics.get('precision_weighted', metrics.get('precision_macro', 0))
        self.precision_label.setText(f"{prec:.4f}")
        rec = metrics.get('recall_weighted', metrics.get('recall_macro', 0))
        self.recall_label.setText(f"{rec:.4f}")
        
        # Compute improvement between rounds
        if stage.startswith('round_'):
            rn = int(stage.split('_')[1])
            curr = metrics.get('accuracy', 0)
            prev_metrics = self.metrics_history.get(f'round_{rn-1}', {})
            prev = prev_metrics.get('accuracy', 0)
            self.pre_transfer_label.setText(f"{prev:.4f}" if prev_metrics else "--")
            self.post_transfer_label.setText(f"{curr:.4f}")
            imp_pct = (curr - prev)*100 if prev>0 else 0
            self.improvement_label.setText(f"{imp_pct:.2f}%")
        elif stage == 'initial':
            # clear only on initial
            self.pre_transfer_label.setText("--")
            self.post_transfer_label.setText("--")
            self.improvement_label.setText("--")
        
        # Update status
        self.training_status_label.setText(f"Updated - {stage}")
        self.statusBar.showMessage(f"Metrics updated: {stage}")
        
        # Update plots
        self.update_plots()
    
    def update_plots(self):
        """Update all plots with the latest metrics."""
        # Get all rounds from history
        rounds = []
        accuracies = []
        losses = []
        cms = []
        class_names = []
        
        for stage, metrics in sorted(self.metrics_history.items()):
            if 'round_' in stage:
                round_num = int(stage.split('_')[1])
                rounds.append(round_num)
                accuracies.append(metrics.get('accuracy', 0))
                losses.append(metrics.get('loss', 0) if 'loss' in metrics else 1.0 - metrics.get('accuracy', 0))
                # collect confusion matrix
                if 'confusion_matrix' in metrics:
                    cms.append(metrics['confusion_matrix'])
                    class_names = metrics.get('class_names', class_names)
        
        # Update line plots if we have data
        if rounds:
            self.accuracy_curve.setData(rounds, accuracies)
            self.loss_curve.setData(rounds, losses)
        
        # show confusion matrix per round in a new window
        if cms:
            # overlay class matrices sequentially: for simplicity show last
            cm = cms[-1]
            self.confusion_matrix_canvas.axes.clear()
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        ax=self.confusion_matrix_canvas.axes)
            self.confusion_matrix_canvas.draw()
        
        # Update feature importance if available
        if 'feature_importance' in self.metrics_history.get(list(self.metrics_history.keys())[-1], {}):
            self.update_feature_importance()
    
    def update_confusion_matrix(self):
        """Update the confusion matrix plot."""
        latest_stage = list(self.metrics_history.keys())[-1]
        metrics = self.metrics_history[latest_stage]
        
        if 'confusion_matrix' in metrics and 'class_names' in metrics:
            cm = metrics['confusion_matrix']
            class_names = metrics['class_names']
            
            self.confusion_matrix_canvas.axes.clear()
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, 
                        yticklabels=class_names, 
                        ax=self.confusion_matrix_canvas.axes)
            
            self.confusion_matrix_canvas.axes.set_title('Confusion Matrix')
            self.confusion_matrix_canvas.axes.set_xlabel('Predicted Label')
            self.confusion_matrix_canvas.axes.set_ylabel('True Label')
            self.confusion_matrix_canvas.draw()
    
    def update_feature_importance(self):
        """Update the feature importance plot."""
        latest_stage = list(self.metrics_history.keys())[-1]
        metrics = self.metrics_history[latest_stage]
        
        if 'feature_importance' in metrics and 'feature_names' in metrics:
            importances = metrics['feature_importance']
            feature_names = metrics['feature_names']
            
            # Get top N features
            N = min(20, len(importances))
            indices = np.argsort(importances)[-N:]
            
            self.feature_importance_canvas.axes.clear()
            self.feature_importance_canvas.axes.barh(range(N), 
                                                   [importances[i] for i in indices],
                                                   align='center')
            self.feature_importance_canvas.axes.set_yticks(range(N))
            self.feature_importance_canvas.axes.set_yticklabels([feature_names[i] for i in indices])
            self.feature_importance_canvas.axes.set_title('Feature Importance')
            self.feature_importance_canvas.axes.set_xlabel('Importance')
            self.feature_importance_canvas.draw()


class GlobalModelWindow(QMainWindow):
    """Main window for the global model and system orchestration."""
    def __init__(self):
        super().__init__()
        # Tracking local accuracies and transfer improvements
        self.local_metrics_history = {}
        self.transfer_rounds = []
        self.transfer_vals = []
        self.setWindowTitle("HETROFL - Global Model Dashboard")
        self.setMinimumSize(1200, 900)
        
        # Store local model windows
        self.local_windows = {}
        
        # Initialize theme manager
        self.theme_manager = ThemeManager()
        
        # Setup menu bar and toolbar
        self.setup_menu_bar()
        self.setup_toolbar()
        
        # Initialize FL system
        self.init_fl_system()
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Tab 1: Configuration & Control
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Configuration section
        config_group = QGroupBox("System Configuration")
        form_layout = QFormLayout(config_group)
        
        self.data_dist_combo = QComboBox()
        for dist in DATA_DISTRIBUTIONS:
            self.data_dist_combo.addItem(dist)
        
        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(1, 100)
        self.rounds_spin.setValue(5)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 50)
        self.epochs_spin.setValue(LOCAL_EPOCHS)
        
        self.hp_tuning_check = QCheckBox()
        self.hp_tuning_check.setChecked(True)
        
        self.cumulative_check = QCheckBox()
        self.cumulative_check.setChecked(CUMULATIVE_LEARNING)
        
        self.save_models_check = QCheckBox()
        self.save_models_check.setChecked(True)
        
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(-1, 1000000)
        self.sample_size_spin.setValue(DATASET_SAMPLE_SIZE)
        self.sample_size_spin.setSpecialValueText("All")
        
        form_layout.addRow("Data Distribution:", self.data_dist_combo)
        form_layout.addRow("Communication Rounds:", self.rounds_spin)
        form_layout.addRow("Local Epochs:", self.epochs_spin)
        form_layout.addRow("Sample Size:", self.sample_size_spin)
        form_layout.addRow("Hyperparameter Tuning:", self.hp_tuning_check)
        form_layout.addRow("Cumulative Learning:", self.cumulative_check)
        form_layout.addRow("Save Models:", self.save_models_check)
        
        config_layout.addWidget(config_group)
        
        # Controls section
        controls_group = QGroupBox("System Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.start_button = QPushButton("Start Training")
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setEnabled(False)
        self.view_local_button = QPushButton("View Local Models")
        self.export_button = QPushButton("Export Results")
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.view_local_button)
        controls_layout.addWidget(self.export_button)
        
        config_layout.addWidget(controls_group)
        
        # Progress section
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        config_layout.addWidget(progress_group)
        
        # Tab 2: Global Performance and Visualization
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # Create splitter for metrics and plots
        splitter = QSplitter(Qt.Horizontal)
        
        # Metrics panel
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        metrics_group = QGroupBox("Global Performance Metrics")
        metrics_form = QFormLayout(metrics_group)
        
        self.global_acc_label = QLabel("--")
        self.global_f1_label = QLabel("--")
        self.global_loss_label = QLabel("--")
        self.avg_client_acc_label = QLabel("--")
        
        metrics_form.addRow("Global Accuracy:", self.global_acc_label)
        metrics_form.addRow("Global F1 Score:", self.global_f1_label)
        metrics_form.addRow("Global Loss:", self.global_loss_label)
        metrics_form.addRow("Avg Client Accuracy:", self.avg_client_acc_label)
        
        metrics_layout.addWidget(metrics_group)
        
        # Knowledge transfer metrics
        transfer_group = QGroupBox("Knowledge Transfer Effectiveness")
        transfer_form = QFormLayout(transfer_group)
        
        self.avg_improvement_label = QLabel("--")
        self.best_improvement_label = QLabel("--")
        self.worst_improvement_label = QLabel("--")
        
        transfer_form.addRow("Average Improvement:", self.avg_improvement_label)
        transfer_form.addRow("Best Improvement:", self.best_improvement_label)
        transfer_form.addRow("Worst Improvement:", self.worst_improvement_label)
        
        metrics_layout.addWidget(transfer_group)
        
        # Plots panel
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        
        # Global accuracy plot
        self.global_acc_plot = pg.PlotWidget(title="Global vs Local Accuracy")
        self.global_acc_plot.showGrid(x=True, y=True)
        self.global_acc_plot.setLabel('left', 'Accuracy')
        self.global_acc_plot.setLabel('bottom', 'Round')
        self.global_acc_curve = self.global_acc_plot.plot(name="Global", pen=pg.mkPen('r', width=3), symbol='o', symbolSize=8, symbolBrush=pg.mkBrush('r'))
        # Add legend and initialize client curves
        self.global_acc_plot.addLegend()
        self.local_acc_curves = {}
        for idx in range(len(CLIENT_MODELS)):
            color = pg.intColor(idx, hues=len(CLIENT_MODELS), alpha=150)
            curve = self.global_acc_plot.plot(name=f"Client {idx+1}", pen=pg.mkPen(color=color, width=2), symbol='t', symbolSize=6, symbolBrush=pg.mkBrush(color))
            self.local_acc_curves[idx] = curve
        
        plots_layout.addWidget(self.global_acc_plot)
        
        # Knowledge transfer impact plot
        self.transfer_plot = pg.PlotWidget(title="Knowledge Transfer Impact")
        self.transfer_plot.setLabel('left', 'Improvement')
        self.transfer_plot.setLabel('bottom', 'Round')
        self.transfer_curve = self.transfer_plot.plot(pen=pg.mkPen('g', width=2))
        
        plots_layout.addWidget(self.transfer_plot)
        
        # Global confusion matrix
        self.global_confusion_canvas = MatplotlibCanvas(width=6, height=4)
        plots_layout.addWidget(QLabel("Global Confusion Matrix"))
        plots_layout.addWidget(self.global_confusion_canvas)
        
        # Add widgets to splitter
        splitter.addWidget(metrics_widget)
        splitter.addWidget(plots_widget)
        splitter.setSizes([300, 700])
        viz_layout.addWidget(splitter)
        
        # Tab 3: Communication Log
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Add tabs to tab widget
        self.tabs.addTab(config_tab, "Configuration & Control")
        self.tabs.addTab(viz_tab, "Global Performance")
        self.tabs.addTab(log_tab, "Communication Log")
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Connect signals
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        self.view_local_button.clicked.connect(self.show_local_models)
        self.export_button.clicked.connect(self.export_results)
        
        # Worker thread
        self.worker = None
        
        # Apply initial theme
        self.apply_theme()
    
    def setup_menu_bar(self):
        """Setup the menu bar with modern options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # New experiment action
        new_action = QAction('&New Experiment', self)
        new_action.setShortcut(QKeySequence.New)
        new_action.setStatusTip('Start a new federated learning experiment')
        new_action.triggered.connect(self.new_experiment)
        file_menu.addAction(new_action)
        
        # Load experiment action
        load_action = QAction('&Load Experiment', self)
        load_action.setShortcut(QKeySequence.Open)
        load_action.setStatusTip('Load a previous experiment')
        load_action.triggered.connect(self.load_experiment)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        # Export results action
        export_action = QAction('&Export Results', self)
        export_action.setShortcut(QKeySequence('Ctrl+E'))
        export_action.setStatusTip('Export experiment results')
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Theme submenu
        theme_menu = view_menu.addMenu('&Theme')
        
        # Light theme action
        light_action = QAction('&Light', self)
        light_action.setCheckable(True)
        light_action.setChecked(self.theme_manager.current_theme == 'light')
        light_action.triggered.connect(lambda: self.change_theme('light'))
        theme_menu.addAction(light_action)
        
        # Dark theme action
        dark_action = QAction('&Dark', self)
        dark_action.setCheckable(True)
        dark_action.setChecked(self.theme_manager.current_theme == 'dark')
        dark_action.triggered.connect(lambda: self.change_theme('dark'))
        theme_menu.addAction(dark_action)
        
        # Auto theme action
        auto_action = QAction('&Auto', self)
        auto_action.setCheckable(True)
        auto_action.setChecked(self.theme_manager.current_theme == 'auto')
        auto_action.triggered.connect(lambda: self.change_theme('auto'))
        theme_menu.addAction(auto_action)
        
        view_menu.addSeparator()
        
        # Show local models action
        show_local_action = QAction('Show &Local Models', self)
        show_local_action.setShortcut(QKeySequence('Ctrl+L'))
        show_local_action.setStatusTip('Show all local model windows')
        show_local_action.triggered.connect(self.show_local_models)
        view_menu.addAction(show_local_action)
        
        # Dataset manager action
        dataset_action = QAction('&Dataset Manager', self)
        dataset_action.setShortcut(QKeySequence('Ctrl+D'))
        dataset_action.setStatusTip('Open dataset manager')
        dataset_action.triggered.connect(self.show_dataset_manager)
        view_menu.addAction(dataset_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        # About action
        about_action = QAction('&About', self)
        about_action.setStatusTip('About HETROFL')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """Setup the toolbar with quick actions."""
        toolbar = self.addToolBar('Main')
        toolbar.setMovable(False)
        
        # Start training action
        start_action = QAction('Start Training', self)
        start_action.setStatusTip('Start federated learning training')
        start_action.triggered.connect(self.start_training)
        toolbar.addAction(start_action)
        
        # Stop training action
        stop_action = QAction('Stop Training', self)
        stop_action.setStatusTip('Stop federated learning training')
        stop_action.triggered.connect(self.stop_training)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        # View local models action
        local_action = QAction('Local Models', self)
        local_action.setStatusTip('Show local model windows')
        local_action.triggered.connect(self.show_local_models)
        toolbar.addAction(local_action)
        
        # Dataset manager action
        dataset_action = QAction('Dataset Manager', self)
        dataset_action.setStatusTip('Open dataset manager')
        dataset_action.triggered.connect(self.show_dataset_manager)
        toolbar.addAction(dataset_action)
        
        toolbar.addSeparator()
        
        # Theme toggle action
        theme_action = QAction('Toggle Theme', self)
        theme_action.setStatusTip('Toggle between light and dark theme')
        theme_action.triggered.connect(lambda: self.theme_manager.toggle_theme(QApplication.instance()))
        toolbar.addAction(theme_action)
    
    def apply_theme(self):
        """Apply the current theme to the application."""
        app = QApplication.instance()
        if app:
            self.theme_manager.apply_theme(app)
    
    def change_theme(self, theme_name):
        """Change to a specific theme."""
        app = QApplication.instance()
        if app:
            self.theme_manager.apply_theme(app, theme_name)
    
    def new_experiment(self):
        """Start a new experiment."""
        # Reset any existing state
        self.local_metrics_history = {}
        self.transfer_rounds = []
        self.transfer_vals = []
        
        # Clear plots
        self.global_acc_curve.setData([], [])
        self.transfer_curve.setData([], [])
        for curve in self.local_acc_curves.values():
            curve.setData([], [])
        
        # Reset labels
        self.global_acc_label.setText("--")
        self.global_f1_label.setText("--")
        self.global_loss_label.setText("--")
        self.avg_client_acc_label.setText("--")
        
        self.log_message("New experiment started")
    
    def load_experiment(self):
        """Load a previous experiment."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Experiment", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            # Implementation would load experiment state
            self.log_message(f"Loading experiment from {file_path}")
            QMessageBox.information(self, "Info", "Experiment loading not yet implemented")
    
    def show_dataset_manager(self):
        """Show the dataset manager window."""
        if not hasattr(self, 'dataset_manager_window'):
            self.dataset_manager_window = DatasetManager()
            self.dataset_manager_window.setWindowTitle("HETROFL - Dataset Manager")
            
        self.dataset_manager_window.show()
        self.dataset_manager_window.activateWindow()
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>HETROFL</h2>
        <p><b>Heterogeneous Federated Learning System</b></p>
        <p>Version 2.0</p>
        <p>A comprehensive federated learning framework supporting multiple model types 
        with advanced knowledge distillation and transfer learning capabilities.</p>
        <p><b>Features:</b></p>
        <ul>
        <li>Multiple ML model support (XGBoost, Random Forest, LightGBM, CNN, Autoencoder)</li>
        <li>Advanced knowledge distillation</li>
        <li>Real-time visualization</li>
        <li>Comprehensive model testing</li>
        <li>Modern Material Design interface</li>
        </ul>
        """
        QMessageBox.about(self, "About HETROFL", about_text)
    
    def init_fl_system(self):
        """Initialize the federated learning system."""
        try:
            # Initialize data loader
            self.data_loader = DataLoader(sample_size=-1)
            
            # Initialize local model classes
            self.local_model_classes = {
                "xgboost": XGBoostModel,
                "random_forest": RandomForestModel,
                "lightgbm": LightGBMModel,
                "cnn": CNNModel,
                "autoencoder": AutoencoderModel
            }
            
            # Initialize federated learning system
            self.fl_system = HeterogeneousFederatedLearning(
                data_loader=self.data_loader,
                local_model_classes=self.local_model_classes,
                experiment_name=f"gui_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize FL system: {str(e)}")
    
    def start_training(self):
        """Start the federated learning training process."""
        try:
            # Get parameters from UI
            params = {
                "data_distribution": self.data_dist_combo.currentText(),
                "communication_rounds": self.rounds_spin.value(),
                "local_epochs": self.epochs_spin.value(),
                "hyperparameter_tuning": self.hp_tuning_check.isChecked(),
                "load_previous_models": self.cumulative_check.isChecked(),
                "save_models": self.save_models_check.isChecked(),
                "sample_size": self.sample_size_spin.value()
            }
            
            # Update data loader's sample size
            self.data_loader.sample_size = params["sample_size"]
            
            # Show all local model windows
            self.show_local_models()
            
            # Create and start worker thread
            self.worker = FederatedLearningWorker(self.fl_system, params)
            
            # Connect signals from worker
            self.worker.signals.started.connect(self.on_training_started)
            self.worker.signals.finished.connect(self.on_training_finished)
            self.worker.signals.error.connect(self.on_training_error)
            self.worker.signals.progress.connect(self.update_progress)
            self.worker.signals.status.connect(self.update_status)
            self.worker.signals.metrics_update.connect(self.update_metrics)
            self.worker.signals.local_update.connect(self.update_local_model)
            self.worker.signals.global_update.connect(self.update_global_model)
            
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"Failed to start training: {str(e)}")
    
    def stop_training(self):
        """Stop the federated learning training process."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.update_status("Stopping training... Please wait.")
    
    def on_training_started(self):
        """Handle the training start event."""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.data_dist_combo.setEnabled(False)
        self.rounds_spin.setEnabled(False)
        self.epochs_spin.setEnabled(False)
        self.hp_tuning_check.setEnabled(False)
        self.cumulative_check.setEnabled(False)
        self.save_models_check.setEnabled(False)
        self.sample_size_spin.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.status_label.setText("Training in progress...")
        self.log_message("Training started")
    
    def on_training_finished(self):
        """Handle the training completion event."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.data_dist_combo.setEnabled(True)
        self.rounds_spin.setEnabled(True)
        self.epochs_spin.setEnabled(True)
        self.hp_tuning_check.setEnabled(True)
        self.cumulative_check.setEnabled(True)
        self.save_models_check.setEnabled(True)
        self.sample_size_spin.setEnabled(True)
        
        self.progress_bar.setValue(100)
        self.status_label.setText("Training completed")
        self.log_message("Training completed")
    
    def on_training_error(self, error_message):
        """Handle training errors."""
        self.on_training_finished()  # Reset UI
        self.status_label.setText("Training failed")
        self.log_message(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Training Error", error_message)
    
    def update_progress(self, progress, message):
        """Update the progress bar and message."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def update_status(self, status):
        """Update status message and log it."""
        self.status_bar.showMessage(status)
        self.log_message(status)
    
    def log_message(self, message):
        """Add a message to the log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def update_metrics(self, metrics_dict):
        """Update overall system metrics."""
        for stage, metrics in metrics_dict.items():
            self.log_message(f"Received metrics for {stage}")
            # Determine local metrics dict
            local_m = None
            if isinstance(metrics, dict):
                if 'local_metrics' in metrics:
                    local_m = metrics['local_metrics']
                elif metrics and all(isinstance(k, int) for k in metrics.keys()):
                    local_m = metrics
            # Update average client accuracy
            if local_m is not None:
                if local_m:
                    avg_acc = sum(m.get('accuracy', 0) for m in local_m.values()) / len(local_m)
                    self.avg_client_acc_label.setText(f"{avg_acc:.4f}")
                    # Compute transfer improvements for rounds
                    if stage.startswith('round_'):
                        round_num = int(stage.split('_')[1])
                        improvements = []
                        for cid, m in local_m.items():
                            prev = self.local_metrics_history.get(cid, {}).get(round_num-1, {})
                            prev_acc = prev.get('accuracy', 0)
                            curr_acc = m.get('accuracy', 0)
                            improvements.append(curr_acc - prev_acc)
                        if improvements:
                            avg_imp = sum(improvements)/len(improvements)
                            best_imp = max(improvements)
                            worst_imp = min(improvements)
                            # update labels as percentages
                            self.avg_improvement_label.setText(f"{avg_imp*100:.2f}%")
                            self.best_improvement_label.setText(f"{best_imp*100:.2f}%")
                            self.worst_improvement_label.setText(f"{worst_imp*100:.2f}%")
                            # update transfer plot
                            self.transfer_rounds.append(round_num)
                            self.transfer_vals.append(avg_imp*100)
                            self.transfer_curve.setData(self.transfer_rounds, self.transfer_vals)
                        # store current for next round
                        for cid, m in local_m.items():
                            self.local_metrics_history.setdefault(cid, {})[round_num] = m
                else:
                    self.avg_client_acc_label.setText("--")
            else:
                self.avg_client_acc_label.setText("--")
    
    def update_local_model(self, client_id, metrics):
        """Update a specific local model with new metrics."""
        if client_id in self.local_windows:
            self.local_windows[client_id].update_metrics(metrics)
            
            # Update model instance for testing if available
            if hasattr(self.fl_system, 'local_models') and client_id < len(self.fl_system.local_models):
                model_instance = self.fl_system.local_models[client_id]
                self.local_windows[client_id].set_model_instance(model_instance)
            
            # Also update the global vs local accuracy plot for this client
            stage = list(metrics.keys())[0]
            if 'round_' in stage:
                round_num = int(stage.split('_')[1])
                m = metrics[stage]
                acc = m.get('accuracy', 0)
                curve = self.local_acc_curves.get(client_id)
                if curve:
                    x_data, y_data = curve.getData()
                    x = list(x_data) if x_data is not None else []
                    y = list(y_data) if y_data is not None else []
                    x.append(round_num)
                    y.append(acc)
                    curve.setData(x, y)
    
    def update_global_model(self, metrics):
        """Update the global model metrics."""
        stage = list(metrics.keys())[0]  # e.g., "round_1"
        stage_metrics = metrics[stage]
        
        # Update global metrics display
        acc_val = stage_metrics.get('accuracy', 0)
        self.global_acc_label.setText(f"{acc_val:.4f}")
        f1_val = stage_metrics.get('f1_score', stage_metrics.get('f1_weighted', 0))
        self.global_f1_label.setText(f"{f1_val:.4f}")
        loss_val = stage_metrics.get('loss', 0)
        self.global_loss_label.setText(f"{loss_val:.4f}")
        
        # Add to global accuracy plot if it's a round
        if 'round_' in stage:
            round_num = int(stage.split('_')[1])
            
            # Update global curve with new point
            x_data, y_data = self.global_acc_curve.getData()
            x_data = list(x_data) if x_data is not None else []
            y_data = list(y_data) if y_data is not None else []
            x_data.append(round_num)
            y_data.append(stage_metrics['accuracy'])
            self.global_acc_curve.setData(x_data, y_data)
        
        # Update global confusion matrix if available
        if 'confusion_matrix' in stage_metrics:
            import seaborn as sns
            cm = stage_metrics['confusion_matrix']
            class_names = stage_metrics.get('class_names', [])
            self.global_confusion_canvas.axes.clear()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        ax=self.global_confusion_canvas.axes)
            self.global_confusion_canvas.draw()
        
        # Log the update
        self.log_message(f"Updated global model metrics for {stage}")
    
    def show_local_models(self):
        """Show all local model windows."""
        # Create windows if they don't exist
        for client_id, model_name in enumerate(CLIENT_MODELS):
            if client_id not in self.local_windows:
                local_window = LocalModelWindow(client_id, model_name, self.theme_manager)
                self.local_windows[client_id] = local_window
                
                # Apply current theme to the window
                self.theme_manager.apply_theme(QApplication.instance())
                
                # Position windows in a grid
                x_pos = (client_id % 3) * 900
                y_pos = (client_id // 3) * 700
                local_window.move(x_pos, y_pos)
            
            # Show the window
            self.local_windows[client_id].show()
            self.local_windows[client_id].activateWindow()
    
    def export_results(self):
        """Export training results and metrics."""
        try:
            # Ask user for export directory
            export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if not export_dir:
                return
                
            # Export results using the FL system's built-in functionality
            if hasattr(self.fl_system, 'logger') and hasattr(self.fl_system.logger, 'experiment_dir'):
                # We'll copy the experiment files to the user's chosen location
                import shutil
                source_dir = self.fl_system.logger.experiment_dir
                target_dir = os.path.join(export_dir, os.path.basename(source_dir))
                
                # Create the target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy all files
                for item in os.listdir(source_dir):
                    s = os.path.join(source_dir, item)
                    d = os.path.join(target_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
                
                QMessageBox.information(self, "Export Successful", 
                                       f"Results exported to:\n{target_dir}")
            else:
                QMessageBox.warning(self, "Export Failed", 
                                  "No experiment results available to export.")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")


def main():
    """Main function to run the GUI application."""
    parser = argparse.ArgumentParser(description="HETROFL GUI Enhanced")
    parser.add_argument('--style', default='Fusion', help='Qt style to use')
    parser.add_argument('--theme', default='light', choices=['light', 'dark', 'auto'], help='Theme to use')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setStyle(args.style)
    app.setApplicationName("HETROFL")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("HETROFL Team")
    
    # Create and show enhanced splash screen
    splash_pixmap = QPixmap(600, 400)
    splash_pixmap.fill(QColor("#2196F3"))
    splash = QSplashScreen(splash_pixmap)
    splash.showMessage(
        "HETROFL v2.0\nHeterogeneous Federated Learning System\nLoading...",
        alignment=Qt.AlignCenter | Qt.AlignBottom,
        color=Qt.white
    )
    splash.show()
    app.processEvents()
    
    # Create main window
    main_window = GlobalModelWindow()
    
    # Apply initial theme
    if args.theme:
        main_window.theme_manager.apply_theme(app, args.theme)
    
    # Hide splash and show main window
    splash.finish(main_window)
    main_window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 
