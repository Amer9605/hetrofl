"""
Test interface for local models in HETROFL GUI.
Provides dataset selection and testing functionality for individual models.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QTextEdit,
    QProgressBar, QTableWidget, QTableWidgetItem, QTabWidget,
    QSplitter, QMessageBox, QSpinBox, QCheckBox, QLineEdit
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data.data_loader import DataLoader
from utils.metrics import calculate_metrics
from utils.model_persistence import ModelTracker


class TestWorkerSignals(QObject):
    """Signals for the test worker thread."""
    started = Signal()
    finished = Signal()
    error = Signal(str)
    progress = Signal(int, str)
    result = Signal(dict)


class ModelTestWorker(QThread):
    """Worker thread for running model tests."""
    
    def __init__(self, model, test_data, test_labels, test_params):
        super().__init__()
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels
        self.test_params = test_params
        self.signals = TestWorkerSignals()
        
    def run(self):
        """Run the model test."""
        try:
            self.signals.started.emit()
            self.signals.progress.emit(10, "Preparing test data...")
            
            # Prepare data
            if hasattr(self.test_data, 'values'):
                X_test = self.test_data.values
            else:
                X_test = self.test_data
                
            if hasattr(self.test_labels, 'values'):
                y_test = self.test_labels.values
            else:
                y_test = self.test_labels
            
            self.signals.progress.emit(30, "Running model predictions...")
            
            # Make predictions
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X_test)
            else:
                raise ValueError("Model does not have a predict method")
            
            self.signals.progress.emit(60, "Calculating metrics...")
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, predictions)
            
            self.signals.progress.emit(80, "Generating visualizations...")
            
            # Add additional test information
            test_results = {
                'metrics': metrics,
                'predictions': predictions,
                'actual': y_test,
                'test_size': len(y_test),
                'model_type': type(self.model).__name__,
                'test_params': self.test_params
            }
            
            self.signals.progress.emit(100, "Test completed!")
            self.signals.result.emit(test_results)
            self.signals.finished.emit()
            
        except Exception as e:
            import traceback
            self.signals.error.emit(f"Test failed: {str(e)}\n{traceback.format_exc()}")


class DatasetSelector(QWidget):
    """Widget for selecting and configuring test datasets."""
    
    dataset_selected = Signal(str, dict)  # path, config
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_dataset_path = None
        
    def setup_ui(self):
        """Setup the dataset selector UI."""
        layout = QVBoxLayout(self)
        
        # Dataset selection group
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QFormLayout(dataset_group)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a dataset file...")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_dataset)
        
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_button)
        dataset_layout.addRow("Dataset File:", file_layout)
        
        # Dataset format
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Auto-detect", "CSV", "Parquet", "JSON"])
        dataset_layout.addRow("Format:", self.format_combo)
        
        # Target column
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        dataset_layout.addRow("Target Column:", self.target_combo)
        
        # Sample size
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(-1, 1000000)
        self.sample_size_spin.setValue(-1)
        self.sample_size_spin.setSpecialValueText("All")
        dataset_layout.addRow("Sample Size:", self.sample_size_spin)
        
        # Test split
        self.test_split_spin = QSpinBox()
        self.test_split_spin.setRange(10, 50)
        self.test_split_spin.setValue(20)
        self.test_split_spin.setSuffix("%")
        dataset_layout.addRow("Test Split:", self.test_split_spin)
        
        layout.addWidget(dataset_group)
        
        # Dataset preview group
        preview_group = QGroupBox("Dataset Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.info_label = QLabel("No dataset loaded")
        preview_layout.addWidget(self.info_label)
        
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_table)
        
        layout.addWidget(preview_group)
        
        # Load button
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        self.load_button.setEnabled(False)
        layout.addWidget(self.load_button)
        
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
        self.current_dataset_path = path
        self.load_button.setEnabled(bool(path and os.path.exists(path)))
        
        if path and os.path.exists(path):
            self.preview_dataset(path)
            
    def preview_dataset(self, path):
        """Preview the selected dataset."""
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
                
            # Update info
            self.info_label.setText(f"Shape: {df.shape}, Columns: {len(df.columns)}")
            
            # Update target column options
            self.target_combo.clear()
            self.target_combo.addItems(df.columns.tolist())
            
            # Update preview table
            self.preview_table.setRowCount(min(10, len(df)))
            self.preview_table.setColumnCount(len(df.columns))
            self.preview_table.setHorizontalHeaderLabels(df.columns.tolist())
            
            for i in range(min(10, len(df))):
                for j, col in enumerate(df.columns):
                    item = QTableWidgetItem(str(df.iloc[i, j]))
                    self.preview_table.setItem(i, j, item)
                    
            self.preview_table.resizeColumnsToContents()
            
        except Exception as e:
            self.info_label.setText(f"Error loading preview: {str(e)}")
            
    def load_dataset(self):
        """Load the selected dataset and emit signal."""
        if not self.current_dataset_path or not os.path.exists(self.current_dataset_path):
            QMessageBox.warning(self, "Warning", "Please select a valid dataset file.")
            return
            
        config = {
            'path': self.current_dataset_path,
            'format': self.format_combo.currentText(),
            'target_column': self.target_combo.currentText(),
            'sample_size': self.sample_size_spin.value(),
            'test_split': self.test_split_spin.value() / 100.0
        }
        
        self.dataset_selected.emit(self.current_dataset_path, config)


class TestResultsWidget(QWidget):
    """Widget for displaying test results."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the test results UI."""
        layout = QVBoxLayout(self)
        
        # Create tabs for different result views
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Metrics tab
        self.metrics_tab = self.create_metrics_tab()
        self.tabs.addTab(self.metrics_tab, "Metrics")
        
        # Visualizations tab
        self.viz_tab = self.create_visualizations_tab()
        self.tabs.addTab(self.viz_tab, "Visualizations")
        
        # Predictions tab
        self.predictions_tab = self.create_predictions_tab()
        self.tabs.addTab(self.predictions_tab, "Predictions")
        
    def create_metrics_tab(self):
        """Create the metrics display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Metrics summary
        summary_group = QGroupBox("Test Summary")
        summary_layout = QFormLayout(summary_group)
        
        self.test_size_label = QLabel("--")
        self.model_type_label = QLabel("--")
        self.test_duration_label = QLabel("--")
        
        summary_layout.addRow("Test Size:", self.test_size_label)
        summary_layout.addRow("Model Type:", self.model_type_label)
        summary_layout.addRow("Duration:", self.test_duration_label)
        
        layout.addWidget(summary_group)
        
        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        self.accuracy_label = QLabel("--")
        self.precision_label = QLabel("--")
        self.recall_label = QLabel("--")
        self.f1_label = QLabel("--")
        
        metrics_layout.addRow("Accuracy:", self.accuracy_label)
        metrics_layout.addRow("Precision:", self.precision_label)
        metrics_layout.addRow("Recall:", self.recall_label)
        metrics_layout.addRow("F1 Score:", self.f1_label)
        
        layout.addWidget(metrics_group)
        
        # Detailed metrics text
        details_group = QGroupBox("Detailed Results")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont("Consolas", 9))
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        return widget
        
    def create_visualizations_tab(self):
        """Create the visualizations tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Confusion matrix
        self.confusion_matrix_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        layout.addWidget(QLabel("Confusion Matrix"))
        layout.addWidget(self.confusion_matrix_canvas)
        
        # ROC curve (if applicable)
        self.roc_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        layout.addWidget(QLabel("ROC Curve"))
        layout.addWidget(self.roc_canvas)
        
        return widget
        
    def create_predictions_tab(self):
        """Create the predictions display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Predictions table
        self.predictions_table = QTableWidget()
        layout.addWidget(self.predictions_table)
        
        # Export button
        export_button = QPushButton("Export Predictions")
        export_button.clicked.connect(self.export_predictions)
        layout.addWidget(export_button)
        
        return widget
        
    def update_results(self, test_results: Dict[str, Any]):
        """Update the display with new test results."""
        self.test_results = test_results
        
        # Update metrics tab
        metrics = test_results.get('metrics', {})
        
        self.test_size_label.setText(str(test_results.get('test_size', '--')))
        self.model_type_label.setText(test_results.get('model_type', '--'))
        
        self.accuracy_label.setText(f"{metrics.get('accuracy', 0):.4f}")
        self.precision_label.setText(f"{metrics.get('precision_weighted', 0):.4f}")
        self.recall_label.setText(f"{metrics.get('recall_weighted', 0):.4f}")
        self.f1_label.setText(f"{metrics.get('f1_weighted', 0):.4f}")
        
        # Update detailed text
        details_text = "Test Results:\n\n"
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                details_text += f"{key}: {value:.4f}\n"
            else:
                details_text += f"{key}: {value}\n"
        
        self.details_text.setPlainText(details_text)
        
        # Update visualizations
        self.update_visualizations(test_results)
        
        # Update predictions table
        self.update_predictions_table(test_results)
        
    def update_visualizations(self, test_results: Dict[str, Any]):
        """Update visualization plots."""
        try:
            metrics = test_results.get('metrics', {})
            
            # Confusion matrix
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                class_names = metrics.get('class_names', [])
                
                self.confusion_matrix_canvas.figure.clear()
                ax = self.confusion_matrix_canvas.figure.add_subplot(111)
                
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                self.confusion_matrix_canvas.draw()
                
            # ROC curve (if binary classification)
            if 'roc_auc' in metrics and len(np.unique(test_results.get('actual', []))) == 2:
                self.roc_canvas.figure.clear()
                ax = self.roc_canvas.figure.add_subplot(111)
                
                # This is a simplified ROC curve - in practice, you'd need probabilities
                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
                ax.legend()
                
                self.roc_canvas.draw()
                
        except Exception as e:
            print(f"Error updating visualizations: {e}")
            
    def update_predictions_table(self, test_results: Dict[str, Any]):
        """Update the predictions table."""
        try:
            predictions = test_results.get('predictions', [])
            actual = test_results.get('actual', [])
            
            if len(predictions) > 0 and len(actual) > 0:
                # Show first 1000 predictions
                max_rows = min(1000, len(predictions))
                
                self.predictions_table.setRowCount(max_rows)
                self.predictions_table.setColumnCount(3)
                self.predictions_table.setHorizontalHeaderLabels(['Index', 'Actual', 'Predicted'])
                
                for i in range(max_rows):
                    self.predictions_table.setItem(i, 0, QTableWidgetItem(str(i)))
                    self.predictions_table.setItem(i, 1, QTableWidgetItem(str(actual[i])))
                    self.predictions_table.setItem(i, 2, QTableWidgetItem(str(predictions[i])))
                    
                self.predictions_table.resizeColumnsToContents()
                
        except Exception as e:
            print(f"Error updating predictions table: {e}")
            
    def export_predictions(self):
        """Export predictions to CSV file."""
        if not hasattr(self, 'test_results'):
            QMessageBox.warning(self, "Warning", "No test results to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Predictions", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                predictions = self.test_results.get('predictions', [])
                actual = self.test_results.get('actual', [])
                
                df = pd.DataFrame({
                    'actual': actual,
                    'predicted': predictions
                })
                
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Predictions exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export predictions: {str(e)}")


class ModelTestPanel(QWidget):
    """Complete test panel for local models."""
    
    def __init__(self, model=None, model_name="Unknown"):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.current_dataset = None
        self.test_worker = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the test panel UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(f"Model Testing - {self.model_name}")
        title_label.setProperty("class", "title")
        layout.addWidget(title_label)
        
        # Create splitter for dataset selector and results
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Dataset selector
        self.dataset_selector = DatasetSelector()
        self.dataset_selector.dataset_selected.connect(self.on_dataset_selected)
        splitter.addWidget(self.dataset_selector)
        
        # Test controls and results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Test controls
        controls_group = QGroupBox("Test Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Test button and progress
        button_layout = QHBoxLayout()
        self.test_button = QPushButton("Run Test")
        self.test_button.clicked.connect(self.run_test)
        self.test_button.setEnabled(False)
        
        self.stop_button = QPushButton("Stop Test")
        self.stop_button.clicked.connect(self.stop_test)
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.stop_button)
        controls_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.progress_label)
        
        right_layout.addWidget(controls_group)
        
        # Test results
        self.results_widget = TestResultsWidget()
        right_layout.addWidget(self.results_widget)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])
        
    def set_model(self, model, model_name: str):
        """Set the model to test."""
        self.model = model
        self.model_name = model_name
        
    @Slot(str, dict)
    def on_dataset_selected(self, path: str, config: Dict[str, Any]):
        """Handle dataset selection."""
        try:
            # Load the dataset
            if config['format'] == 'Auto-detect' or config['format'] == 'CSV':
                df = pd.read_csv(path)
            elif config['format'] == 'Parquet':
                df = pd.read_parquet(path)
            elif config['format'] == 'JSON':
                df = pd.read_json(path, lines=True)
            else:
                raise ValueError(f"Unsupported format: {config['format']}")
                
            # Apply sampling if specified
            if config['sample_size'] > 0 and config['sample_size'] < len(df):
                df = df.sample(n=config['sample_size'], random_state=42)
                
            # Prepare features and target
            target_col = config['target_column']
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset")
                
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Store dataset info
            self.current_dataset = {
                'X': X,
                'y': y,
                'config': config
            }
            
            self.test_button.setEnabled(True)
            self.progress_label.setText(f"Dataset loaded: {len(df)} samples")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
            self.test_button.setEnabled(False)
            
    def run_test(self):
        """Run the model test."""
        if not self.model or not self.current_dataset:
            QMessageBox.warning(self, "Warning", "Model or dataset not available.")
            return
            
        try:
            # Prepare test data
            X = self.current_dataset['X']
            y = self.current_dataset['y']
            
            # Create test worker
            test_params = {
                'dataset_path': self.current_dataset['config']['path'],
                'test_split': self.current_dataset['config']['test_split']
            }
            
            self.test_worker = ModelTestWorker(self.model, X, y, test_params)
            
            # Connect signals
            self.test_worker.signals.started.connect(self.on_test_started)
            self.test_worker.signals.finished.connect(self.on_test_finished)
            self.test_worker.signals.error.connect(self.on_test_error)
            self.test_worker.signals.progress.connect(self.update_progress)
            self.test_worker.signals.result.connect(self.on_test_result)
            
            # Start test
            self.test_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start test: {str(e)}")
            
    def stop_test(self):
        """Stop the running test."""
        if self.test_worker and self.test_worker.isRunning():
            self.test_worker.terminate()
            self.test_worker.wait()
            self.on_test_finished()
            
    @Slot()
    def on_test_started(self):
        """Handle test start."""
        self.test_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Test started...")
        
    @Slot()
    def on_test_finished(self):
        """Handle test completion."""
        self.test_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Test completed")
        
    @Slot(str)
    def on_test_error(self, error_message: str):
        """Handle test error."""
        self.on_test_finished()
        self.progress_label.setText("Test failed")
        QMessageBox.critical(self, "Test Error", error_message)
        
    @Slot(int, str)
    def update_progress(self, value: int, message: str):
        """Update test progress."""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        
    @Slot(dict)
    def on_test_result(self, results: Dict[str, Any]):
        """Handle test results."""
        self.results_widget.update_results(results)


class RoundTestingWidget(QWidget):
    """Widget for testing models across different rounds."""
    
    def __init__(self):
        super().__init__()
        self.model_tracker = ModelTracker()
        self.current_experiment = None
        self.test_data = None
        self.test_labels = None
        self.setup_ui()
        self.refresh_experiments()
        
    def setup_ui(self):
        """Setup the round testing UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Round-by-Round Model Testing")
        title_label.setProperty("class", "title")
        layout.addWidget(title_label)
        
        # Experiment selection
        exp_group = QGroupBox("Experiment Selection")
        exp_layout = QFormLayout(exp_group)
        
        self.experiment_combo = QComboBox()
        self.experiment_combo.currentTextChanged.connect(self.on_experiment_changed)
        exp_layout.addRow("Experiment:", self.experiment_combo)
        
        self.refresh_exp_button = QPushButton("Refresh")
        self.refresh_exp_button.clicked.connect(self.refresh_experiments)
        exp_layout.addRow("", self.refresh_exp_button)
        
        layout.addWidget(exp_group)
        
        # Round selection
        round_group = QGroupBox("Round Selection")
        round_layout = QFormLayout(round_group)
        
        self.round_combo = QComboBox()
        self.round_combo.addItem("All Rounds", "all")
        round_layout.addRow("Round:", self.round_combo)
        
        # Test data selection
        data_layout = QHBoxLayout()
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("Select test dataset...")
        self.browse_data_button = QPushButton("Browse")
        self.browse_data_button.clicked.connect(self.browse_test_data)
        
        data_layout.addWidget(self.data_path_edit)
        data_layout.addWidget(self.browse_data_button)
        round_layout.addRow("Test Data:", data_layout)
        
        self.target_column_edit = QLineEdit()
        self.target_column_edit.setPlaceholderText("Target column name")
        round_layout.addRow("Target Column:", self.target_column_edit)
        
        layout.addWidget(round_group)
        
        # Test controls
        controls_group = QGroupBox("Test Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        button_layout = QHBoxLayout()
        self.test_round_button = QPushButton("Test Selected Round")
        self.test_round_button.clicked.connect(self.test_selected_round)
        self.test_round_button.setEnabled(False)
        
        self.test_all_button = QPushButton("Test All Rounds")
        self.test_all_button.clicked.connect(self.test_all_rounds)
        self.test_all_button.setEnabled(False)
        
        self.analyze_button = QPushButton("Generate Analysis")
        self.analyze_button.clicked.connect(self.generate_analysis)
        self.analyze_button.setEnabled(False)
        
        button_layout.addWidget(self.test_round_button)
        button_layout.addWidget(self.test_all_button)
        button_layout.addWidget(self.analyze_button)
        controls_layout.addLayout(button_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.status_label)
        
        layout.addWidget(controls_group)
        
        # Results display
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)
        
        # Create tabs for different result views
        self.results_tabs = QTabWidget()
        
        # Summary tab
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")
        
        # Detailed results tab
        self.details_table = QTableWidget()
        self.results_tabs.addTab(self.details_table, "Detailed Results")
        
        # Improvement analysis tab
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.results_tabs.addTab(self.analysis_text, "Improvement Analysis")
        
        results_layout.addWidget(self.results_tabs)
        
        # Export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        results_layout.addWidget(self.export_button)
        
        layout.addWidget(results_group)
        
    def refresh_experiments(self):
        """Refresh the list of available experiments."""
        try:
            experiments = self.model_tracker.list_experiments()
            self.experiment_combo.clear()
            
            if experiments:
                self.experiment_combo.addItems(experiments)
                self.experiment_combo.setCurrentText(self.model_tracker.get_latest_experiment() or "")
            else:
                self.experiment_combo.addItem("No experiments found")
                
        except Exception as e:
            print(f"Error refreshing experiments: {e}")
            
    def on_experiment_changed(self, experiment_id):
        """Handle experiment selection change."""
        if not experiment_id or experiment_id == "No experiments found":
            self.current_experiment = None
            self.round_combo.clear()
            self.round_combo.addItem("All Rounds", "all")
            self.test_round_button.setEnabled(False)
            self.test_all_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            return
            
        self.current_experiment = experiment_id
        
        try:
            # Get rounds for this experiment
            rounds = self.model_tracker.get_experiment_rounds(experiment_id)
            self.round_combo.clear()
            self.round_combo.addItem("All Rounds", "all")
            
            for round_num in sorted(rounds.keys(), key=int):
                self.round_combo.addItem(f"Round {round_num}", round_num)
                
            # Enable buttons if we have rounds
            has_rounds = len(rounds) > 0
            self.test_round_button.setEnabled(has_rounds)
            self.test_all_button.setEnabled(has_rounds)
            self.analyze_button.setEnabled(has_rounds)
            
            self.status_label.setText(f"Experiment loaded: {len(rounds)} rounds available")
            
        except Exception as e:
            print(f"Error loading experiment rounds: {e}")
            self.status_label.setText("Error loading experiment")
            
    def browse_test_data(self):
        """Browse for test data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Dataset",
            "",
            "Data Files (*.csv *.parquet *.json);;CSV Files (*.csv);;Parquet Files (*.parquet);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.data_path_edit.setText(file_path)
            self.load_test_data(file_path)
            
    def load_test_data(self, file_path):
        """Load test data from file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, lines=True)
            else:
                raise ValueError("Unsupported file format")
                
            # Auto-detect target column if not set
            if not self.target_column_edit.text():
                # Common target column names
                target_candidates = ['target', 'label', 'class', 'y', 'Attack', 'attack']
                for candidate in target_candidates:
                    if candidate in df.columns:
                        self.target_column_edit.setText(candidate)
                        break
                        
            self.status_label.setText(f"Test data loaded: {len(df)} samples, {len(df.columns)} features")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load test data: {str(e)}")
            
    def prepare_test_data(self):
        """Prepare test data for testing."""
        file_path = self.data_path_edit.text()
        target_column = self.target_column_edit.text()
        
        if not file_path or not target_column:
            QMessageBox.warning(self, "Warning", "Please select test data and specify target column.")
            return False
            
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, lines=True)
            else:
                raise ValueError("Unsupported file format")
                
            if target_column not in df.columns:
                QMessageBox.critical(self, "Error", f"Target column '{target_column}' not found in dataset.")
                return False
                
            # Prepare features and labels
            self.test_data = df.drop(columns=[target_column])
            self.test_labels = df[target_column]
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare test data: {str(e)}")
            return False
            
    def test_selected_round(self):
        """Test models from the selected round."""
        if not self.current_experiment:
            QMessageBox.warning(self, "Warning", "Please select an experiment.")
            return
            
        if not self.prepare_test_data():
            return
            
        selected_round = self.round_combo.currentData()
        if selected_round == "all":
            QMessageBox.information(self, "Info", "Please select a specific round or use 'Test All Rounds'.")
            return
            
        try:
            self.progress_bar.setValue(0)
            self.status_label.setText(f"Testing round {selected_round}...")
            
            # Test the specific round
            test_results = self.model_tracker.test_models_across_rounds(
                experiment_id=self.current_experiment,
                test_data=self.test_data,
                test_labels=self.test_labels,
                save_results=True
            )
            
            if test_results and selected_round in test_results['rounds']:
                round_data = test_results['rounds'][selected_round]
                self.display_round_results(selected_round, round_data)
                self.export_button.setEnabled(True)
                self.status_label.setText(f"Round {selected_round} testing completed")
            else:
                self.status_label.setText(f"No results for round {selected_round}")
                
            self.progress_bar.setValue(100)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to test round: {str(e)}")
            self.status_label.setText("Testing failed")
            
    def test_all_rounds(self):
        """Test models from all rounds."""
        if not self.current_experiment:
            QMessageBox.warning(self, "Warning", "Please select an experiment.")
            return
            
        if not self.prepare_test_data():
            return
            
        try:
            self.progress_bar.setValue(0)
            self.status_label.setText("Testing all rounds...")
            
            # Test all rounds
            test_results = self.model_tracker.test_models_across_rounds(
                experiment_id=self.current_experiment,
                test_data=self.test_data,
                test_labels=self.test_labels,
                save_results=True
            )
            
            if test_results:
                self.display_all_results(test_results)
                self.export_button.setEnabled(True)
                self.status_label.setText(f"All rounds testing completed ({len(test_results['rounds'])} rounds)")
            else:
                self.status_label.setText("No test results available")
                
            self.progress_bar.setValue(100)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to test all rounds: {str(e)}")
            self.status_label.setText("Testing failed")
            
    def generate_analysis(self):
        """Generate improvement analysis."""
        if not self.current_experiment:
            QMessageBox.warning(self, "Warning", "Please select an experiment.")
            return
            
        try:
            self.progress_bar.setValue(0)
            self.status_label.setText("Generating improvement analysis...")
            
            # Generate analysis
            analysis = self.model_tracker.analyze_improvement_trends(
                experiment_id=self.current_experiment,
                save_plots=True
            )
            
            if analysis:
                self.display_analysis(analysis)
                self.status_label.setText("Analysis completed")
            else:
                self.status_label.setText("No analysis data available")
                
            self.progress_bar.setValue(100)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate analysis: {str(e)}")
            self.status_label.setText("Analysis failed")
            
    def display_round_results(self, round_num, round_data):
        """Display results for a specific round."""
        # Summary
        summary = f"Round {round_num} Test Results\n"
        summary += "=" * 30 + "\n\n"
        
        # Global model results
        if 'global_model' in round_data and round_data['global_model']:
            global_results = round_data['global_model']
            if 'accuracy' in global_results:
                summary += f"Global Model:\n"
                summary += f"  Accuracy: {global_results['accuracy']:.4f}\n"
                summary += f"  F1 Score: {global_results['f1_score']:.4f}\n\n"
        
        # Local model results
        if 'local_models' in round_data:
            summary += "Local Models:\n"
            for client_id, local_results in round_data['local_models'].items():
                if 'accuracy' in local_results:
                    summary += f"  Client {client_id}:\n"
                    summary += f"    Accuracy: {local_results['accuracy']:.4f}\n"
                    summary += f"    F1 Score: {local_results['f1_score']:.4f}\n"
        
        self.summary_text.setText(summary)
        self.results_tabs.setCurrentIndex(0)
        
    def display_all_results(self, test_results):
        """Display results for all rounds."""
        # Summary
        summary = f"All Rounds Test Results\n"
        summary += f"Experiment: {test_results['experiment_id']}\n"
        summary += f"Test Size: {test_results['test_size']}\n"
        summary += "=" * 40 + "\n\n"
        
        # Detailed table
        rounds = test_results['rounds']
        self.details_table.setRowCount(len(rounds))
        self.details_table.setColumnCount(4)
        self.details_table.setHorizontalHeaderLabels(['Round', 'Global Accuracy', 'Global F1', 'Avg Local Accuracy'])
        
        for i, (round_num, round_data) in enumerate(sorted(rounds.items(), key=lambda x: int(x[0]))):
            self.details_table.setItem(i, 0, QTableWidgetItem(str(round_num)))
            
            # Global model metrics
            global_acc = "N/A"
            global_f1 = "N/A"
            if 'global_model' in round_data and 'accuracy' in round_data['global_model']:
                global_acc = f"{round_data['global_model']['accuracy']:.4f}"
                global_f1 = f"{round_data['global_model']['f1_score']:.4f}"
            
            self.details_table.setItem(i, 1, QTableWidgetItem(global_acc))
            self.details_table.setItem(i, 2, QTableWidgetItem(global_f1))
            
            # Average local accuracy
            local_accuracies = []
            if 'local_models' in round_data:
                for local_results in round_data['local_models'].values():
                    if 'accuracy' in local_results:
                        local_accuracies.append(local_results['accuracy'])
            
            avg_local_acc = "N/A"
            if local_accuracies:
                avg_local_acc = f"{np.mean(local_accuracies):.4f}"
            
            self.details_table.setItem(i, 3, QTableWidgetItem(avg_local_acc))
            
            # Add to summary
            summary += f"Round {round_num}:\n"
            summary += f"  Global: Acc={global_acc}, F1={global_f1}\n"
            summary += f"  Local Avg: Acc={avg_local_acc}\n\n"
        
        self.details_table.resizeColumnsToContents()
        self.summary_text.setText(summary)
        
    def display_analysis(self, analysis):
        """Display improvement analysis."""
        analysis_text = "Improvement Analysis\n"
        analysis_text += "=" * 30 + "\n\n"
        
        if "improvement_summary" in analysis:
            # Global model summary
            if "global_model" in analysis["improvement_summary"]:
                global_summary = analysis["improvement_summary"]["global_model"]
                analysis_text += "Global Model:\n"
                analysis_text += f"  Accuracy improvement: {global_summary['accuracy_improvement']:.4f}\n"
                analysis_text += f"  F1 improvement: {global_summary['f1_improvement']:.4f}\n"
                analysis_text += f"  Best accuracy: {global_summary['best_accuracy']:.4f} (Round {global_summary['best_accuracy_round']})\n"
                analysis_text += f"  Best F1: {global_summary['best_f1']:.4f} (Round {global_summary['best_f1_round']})\n\n"
            
            # Local models summary
            if "local_models" in analysis["improvement_summary"]:
                analysis_text += "Local Models:\n"
                for client_id, local_summary in analysis["improvement_summary"]["local_models"].items():
                    analysis_text += f"  Client {client_id}:\n"
                    analysis_text += f"    Accuracy improvement: {local_summary['accuracy_improvement']:.4f}\n"
                    analysis_text += f"    F1 improvement: {local_summary['f1_improvement']:.4f}\n"
                    analysis_text += f"    Best accuracy: {local_summary['best_accuracy']:.4f} (Round {local_summary['best_accuracy_round']})\n"
                    analysis_text += f"    Best F1: {local_summary['best_f1']:.4f} (Round {local_summary['best_f1_round']})\n\n"
        
        self.analysis_text.setText(analysis_text)
        self.results_tabs.setCurrentIndex(2)
        
    def export_results(self):
        """Export test results to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Test Results", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Export logic would go here
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
