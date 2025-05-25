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