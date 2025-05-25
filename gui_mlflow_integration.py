#!/usr/bin/env python
"""
Module for MLflow integration in the HETROFL GUI system.
Provides tools for logging experiments, tracking metrics, and
managing model artifacts through MLflow.
"""

import os
import json
import time
import tempfile
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject


class MLflowLogger:
    """
    Handles MLflow experiment tracking, logging parameters and metrics,
    and saving model artifacts.
    """
    def __init__(self, experiment_name="HETROFL", tracking_uri=None):
        """
        Initialize the MLflow logger.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server (optional)
        """
        self.experiment_name = experiment_name
        
        # Use local directory if no tracking URI is provided
        if tracking_uri is None:
            # Create mlruns directory if it doesn't exist
            os.makedirs("mlruns", exist_ok=True)
            tracking_uri = "file:./mlruns"
        
        self.tracking_uri = tracking_uri
        
        try:
            mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient(tracking_uri)
            
            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            except MlflowException:
                self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            
            mlflow.set_experiment(experiment_name)
            self.active_run = None
            
        except Exception as e:
            print(f"Warning: Failed to initialize MLflow logging: {e}")
            self.client = None
            self.experiment_id = None
    
    def start_run(self, run_name=None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            The MLflow run object
        """
        if self.experiment_id is None:
            print("Warning: MLflow not initialized, can't start run")
            return None
        
        if run_name is None:
            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.active_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name
            )
            return self.active_run
        except Exception as e:
            print(f"Warning: Failed to start MLflow run: {e}")
            return None
    
    def end_run(self):
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
    
    def log_params(self, params):
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if self.active_run is None:
            print("Warning: No active MLflow run, can't log parameters")
            return
        
        try:
            for key, value in params.items():
                # Convert to strings for MLflow
                if isinstance(value, (list, dict, tuple, set)):
                    value = json.dumps(value)
                mlflow.log_param(key, value)
        except Exception as e:
            print(f"Warning: Failed to log MLflow parameters: {e}")
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        if self.active_run is None:
            print("Warning: No active MLflow run, can't log metrics")
            return
        
        try:
            for key, value in metrics.items():
                # Ensure value is numeric
                if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"Warning: Failed to log MLflow metrics: {e}")
    
    def log_model(self, model, model_name, flavor=None, extra_info=None):
        """
        Log a model to the current run.
        
        Args:
            model: The model object to log
            model_name: Name for the model artifact
            flavor: MLflow model flavor (sklearn, pytorch, pyfunc, etc.)
            extra_info: Additional metadata to log with the model
        """
        if self.active_run is None:
            print("Warning: No active MLflow run, can't log model")
            return
        
        try:
            # Create temporary directory for artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Log model based on flavor
                if flavor == "sklearn" or (flavor is None and hasattr(model, "predict")):
                    mlflow.sklearn.log_model(model, model_name)
                
                elif flavor == "pytorch" or hasattr(model, "state_dict"):
                    mlflow.pytorch.log_model(model, model_name)
                
                else:
                    # Use custom PYFUNC model wrapper
                    mlflow.pyfunc.log_model(model_name, python_model=model)
                
                # Log extra model info if provided
                if extra_info:
                    info_path = os.path.join(tmpdir, f"{model_name}_info.json")
                    with open(info_path, "w") as f:
                        json.dump(extra_info, f)
                    mlflow.log_artifact(info_path, f"models/{model_name}")
                
        except Exception as e:
            print(f"Warning: Failed to log model to MLflow: {e}")
    
    def log_figure(self, figure, figure_name):
        """
        Log a matplotlib figure to the current run.
        
        Args:
            figure: The matplotlib figure object
            figure_name: Name for the figure artifact
        """
        if self.active_run is None:
            print("Warning: No active MLflow run, can't log figure")
            return
        
        try:
            # Create temporary file for the figure
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                figure.savefig(tmpfile.name, dpi=300, bbox_inches="tight")
            
            # Log the figure as an artifact
            mlflow.log_artifact(tmpfile.name, "figures")
            
            # Clean up the temporary file
            os.unlink(tmpfile.name)
            
        except Exception as e:
            print(f"Warning: Failed to log figure to MLflow: {e}")
    
    def log_data(self, data, data_name):
        """
        Log a pandas DataFrame or numpy array to the current run.
        
        Args:
            data: DataFrame or array to log
            data_name: Name for the data artifact
        """
        if self.active_run is None:
            print("Warning: No active MLflow run, can't log data")
            return
        
        try:
            # Create temporary file for the data
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpfile:
                if isinstance(data, pd.DataFrame):
                    data.to_csv(tmpfile.name, index=False)
                elif isinstance(data, np.ndarray):
                    pd.DataFrame(data).to_csv(tmpfile.name, index=False)
                else:
                    print(f"Warning: Unsupported data type {type(data)}")
                    return
            
            # Log the data as an artifact
            mlflow.log_artifact(tmpfile.name, "data")
            
            # Clean up the temporary file
            os.unlink(tmpfile.name)
            
        except Exception as e:
            print(f"Warning: Failed to log data to MLflow: {e}")


class MLflowRunSignals(QObject):
    """Signals for MLflow run worker thread."""
    finished = Signal()
    error = Signal(str)
    progress = Signal(int)
    run_loaded = Signal(str, dict, dict)  # run_id, params, metrics


class MLflowRunLoader(QThread):
    """Thread for loading run data from MLflow."""
    def __init__(self, mlflow_logger, run_id=None):
        super().__init__()
        self.mlflow_logger = mlflow_logger
        self.run_id = run_id
        self.signals = MLflowRunSignals()
        
    def run(self):
        """Run the thread to load the run data."""
        try:
            if not self.mlflow_logger.client:
                self.signals.error.emit("MLflow client not initialized")
                return
                
            if not self.run_id:
                self.signals.error.emit("No run ID provided")
                return
                
            # Get run data
            run = self.mlflow_logger.client.get_run(self.run_id)
            
            # Extract parameters and metrics
            params = run.data.params
            metrics = run.data.metrics
            
            # Emit the loaded data
            self.signals.run_loaded.emit(self.run_id, params, metrics)
            self.signals.finished.emit()
            
        except Exception as e:
            self.signals.error.emit(f"Error loading run data: {str(e)}")


class MLflowWidget(QWidget):
    """Widget for MLflow experiment tracking and visualization."""
    def __init__(self, experiment_name="HETROFL"):
        super().__init__()
        self.experiment_name = experiment_name
        self.mlflow_logger = MLflowLogger(experiment_name)
        
        # Set up UI
        self.setup_ui()
        
        # Load runs
        self.load_runs()
    
    def setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        
        # Header section
        header_group = QGroupBox("MLflow Experiment Tracking")
        header_layout = QHBoxLayout(header_group)
        
        self.experiment_label = QLabel(f"Experiment: {self.experiment_name}")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.load_runs)
        
        header_layout.addWidget(self.experiment_label)
        header_layout.addStretch()
        header_layout.addWidget(self.refresh_button)
        
        layout.addWidget(header_group)
        
        # Runs table
        self.runs_table = QTableWidget()
        self.runs_table.setColumnCount(7)
        self.runs_table.setHorizontalHeaderLabels([
            "Run ID", 
            "Start Time", 
            "Status", 
            "Duration", 
            "Accuracy",
            "F1 Score",
            "Models"
        ])
        
        # Configure table
        self.runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.runs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.runs_table)
        
        # Detail section
        detail_group = QGroupBox("Run Details")
        detail_layout = QVBoxLayout(detail_group)
        
        # Parameters and metrics tables
        params_metrics_layout = QHBoxLayout()
        
        # Parameters table
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(2)
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        params_layout.addWidget(self.params_table)
        
        # Metrics table
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        metrics_layout.addWidget(self.metrics_table)
        
        params_metrics_layout.addWidget(params_group)
        params_metrics_layout.addWidget(metrics_group)
        
        detail_layout.addLayout(params_metrics_layout)
        
        # Actions
        actions_layout = QHBoxLayout()
        self.view_model_button = QPushButton("View Models")
        self.view_model_button.setEnabled(False)
        self.compare_button = QPushButton("Compare Runs")
        self.compare_button.setEnabled(False)
        self.export_button = QPushButton("Export Results")
        
        actions_layout.addWidget(self.view_model_button)
        actions_layout.addWidget(self.compare_button)
        actions_layout.addWidget(self.export_button)
        
        detail_layout.addLayout(actions_layout)
        
        layout.addWidget(detail_group)
        
        # Connect signals
        self.runs_table.itemSelectionChanged.connect(self.on_run_selected)
        self.export_button.clicked.connect(self.export_run_results)
        self.view_model_button.clicked.connect(self.view_run_models)
        
    def load_runs(self):
        """Load runs from MLflow."""
        if not self.mlflow_logger.client:
            return
            
        try:
            # Clear the table
            self.runs_table.setRowCount(0)
            
            # Get runs for the experiment
            runs = self.mlflow_logger.client.search_runs(
                experiment_ids=[self.mlflow_logger.experiment_id]
            )
            
            # Add runs to the table
            for i, run in enumerate(runs):
                self.runs_table.insertRow(i)
                
                # Run ID
                self.runs_table.setItem(i, 0, QTableWidgetItem(run.info.run_id))
                
                # Start time
                start_time = time.strftime(
                    '%Y-%m-%d %H:%M:%S', 
                    time.localtime(run.info.start_time / 1000)
                )
                self.runs_table.setItem(i, 1, QTableWidgetItem(start_time))
                
                # Status
                self.runs_table.setItem(i, 2, QTableWidgetItem(run.info.status))
                
                # Duration
                if run.info.end_time:
                    duration_sec = (run.info.end_time - run.info.start_time) / 1000
                    duration = f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"
                else:
                    duration = "Running"
                self.runs_table.setItem(i, 3, QTableWidgetItem(duration))
                
                # Accuracy
                accuracy = run.data.metrics.get("global_accuracy", "N/A")
                if isinstance(accuracy, (int, float)):
                    accuracy = f"{accuracy:.4f}"
                self.runs_table.setItem(i, 4, QTableWidgetItem(str(accuracy)))
                
                # F1 Score
                f1_score = run.data.metrics.get("global_f1_score", "N/A")
                if isinstance(f1_score, (int, float)):
                    f1_score = f"{f1_score:.4f}"
                self.runs_table.setItem(i, 5, QTableWidgetItem(str(f1_score)))
                
                # Models
                try:
                    model_count = len(self.mlflow_logger.client.list_artifacts(
                        run.info.run_id, 
                        path="models"
                    ))
                    self.runs_table.setItem(i, 6, QTableWidgetItem(str(model_count)))
                except:
                    self.runs_table.setItem(i, 6, QTableWidgetItem("0"))
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load runs: {str(e)}")
    
    def on_run_selected(self):
        """Handler for run selection."""
        # Get the selected row
        selected_rows = self.runs_table.selectedItems()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        run_id = self.runs_table.item(row, 0).text()
        
        # Enable the action buttons
        self.view_model_button.setEnabled(True)
        self.compare_button.setEnabled(True)
        
        # Load run details
        self.load_run_details(run_id)
    
    def load_run_details(self, run_id):
        """Load details for the selected run."""
        # Create and start the loader thread
        self.loader = MLflowRunLoader(self.mlflow_logger, run_id)
        self.loader.signals.run_loaded.connect(self.display_run_details)
        self.loader.signals.error.connect(self.show_error)
        self.loader.start()
    
    @Slot(str, dict, dict)
    def display_run_details(self, run_id, params, metrics):
        """Display run details in the UI."""
        # Clear the tables
        self.params_table.setRowCount(0)
        self.metrics_table.setRowCount(0)
        
        # Add parameters to the table
        for i, (key, value) in enumerate(params.items()):
            self.params_table.insertRow(i)
            self.params_table.setItem(i, 0, QTableWidgetItem(key))
            self.params_table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        # Add metrics to the table
        for i, (key, value) in enumerate(metrics.items()):
            self.metrics_table.insertRow(i)
            self.metrics_table.setItem(i, 0, QTableWidgetItem(key))
            
            # Format numeric metrics
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
                
            self.metrics_table.setItem(i, 1, QTableWidgetItem(formatted_value))
    
    def show_error(self, message):
        """Display an error message."""
        QMessageBox.critical(self, "Error", message)
    
    def export_run_results(self):
        """Export results from the selected run."""
        # Get the selected row
        selected_rows = self.runs_table.selectedItems()
        if not selected_rows:
            QMessageBox.information(self, "Selection Required", "Please select a run to export.")
            return
            
        row = selected_rows[0].row()
        run_id = self.runs_table.item(row, 0).text()
        
        # Ask for export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
            
        try:
            # Create run directory
            run_dir = os.path.join(export_dir, f"run_{run_id}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Get run data
            run = self.mlflow_logger.client.get_run(run_id)
            
            # Export parameters
            params_file = os.path.join(run_dir, "parameters.json")
            with open(params_file, "w") as f:
                json.dump(run.data.params, f, indent=2)
            
            # Export metrics
            metrics_file = os.path.join(run_dir, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(run.data.metrics, f, indent=2)
            
            # Export metadata
            metadata_file = os.path.join(run_dir, "metadata.json")
            metadata = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status,
                "artifact_uri": run.info.artifact_uri,
            }
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Download artifacts
            artifacts_dir = os.path.join(run_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Get artifact paths
            artifacts = self.mlflow_logger.client.list_artifacts(run_id)
            for artifact in artifacts:
                if artifact.is_dir:
                    self._download_artifact_dir(run_id, artifact.path, artifacts_dir)
                else:
                    local_path = os.path.join(artifacts_dir, artifact.path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    self.mlflow_logger.client.download_artifacts(run_id, artifact.path, artifacts_dir)
            
            QMessageBox.information(self, "Export Complete", 
                                   f"Run exported to:\n{run_dir}")
                                   
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export run: {str(e)}")
    
    def _download_artifact_dir(self, run_id, artifact_path, local_dir):
        """Recursively download directory artifacts."""
        artifacts = self.mlflow_logger.client.list_artifacts(run_id, artifact_path)
        for artifact in artifacts:
            if artifact.is_dir:
                self._download_artifact_dir(run_id, artifact.path, local_dir)
            else:
                local_path = os.path.join(local_dir, artifact.path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.mlflow_logger.client.download_artifacts(run_id, artifact.path, local_dir)
    
    def view_run_models(self):
        """View models from the selected run."""
        # Get the selected row
        selected_rows = self.runs_table.selectedItems()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        run_id = self.runs_table.item(row, 0).text()
        
        try:
            # Get models
            models_dir = "models"
            models = self.mlflow_logger.client.list_artifacts(run_id, models_dir)
            
            if not models:
                QMessageBox.information(self, "No Models", "No models found for this run.")
                return
                
            # Show model information
            models_info = []
            for model in models:
                if model.is_dir:
                    # This is a model directory
                    model_name = os.path.basename(model.path)
                    
                    # Try to get model info
                    try:
                        info_files = self.mlflow_logger.client.list_artifacts(run_id, model.path)
                        info_file = next((f for f in info_files if f.path.endswith("_info.json")), None)
                        
                        if info_file:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                local_path = self.mlflow_logger.client.download_artifacts(
                                    run_id, info_file.path, tmpdir
                                )
                                with open(local_path, "r") as f:
                                    model_info = json.load(f)
                                    models_info.append({
                                        "name": model_name,
                                        **model_info
                                    })
                        else:
                            models_info.append({
                                "name": model_name,
                                "type": "Unknown",
                                "accuracy": "N/A",
                                "f1_score": "N/A"
                            })
                            
                    except Exception:
                        models_info.append({
                            "name": model_name,
                            "type": "Unknown",
                            "accuracy": "N/A",
                            "f1_score": "N/A"
                        })
            
            # Display models in a message box
            if models_info:
                message = "Models in this run:\n\n"
                for info in models_info:
                    message += f"Name: {info['name']}\n"
                    for key, value in info.items():
                        if key != 'name':
                            message += f"  {key}: {value}\n"
                    message += "\n"
                    
                QMessageBox.information(self, "Run Models", message)
            else:
                QMessageBox.information(self, "No Models", "No model information found for this run.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to retrieve model information: {str(e)}") 