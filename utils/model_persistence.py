"""
Model persistence utilities for the HETROFL system.
This module handles saving and loading models across multiple runs for cumulative learning.
"""

import os
import json
import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import MODEL_SAVE_DIR, RESULTS_DIR


class ModelTracker:
    """
    Tracks model performance across multiple runs to enable cumulative learning.
    """
    
    def __init__(self):
        """Initialize the model tracker."""
        self.history_dir = os.path.join(RESULTS_DIR, "history")
        self.performance_history_path = os.path.join(self.history_dir, "performance_history.json")
        self.model_registry_path = os.path.join(self.history_dir, "model_registry.json")
        
        # Create directories if they don't exist
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Initialize or load performance history
        self.performance_history = self._load_performance_history()
        
        # Initialize or load model registry
        self.model_registry = self._load_model_registry()
    
    def _load_performance_history(self):
        """Load performance history from file or initialize if not exists."""
        if os.path.exists(self.performance_history_path):
            with open(self.performance_history_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "global_model": {
                    "runs": [],
                    "accuracy": [],
                    "f1_weighted": [],
                    "timestamp": []
                },
                "local_models": {}
            }
    
    def _load_model_registry(self):
        """Load model registry from file or initialize if not exists."""
        if os.path.exists(self.model_registry_path):
            with open(self.model_registry_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "global_model": {
                    "latest_path": None,
                    "best_path": None,
                    "best_accuracy": 0.0
                },
                "local_models": {}
            }
    
    def register_model_paths(self, model_type, client_id, model_paths, metrics):
        """
        Register model paths for future loading.
        
        Args:
            model_type: Type of model ('global' or 'local')
            client_id: Client ID (None for global model)
            model_paths: Dictionary of paths to model files
            metrics: Dictionary of model performance metrics
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        accuracy = metrics.get('accuracy', 0.0)
        f1_weighted = metrics.get('f1_weighted', 0.0)
        
        if model_type == 'global':
            # Update global model registry
            self.model_registry["global_model"]["latest_path"] = model_paths
            
            # Update best model if current model is better
            if accuracy > self.model_registry["global_model"].get("best_accuracy", 0.0):
                self.model_registry["global_model"]["best_path"] = model_paths
                self.model_registry["global_model"]["best_accuracy"] = accuracy
            
            # Update performance history
            self.performance_history["global_model"]["runs"].append(len(self.performance_history["global_model"]["runs"]) + 1)
            self.performance_history["global_model"]["accuracy"].append(accuracy)
            self.performance_history["global_model"]["f1_weighted"].append(f1_weighted)
            self.performance_history["global_model"]["timestamp"].append(timestamp)
        else:
            # Initialize client entry if not exists
            if client_id not in self.model_registry["local_models"]:
                self.model_registry["local_models"][client_id] = {
                    "latest_path": None,
                    "best_path": None,
                    "best_accuracy": 0.0
                }
                
            if client_id not in self.performance_history["local_models"]:
                self.performance_history["local_models"][client_id] = {
                    "runs": [],
                    "accuracy": [],
                    "f1_weighted": [],
                    "timestamp": []
                }
            
            # Update local model registry
            self.model_registry["local_models"][client_id]["latest_path"] = model_paths
            
            # Update best model if current model is better
            if accuracy > self.model_registry["local_models"][client_id].get("best_accuracy", 0.0):
                self.model_registry["local_models"][client_id]["best_path"] = model_paths
                self.model_registry["local_models"][client_id]["best_accuracy"] = accuracy
            
            # Update performance history
            client_history = self.performance_history["local_models"][client_id]
            client_history["runs"].append(len(client_history["runs"]) + 1)
            client_history["accuracy"].append(accuracy)
            client_history["f1_weighted"].append(f1_weighted)
            client_history["timestamp"].append(timestamp)
        
        # Save updates to disk
        self._save_registry()
        self._save_performance_history()
    
    def get_latest_model_path(self, model_type, client_id=None):
        """
        Get the path to the latest model.
        
        Args:
            model_type: Type of model ('global' or 'local')
            client_id: Client ID (required for local models)
            
        Returns:
            Path to the latest model or None if not found
        """
        if model_type == 'global':
            return self.model_registry["global_model"].get("latest_path")
        else:
            if client_id not in self.model_registry["local_models"]:
                return None
            return self.model_registry["local_models"][client_id].get("latest_path")
    
    def get_best_model_path(self, model_type, client_id=None):
        """
        Get the path to the best performing model.
        
        Args:
            model_type: Type of model ('global' or 'local')
            client_id: Client ID (required for local models)
            
        Returns:
            Path to the best model or None if not found
        """
        if model_type == 'global':
            return self.model_registry["global_model"].get("best_path")
        else:
            if client_id not in self.model_registry["local_models"]:
                return None
            return self.model_registry["local_models"][client_id].get("best_path")
    
    def _save_registry(self):
        """Save the model registry to disk."""
        with open(self.model_registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def _save_performance_history(self):
        """Save the performance history to disk."""
        with open(self.performance_history_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def plot_performance_history(self, save_dir=None):
        """
        Plot the performance history of models across runs.
        
        Args:
            save_dir: Directory to save plots (default: history_dir/plots)
            
        Returns:
            Dictionary of plot file paths
        """
        if save_dir is None:
            save_dir = os.path.join(self.history_dir, "plots")
        
        os.makedirs(save_dir, exist_ok=True)
        plot_paths = {}
        
        # Plot global model performance
        if self.performance_history["global_model"]["runs"]:
            # Create DataFrame for global model
            global_df = pd.DataFrame({
                'Run': self.performance_history["global_model"]["runs"],
                'Accuracy': self.performance_history["global_model"]["accuracy"],
                'F1 Score': self.performance_history["global_model"]["f1_weighted"]
            })
            
            # Plot accuracy and F1 score
            plt.figure(figsize=(12, 6))
            plt.plot(global_df['Run'], global_df['Accuracy'], 'b-o', label='Accuracy')
            plt.plot(global_df['Run'], global_df['F1 Score'], 'r-s', label='F1 Score')
            plt.title('Global Model Performance Over Runs')
            plt.xlabel('Run Number')
            plt.ylabel('Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            global_plot_path = os.path.join(save_dir, 'global_model_performance.png')
            plt.savefig(global_plot_path, dpi=300)
            plot_paths['global_model'] = global_plot_path
            plt.close()
        
        # Plot local models performance
        if self.performance_history["local_models"]:
            # Create a combined plot for all local models
            plt.figure(figsize=(14, 8))
            
            for client_id, history in self.performance_history["local_models"].items():
                if history["runs"]:
                    plt.plot(history["runs"], history["accuracy"], 'o-', label=f'Client {client_id} Accuracy')
            
            plt.title('Local Models Accuracy Over Runs')
            plt.xlabel('Run Number')
            plt.ylabel('Accuracy')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            local_plot_path = os.path.join(save_dir, 'local_models_accuracy.png')
            plt.savefig(local_plot_path, dpi=300)
            plot_paths['local_models_accuracy'] = local_plot_path
            plt.close()
            
            # Plot F1 scores
            plt.figure(figsize=(14, 8))
            
            for client_id, history in self.performance_history["local_models"].items():
                if history["runs"]:
                    plt.plot(history["runs"], history["f1_weighted"], 's-', label=f'Client {client_id} F1 Score')
            
            plt.title('Local Models F1 Score Over Runs')
            plt.xlabel('Run Number')
            plt.ylabel('F1 Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            local_f1_plot_path = os.path.join(save_dir, 'local_models_f1_score.png')
            plt.savefig(local_f1_plot_path, dpi=300)
            plot_paths['local_models_f1'] = local_f1_plot_path
            plt.close()
            
            # Individual plots for each local model
            for client_id, history in self.performance_history["local_models"].items():
                if history["runs"]:
                    plt.figure(figsize=(10, 6))
                    plt.plot(history["runs"], history["accuracy"], 'b-o', label='Accuracy')
                    plt.plot(history["runs"], history["f1_weighted"], 'r-s', label='F1 Score')
                    plt.title(f'Client {client_id} Model Performance Over Runs')
                    plt.xlabel('Run Number')
                    plt.ylabel('Score')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.tight_layout()
                    
                    client_plot_path = os.path.join(save_dir, f'client_{client_id}_performance.png')
                    plt.savefig(client_plot_path, dpi=300)
                    plot_paths[f'client_{client_id}'] = client_plot_path
                    plt.close()
        
        return plot_paths
    
    def get_performance_summary(self):
        """
        Get a summary of model performance across runs.
        
        Returns:
            Dictionary with performance summary
        """
        summary = {
            "global_model": {},
            "local_models": {}
        }
        
        # Global model summary
        global_history = self.performance_history["global_model"]
        if global_history["runs"]:
            summary["global_model"] = {
                "total_runs": len(global_history["runs"]),
                "latest_accuracy": global_history["accuracy"][-1],
                "latest_f1": global_history["f1_weighted"][-1],
                "best_accuracy": max(global_history["accuracy"]),
                "best_f1": max(global_history["f1_weighted"]),
                "improvement_accuracy": global_history["accuracy"][-1] - global_history["accuracy"][0] if len(global_history["accuracy"]) > 1 else 0,
                "improvement_f1": global_history["f1_weighted"][-1] - global_history["f1_weighted"][0] if len(global_history["f1_weighted"]) > 1 else 0
            }
        
        # Local models summary
        for client_id, history in self.performance_history["local_models"].items():
            if history["runs"]:
                summary["local_models"][client_id] = {
                    "total_runs": len(history["runs"]),
                    "latest_accuracy": history["accuracy"][-1],
                    "latest_f1": history["f1_weighted"][-1],
                    "best_accuracy": max(history["accuracy"]),
                    "best_f1": max(history["f1_weighted"]),
                    "improvement_accuracy": history["accuracy"][-1] - history["accuracy"][0] if len(history["accuracy"]) > 1 else 0,
                    "improvement_f1": history["f1_weighted"][-1] - history["f1_weighted"][0] if len(history["f1_weighted"]) > 1 else 0
                }
        
        return summary
    
    def get_performance_history(self):
        """
        Get the complete performance history data.
        
        Returns:
            Dictionary containing performance history for all models
        """
        return self.performance_history 