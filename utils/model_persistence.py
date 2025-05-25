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
    Enhanced to support round-by-round tracking and testing.
    """
    
    def __init__(self):
        """Initialize the model tracker."""
        self.history_dir = os.path.join(RESULTS_DIR, "history")
        self.performance_history_path = os.path.join(self.history_dir, "performance_history.json")
        self.model_registry_path = os.path.join(self.history_dir, "model_registry.json")
        self.round_history_path = os.path.join(self.history_dir, "round_history.json")
        
        # Create directories if they don't exist
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Initialize or load performance history
        self.performance_history = self._load_performance_history()
        
        # Initialize or load model registry
        self.model_registry = self._load_model_registry()
        
        # Initialize or load round-by-round history
        self.round_history = self._load_round_history()
        
        # Current experiment tracking
        self.current_experiment_id = None
        self.current_experiment_dir = None
    
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
    
    def _load_round_history(self):
        """Load round-by-round history from file or initialize if not exists."""
        if os.path.exists(self.round_history_path):
            with open(self.round_history_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "experiments": {}
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
    
    def start_experiment(self, experiment_name=None):
        """
        Start a new experiment for round-by-round tracking.
        
        Args:
            experiment_name: Optional name for the experiment
            
        Returns:
            Experiment ID and directory path
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if experiment_name:
            self.current_experiment_id = f"{experiment_name}_{timestamp}"
        else:
            self.current_experiment_id = f"experiment_{timestamp}"
        
        # Create experiment directory
        self.current_experiment_dir = os.path.join(RESULTS_DIR, "models", self.current_experiment_id)
        os.makedirs(self.current_experiment_dir, exist_ok=True)
        
        # Initialize experiment in round history
        self.round_history["experiments"][self.current_experiment_id] = {
            "start_time": timestamp,
            "rounds": {},
            "config": {},
            "status": "running"
        }
        
        self._save_round_history()
        
        print(f"Started experiment: {self.current_experiment_id}")
        return self.current_experiment_id, self.current_experiment_dir
    
    def register_round_models(self, round_num, global_model_path, local_model_paths, 
                            global_metrics, local_metrics, round_config=None):
        """
        Register models and metrics for a specific round.
        
        Args:
            round_num: Round number
            global_model_path: Path to saved global model
            local_model_paths: Dictionary of client_id -> model_path for local models
            global_metrics: Performance metrics for global model
            local_metrics: Dictionary of client_id -> metrics for local models
            round_config: Optional configuration for this round
        """
        if not self.current_experiment_id:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store round information
        round_data = {
            "timestamp": timestamp,
            "global_model": {
                "path": global_model_path,
                "metrics": global_metrics
            },
            "local_models": {},
            "config": round_config or {}
        }
        
        # Store local model information
        for client_id, model_path in local_model_paths.items():
            round_data["local_models"][client_id] = {
                "path": model_path,
                "metrics": local_metrics.get(client_id, {})
            }
        
        # Add to round history
        self.round_history["experiments"][self.current_experiment_id]["rounds"][str(round_num)] = round_data
        
        self._save_round_history()
        
        print(f"Registered models for round {round_num} in experiment {self.current_experiment_id}")
    
    def get_round_models(self, experiment_id, round_num):
        """
        Get model paths and metrics for a specific round.
        
        Args:
            experiment_id: Experiment ID
            round_num: Round number
            
        Returns:
            Dictionary with model paths and metrics for the round
        """
        if experiment_id not in self.round_history["experiments"]:
            return None
        
        experiment = self.round_history["experiments"][experiment_id]
        round_key = str(round_num)
        
        if round_key not in experiment["rounds"]:
            return None
        
        return experiment["rounds"][round_key]
    
    def get_experiment_rounds(self, experiment_id):
        """
        Get all rounds for a specific experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dictionary of round data
        """
        if experiment_id not in self.round_history["experiments"]:
            return {}
        
        return self.round_history["experiments"][experiment_id]["rounds"]
    
    def list_experiments(self):
        """
        List all available experiments.
        
        Returns:
            List of experiment IDs
        """
        return list(self.round_history["experiments"].keys())
    
    def get_latest_experiment(self):
        """
        Get the most recent experiment ID.
        
        Returns:
            Latest experiment ID or None
        """
        experiments = self.round_history["experiments"]
        if not experiments:
            return None
        
        # Sort by start time
        sorted_experiments = sorted(
            experiments.items(),
            key=lambda x: x[1]["start_time"],
            reverse=True
        )
        
        return sorted_experiments[0][0] if sorted_experiments else None
    
    def _save_round_history(self):
        """Save the round history to disk."""
        with open(self.round_history_path, 'w') as f:
            json.dump(self.round_history, f, indent=2)
    
    def finish_experiment(self, final_metrics=None):
        """
        Mark the current experiment as finished.
        
        Args:
            final_metrics: Optional final metrics for the experiment
        """
        if not self.current_experiment_id:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment = self.round_history["experiments"][self.current_experiment_id]
        experiment["end_time"] = timestamp
        experiment["status"] = "completed"
        
        if final_metrics:
            experiment["final_metrics"] = final_metrics
        
        self._save_round_history()
        
        print(f"Finished experiment: {self.current_experiment_id}")
        self.current_experiment_id = None
        self.current_experiment_dir = None
    
    def test_models_across_rounds(self, experiment_id, test_data, test_labels, 
                                 class_names=None, save_results=True):
        """
        Test all models across all rounds for a given experiment.
        
        Args:
            experiment_id: Experiment ID to test
            test_data: Test dataset features
            test_labels: Test dataset labels
            class_names: List of class names for evaluation
            save_results: Whether to save test results to disk
            
        Returns:
            Dictionary with test results for all rounds
        """
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        import joblib
        
        if experiment_id not in self.round_history["experiments"]:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.round_history["experiments"][experiment_id]
        rounds = experiment["rounds"]
        
        test_results = {
            "experiment_id": experiment_id,
            "test_size": len(test_labels),
            "rounds": {}
        }
        
        print(f"Testing models across {len(rounds)} rounds for experiment {experiment_id}")
        
        for round_num, round_data in rounds.items():
            print(f"\nTesting Round {round_num}...")
            round_results = {
                "round_num": int(round_num),
                "timestamp": round_data["timestamp"],
                "global_model": {},
                "local_models": {}
            }
            
            # Test global model
            global_model_path = round_data["global_model"]["path"]
            if global_model_path and os.path.exists(global_model_path):
                try:
                    # Load and test global model
                    global_model = joblib.load(global_model_path)
                    global_predictions = global_model.predict(test_data)
                    
                    global_accuracy = accuracy_score(test_labels, global_predictions)
                    global_f1 = f1_score(test_labels, global_predictions, average='weighted')
                    
                    round_results["global_model"] = {
                        "accuracy": global_accuracy,
                        "f1_score": global_f1,
                        "predictions": global_predictions.tolist() if hasattr(global_predictions, 'tolist') else global_predictions,
                        "confusion_matrix": confusion_matrix(test_labels, global_predictions).tolist()
                    }
                    
                    print(f"  Global model - Accuracy: {global_accuracy:.4f}, F1: {global_f1:.4f}")
                    
                except Exception as e:
                    print(f"  Error testing global model: {e}")
                    round_results["global_model"]["error"] = str(e)
            
            # Test local models
            for client_id, local_data in round_data["local_models"].items():
                local_model_path = local_data["path"]
                if local_model_path and os.path.exists(local_model_path):
                    try:
                        # Load and test local model
                        local_model = joblib.load(local_model_path)
                        local_predictions = local_model.predict(test_data)
                        
                        local_accuracy = accuracy_score(test_labels, local_predictions)
                        local_f1 = f1_score(test_labels, local_predictions, average='weighted')
                        
                        round_results["local_models"][client_id] = {
                            "accuracy": local_accuracy,
                            "f1_score": local_f1,
                            "predictions": local_predictions.tolist() if hasattr(local_predictions, 'tolist') else local_predictions,
                            "confusion_matrix": confusion_matrix(test_labels, local_predictions).tolist()
                        }
                        
                        print(f"  Client {client_id} - Accuracy: {local_accuracy:.4f}, F1: {local_f1:.4f}")
                        
                    except Exception as e:
                        print(f"  Error testing client {client_id}: {e}")
                        round_results["local_models"][client_id] = {"error": str(e)}
            
            test_results["rounds"][round_num] = round_results
        
        # Save test results if requested
        if save_results:
            results_dir = os.path.join(RESULTS_DIR, "test_results")
            os.makedirs(results_dir, exist_ok=True)
            
            results_file = os.path.join(results_dir, f"{experiment_id}_test_results.json")
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            print(f"\nTest results saved to: {results_file}")
        
        return test_results
    
    def analyze_improvement_trends(self, experiment_id, save_plots=True):
        """
        Analyze improvement trends across rounds for an experiment.
        
        Args:
            experiment_id: Experiment ID to analyze
            save_plots: Whether to save improvement plots
            
        Returns:
            Dictionary with improvement analysis
        """
        if experiment_id not in self.round_history["experiments"]:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.round_history["experiments"][experiment_id]
        rounds = experiment["rounds"]
        
        if not rounds:
            return {"error": "No rounds found for experiment"}
        
        # Extract metrics across rounds
        round_numbers = sorted([int(r) for r in rounds.keys()])
        
        analysis = {
            "experiment_id": experiment_id,
            "total_rounds": len(round_numbers),
            "global_model_trends": {
                "accuracy": [],
                "f1_score": [],
                "round_numbers": round_numbers
            },
            "local_model_trends": {},
            "improvement_summary": {}
        }
        
        # Collect global model metrics
        for round_num in round_numbers:
            round_data = rounds[str(round_num)]
            global_metrics = round_data["global_model"]["metrics"]
            
            analysis["global_model_trends"]["accuracy"].append(
                global_metrics.get("accuracy", 0)
            )
            analysis["global_model_trends"]["f1_score"].append(
                global_metrics.get("f1_score", 0)
            )
        
        # Collect local model metrics
        for round_num in round_numbers:
            round_data = rounds[str(round_num)]
            
            for client_id, local_data in round_data["local_models"].items():
                if client_id not in analysis["local_model_trends"]:
                    analysis["local_model_trends"][client_id] = {
                        "accuracy": [],
                        "f1_score": [],
                        "round_numbers": round_numbers
                    }
                
                local_metrics = local_data["metrics"]
                analysis["local_model_trends"][client_id]["accuracy"].append(
                    local_metrics.get("accuracy", 0)
                )
                analysis["local_model_trends"][client_id]["f1_score"].append(
                    local_metrics.get("f1_score", 0)
                )
        
        # Calculate improvement summary
        if len(round_numbers) > 1:
            # Global model improvement
            global_acc = analysis["global_model_trends"]["accuracy"]
            global_f1 = analysis["global_model_trends"]["f1_score"]
            
            analysis["improvement_summary"]["global_model"] = {
                "accuracy_improvement": global_acc[-1] - global_acc[0],
                "f1_improvement": global_f1[-1] - global_f1[0],
                "best_accuracy": max(global_acc),
                "best_f1": max(global_f1),
                "best_accuracy_round": round_numbers[global_acc.index(max(global_acc))],
                "best_f1_round": round_numbers[global_f1.index(max(global_f1))]
            }
            
            # Local models improvement
            analysis["improvement_summary"]["local_models"] = {}
            for client_id, trends in analysis["local_model_trends"].items():
                local_acc = trends["accuracy"]
                local_f1 = trends["f1_score"]
                
                analysis["improvement_summary"]["local_models"][client_id] = {
                    "accuracy_improvement": local_acc[-1] - local_acc[0],
                    "f1_improvement": local_f1[-1] - local_f1[0],
                    "best_accuracy": max(local_acc),
                    "best_f1": max(local_f1),
                    "best_accuracy_round": round_numbers[local_acc.index(max(local_acc))],
                    "best_f1_round": round_numbers[local_f1.index(max(local_f1))]
                }
        
        # Generate improvement plots if requested
        if save_plots:
            self._plot_improvement_trends(analysis, experiment_id)
        
        return analysis
    
    def _plot_improvement_trends(self, analysis, experiment_id):
        """
        Generate improvement trend plots.
        
        Args:
            analysis: Improvement analysis data
            experiment_id: Experiment ID for file naming
        """
        plots_dir = os.path.join(RESULTS_DIR, "improvement_plots", experiment_id)
        os.makedirs(plots_dir, exist_ok=True)
        
        round_numbers = analysis["global_model_trends"]["round_numbers"]
        
        # Global model improvement plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(round_numbers, analysis["global_model_trends"]["accuracy"], 'b-o', label='Accuracy')
        plt.plot(round_numbers, analysis["global_model_trends"]["f1_score"], 'r-s', label='F1 Score')
        plt.title('Global Model Performance Across Rounds')
        plt.xlabel('Round Number')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Local models improvement plot
        plt.subplot(1, 2, 2)
        for client_id, trends in analysis["local_model_trends"].items():
            plt.plot(round_numbers, trends["accuracy"], 'o-', label=f'Client {client_id}')
        
        plt.title('Local Models Accuracy Across Rounds')
        plt.xlabel('Round Number')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'improvement_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual improvement plots for each model
        for client_id, trends in analysis["local_model_trends"].items():
            plt.figure(figsize=(10, 6))
            plt.plot(round_numbers, trends["accuracy"], 'b-o', label='Accuracy')
            plt.plot(round_numbers, trends["f1_score"], 'r-s', label='F1 Score')
            plt.title(f'Client {client_id} Performance Across Rounds')
            plt.xlabel('Round Number')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'client_{client_id}_improvement.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Improvement plots saved to: {plots_dir}")
