"""
Logger utility for tracking experiment progress and results.
"""

import logging
import os
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from config.config import LOG_DIR

class ExperimentLogger:
    """
    Logger for tracking federated learning experiments.
    """
    
    def __init__(self, experiment_name=None, log_dir=None):
        """
        Initialize the logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for storing logs
        """
        self.log_dir = log_dir if log_dir else LOG_DIR
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(self.log_dir, experiment_name)
        
        # Create log directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Initialize experiment tracking
        self.start_time = time.time()
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {},
            "stages": []
        }
        
        # Initialize metrics tracking
        self.metrics = {
            "rounds": [],
            "global_metrics": {},
            "local_metrics": {}
        }
        
        self.logger.info(f"Experiment '{experiment_name}' initialized")
    
    def _setup_logger(self):
        """
        Setup the logger.
        
        Returns:
            Configured logger object
        """
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        
        # Create handlers
        log_file = os.path.join(self.experiment_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_parameters(self, parameters):
        """
        Log experiment parameters.
        
        Args:
            parameters: Dictionary of parameters
        """
        self.metadata["parameters"] = parameters
        self.logger.info(f"Parameters: {parameters}")
    
    def log_stage(self, stage_name, stage_data=None):
        """
        Log a stage in the experiment.
        
        Args:
            stage_name: Name of the stage
            stage_data: Additional data to log
        """
        stage = {
            "name": stage_name,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": stage_data
        }
        
        self.metadata["stages"].append(stage)
        
        if stage_data:
            self.logger.info(f"Stage '{stage_name}' completed with data: {stage_data}")
        else:
            self.logger.info(f"Stage '{stage_name}' completed")
    
    def log_round(self, round_num, global_metrics=None, local_metrics=None):
        """
        Log metrics for a communication round.
        
        Args:
            round_num: Round number
            global_metrics: Dictionary of global model metrics
            local_metrics: Dictionary of local model metrics
        """
        round_data = {"round": round_num}
        
        if round_num not in self.metrics["rounds"]:
            self.metrics["rounds"].append(round_num)
        
        # Log global metrics
        if global_metrics:
            self.logger.info(f"Round {round_num} - Global model metrics: {global_metrics}")
            
            round_data["global"] = global_metrics
            
            # Store metrics by type
            for metric_name, metric_value in global_metrics.items():
                if metric_name not in self.metrics["global_metrics"]:
                    self.metrics["global_metrics"][metric_name] = []
                
                # Extend list if needed
                while len(self.metrics["global_metrics"][metric_name]) < round_num:
                    self.metrics["global_metrics"][metric_name].append(None)
                
                # Update the metric for this round
                if len(self.metrics["global_metrics"][metric_name]) == round_num:
                    self.metrics["global_metrics"][metric_name].append(metric_value)
                else:
                    self.metrics["global_metrics"][metric_name][round_num] = metric_value
        
        # Log local metrics
        if local_metrics:
            for client_id, metrics in local_metrics.items():
                self.logger.info(f"Round {round_num} - Client {client_id} metrics: {metrics}")
                
                if "local" not in round_data:
                    round_data["local"] = {}
                
                round_data["local"][client_id] = metrics
                
                # Initialize client entry if needed
                if client_id not in self.metrics["local_metrics"]:
                    self.metrics["local_metrics"][client_id] = {}
                
                # Store metrics by type
                for metric_name, metric_value in metrics.items():
                    if metric_name not in self.metrics["local_metrics"][client_id]:
                        self.metrics["local_metrics"][client_id][metric_name] = []
                    
                    # Extend list if needed
                    while len(self.metrics["local_metrics"][client_id][metric_name]) < round_num:
                        self.metrics["local_metrics"][client_id][metric_name].append(None)
                    
                    # Update the metric for this round
                    if len(self.metrics["local_metrics"][client_id][metric_name]) == round_num:
                        self.metrics["local_metrics"][client_id][metric_name].append(metric_value)
                    else:
                        self.metrics["local_metrics"][client_id][metric_name][round_num] = metric_value
        
        # Log the stage
        self.log_stage(f"Round_{round_num}", round_data)
    
    def log_model_performance(self, model_name, metrics, model_type="local"):
        """
        Log performance metrics for a specific model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            model_type: Type of the model ('local' or 'global')
        """
        self.logger.info(f"{model_type.capitalize()} Model '{model_name}' performance: {metrics}")
        
        # Log as a stage
        self.log_stage(f"{model_type}_model_{model_name}_evaluation", metrics)
    
    def save_experiment_data(self):
        """
        Save all experiment data to files.
        """
        # Calculate experiment duration
        duration = time.time() - self.start_time
        self.metadata["duration"] = duration
        self.metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save metadata
        metadata_file = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        # Save metrics as JSON
        metrics_file = os.path.join(self.experiment_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save global metrics as CSV
        if self.metrics["global_metrics"]:
            global_metrics_df = pd.DataFrame(self.metrics["global_metrics"])
            global_metrics_df.insert(0, 'round', range(1, len(global_metrics_df) + 1))
            global_metrics_file = os.path.join(self.experiment_dir, "global_metrics.csv")
            global_metrics_df.to_csv(global_metrics_file, index=False)
        
        # Save local metrics as CSV (one file per client)
        for client_id, metrics in self.metrics["local_metrics"].items():
            client_metrics_df = pd.DataFrame(metrics)
            client_metrics_df.insert(0, 'round', range(1, len(client_metrics_df) + 1))
            client_metrics_file = os.path.join(self.experiment_dir, f"client_{client_id}_metrics.csv")
            client_metrics_df.to_csv(client_metrics_file, index=False)
        
        self.logger.info(f"Experiment data saved to {self.experiment_dir}")
    
    def plot_global_metric_evolution(self, metric_name, save_path=None, figsize=(10, 6)):
        """
        Plot the evolution of a global metric across rounds.
        
        Args:
            metric_name: Name of the metric to plot
            save_path: Path to save the plot (None to use default)
            figsize: Figure size
            
        Returns:
            Figure object
        """
        if metric_name not in self.metrics["global_metrics"]:
            self.logger.error(f"Metric '{metric_name}' not found in global metrics")
            return None
        
        plt.figure(figsize=figsize)
        
        rounds = range(1, len(self.metrics["global_metrics"][metric_name]) + 1)
        metric_values = self.metrics["global_metrics"][metric_name]
        
        plt.plot(rounds, metric_values, 'o-', label=f'Global {metric_name}')
        
        plt.xlabel('Communication Round')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'Global {metric_name.capitalize()} Evolution')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, f"global_{metric_name}_evolution.png")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_local_vs_global_metrics(self, metric_name, save_path=None, figsize=(12, 8)):
        """
        Plot local vs global metrics evolution.
        
        Args:
            metric_name: Name of the metric to plot
            save_path: Path to save the plot (None to use default)
            figsize: Figure size
            
        Returns:
            Figure object
        """
        plt.figure(figsize=figsize)
        
        # Plot global metric
        if metric_name in self.metrics["global_metrics"]:
            rounds = range(1, len(self.metrics["global_metrics"][metric_name]) + 1)
            metric_values = self.metrics["global_metrics"][metric_name]
            
            plt.plot(rounds, metric_values, 'o-', linewidth=2, label=f'Global')
        
        # Plot local metrics
        for client_id, metrics in self.metrics["local_metrics"].items():
            if metric_name in metrics:
                rounds = range(1, len(metrics[metric_name]) + 1)
                metric_values = metrics[metric_name]
                
                plt.plot(rounds, metric_values, '--', alpha=0.7, label=f'Client {client_id}')
        
        plt.xlabel('Communication Round')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'Local vs Global {metric_name.capitalize()} Comparison')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, f"local_vs_global_{metric_name}.png")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_all_metrics(self, metrics_to_plot=None):
        """
        Plot all metrics evolution.
        
        Args:
            metrics_to_plot: List of metrics to plot (None for all)
        """
        # Get all available metrics
        if metrics_to_plot is None:
            metrics_to_plot = list(self.metrics["global_metrics"].keys())
        
        # Create a subdirectory for plots
        plots_dir = os.path.join(self.experiment_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot each metric
        for metric in metrics_to_plot:
            # Plot global metric evolution
            if metric in self.metrics["global_metrics"]:
                save_path = os.path.join(plots_dir, f"global_{metric}_evolution.png")
                self.plot_global_metric_evolution(metric, save_path)
            
            # Plot local vs global comparison
            save_path = os.path.join(plots_dir, f"local_vs_global_{metric}.png")
            self.plot_local_vs_global_metrics(metric, save_path)
        
        self.logger.info(f"All metric plots saved to {plots_dir}")
    
    def log_exception(self, exception, stage=None):
        """
        Log an exception.
        
        Args:
            exception: The exception object
            stage: Current stage when the exception occurred (optional)
        """
        error_message = f"Exception: {str(exception)}"
        if stage:
            error_message = f"Exception in stage '{stage}': {str(exception)}"
            
        self.logger.error(error_message, exc_info=True)
        
    def close(self):
        """
        Close the logger and save all data.
        """
        self.save_experiment_data()
        
        # Plot all metrics
        try:
            self.plot_all_metrics()
        except Exception as e:
            self.logger.error(f"Error plotting metrics: {str(e)}", exc_info=True)
        
        self.logger.info(f"Experiment '{self.experiment_name}' completed")
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler) 