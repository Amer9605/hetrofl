"""
Analysis utilities for evaluating the cumulative learning progress.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils.model_persistence import ModelTracker
from visualization.learning_visualizer import LearningVisualizer


class LearningAnalyzer:
    """
    Analyzes the cumulative learning progress across multiple runs.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.model_tracker = ModelTracker()
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("results", "analysis", f"analysis_{timestamp}")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.visualizer = LearningVisualizer(save_dir=os.path.join(output_dir, "plots"))
    
    def analyze_performance_trends(self):
        """
        Analyze performance trends across runs.
        
        Returns:
            Dictionary with trend analysis results
        """
        print("Analyzing performance trends...")
        
        # Get performance history
        history = self.model_tracker.performance_history
        
        trends = {
            "global_model": {},
            "local_models": {}
        }
        
        # Analyze global model trends
        global_history = history.get("global_model", {})
        if global_history and len(global_history.get("accuracy", [])) > 1:
            accuracy = global_history["accuracy"]
            f1_weighted = global_history["f1_weighted"]
            
            # Calculate trend metrics
            trends["global_model"] = {
                "total_runs": len(accuracy),
                "initial_accuracy": accuracy[0],
                "final_accuracy": accuracy[-1],
                "max_accuracy": max(accuracy),
                "min_accuracy": min(accuracy),
                "accuracy_improvement": accuracy[-1] - accuracy[0],
                "accuracy_improvement_percent": (accuracy[-1] - accuracy[0]) / accuracy[0] * 100 if accuracy[0] > 0 else 0,
                "f1_improvement": f1_weighted[-1] - f1_weighted[0],
                "f1_improvement_percent": (f1_weighted[-1] - f1_weighted[0]) / f1_weighted[0] * 100 if f1_weighted[0] > 0 else 0,
                "accuracy_trend": np.polyfit(range(len(accuracy)), accuracy, 1)[0],  # Slope of the trend line
                "f1_trend": np.polyfit(range(len(f1_weighted)), f1_weighted, 1)[0]
            }
        
        # Analyze local models trends
        local_models = history.get("local_models", {})
        for client_id, client_history in local_models.items():
            if client_history and len(client_history.get("accuracy", [])) > 1:
                accuracy = client_history["accuracy"]
                f1_weighted = client_history["f1_weighted"]
                
                trends["local_models"][client_id] = {
                    "total_runs": len(accuracy),
                    "initial_accuracy": accuracy[0],
                    "final_accuracy": accuracy[-1],
                    "max_accuracy": max(accuracy),
                    "min_accuracy": min(accuracy),
                    "accuracy_improvement": accuracy[-1] - accuracy[0],
                    "accuracy_improvement_percent": (accuracy[-1] - accuracy[0]) / accuracy[0] * 100 if accuracy[0] > 0 else 0,
                    "f1_improvement": f1_weighted[-1] - f1_weighted[0],
                    "f1_improvement_percent": (f1_weighted[-1] - f1_weighted[0]) / f1_weighted[0] * 100 if f1_weighted[0] > 0 else 0,
                    "accuracy_trend": np.polyfit(range(len(accuracy)), accuracy, 1)[0],
                    "f1_trend": np.polyfit(range(len(f1_weighted)), f1_weighted, 1)[0]
                }
        
        # Save trends to CSV
        if trends["global_model"]:
            global_df = pd.DataFrame([trends["global_model"]])
            global_df.to_csv(os.path.join(self.output_dir, "global_model_trends.csv"), index=False)
        
        if trends["local_models"]:
            local_trends = []
            for client_id, client_trends in trends["local_models"].items():
                client_trends["client_id"] = client_id
                local_trends.append(client_trends)
            
            local_df = pd.DataFrame(local_trends)
            local_df.to_csv(os.path.join(self.output_dir, "local_models_trends.csv"), index=False)
        
        return trends
    
    def plot_trend_analysis(self, trends):
        """
        Create visualizations of performance trends.
        
        Args:
            trends: Dictionary with trend analysis results
            
        Returns:
            Dictionary of paths to saved plots
        """
        plot_paths = {}
        
        # Plot global model improvement
        if trends["global_model"]:
            plt.figure(figsize=(10, 6))
            
            # Create bar chart for accuracy and F1 improvement
            labels = ['Accuracy', 'F1 Score']
            improvements = [
                trends["global_model"]["accuracy_improvement"],
                trends["global_model"]["f1_improvement"]
            ]
            
            plt.bar(labels, improvements, color=['blue', 'green'])
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Global Model Improvement')
            plt.ylabel('Absolute Improvement')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(improvements):
                plt.text(i, v + 0.01 if v >= 0 else v - 0.03, f'{v:.4f}', ha='center')
            
            global_path = os.path.join(self.visualizer.save_dir, "global_model_improvement.png")
            plt.savefig(global_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths["global_improvement"] = global_path
        
        # Plot local models improvement comparison
        if trends["local_models"]:
            plt.figure(figsize=(12, 8))
            
            # Extract client IDs and accuracy improvements
            client_ids = list(trends["local_models"].keys())
            accuracy_improvements = [trends["local_models"][client_id]["accuracy_improvement"] for client_id in client_ids]
            f1_improvements = [trends["local_models"][client_id]["f1_improvement"] for client_id in client_ids]
            
            # Set up bar positions
            x = np.arange(len(client_ids))
            width = 0.35
            
            # Create grouped bar chart
            plt.bar(x - width/2, accuracy_improvements, width, label='Accuracy Improvement', color='blue')
            plt.bar(x + width/2, f1_improvements, width, label='F1 Improvement', color='green')
            
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Local Models Improvement')
            plt.xlabel('Client ID')
            plt.ylabel('Absolute Improvement')
            plt.xticks(x, [f'Client {client_id}' for client_id in client_ids])
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            local_path = os.path.join(self.visualizer.save_dir, "local_models_improvement.png")
            plt.savefig(local_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths["local_improvement"] = local_path
            
            # Plot trend lines for each local model
            plt.figure(figsize=(14, 8))
            
            # Get performance history
            history = self.model_tracker.performance_history
            
            # Plot global model trend if available
            global_history = history.get("global_model", {})
            if global_history and "accuracy" in global_history and len(global_history["accuracy"]) > 1:
                runs = global_history["runs"]
                accuracy = global_history["accuracy"]
                
                # Plot actual values
                plt.plot(runs, accuracy, 'bo-', label='Global Model', linewidth=2)
                
                # Plot trend line
                z = np.polyfit(runs, accuracy, 1)
                p = np.poly1d(z)
                plt.plot(runs, p(runs), 'b--', alpha=0.7)
            
            # Plot local model trends
            for client_id, client_history in history.get("local_models", {}).items():
                if "accuracy" in client_history and len(client_history["accuracy"]) > 1:
                    runs = client_history["runs"]
                    accuracy = client_history["accuracy"]
                    
                    # Plot actual values
                    plt.plot(runs, accuracy, 'o-', label=f'Client {client_id}')
                    
                    # Plot trend line
                    z = np.polyfit(runs, accuracy, 1)
                    p = np.poly1d(z)
                    plt.plot(runs, p(runs), '--', alpha=0.7)
            
            plt.title('Accuracy Trends Over Runs')
            plt.xlabel('Run Number')
            plt.ylabel('Accuracy')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            trends_path = os.path.join(self.visualizer.save_dir, "accuracy_trends.png")
            plt.savefig(trends_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths["accuracy_trends"] = trends_path
        
        return plot_paths
    
    def analyze_knowledge_transfer(self):
        """
        Analyze the effectiveness of knowledge transfer between global and local models.
        
        Returns:
            Dictionary with knowledge transfer analysis
        """
        # Get performance history
        history = self.model_tracker.performance_history
        
        # Initialize results
        transfer_analysis = {
            "global_to_local": {},
            "local_to_global": {}
        }
        
        # Check if we have both global and local model data
        if not history.get("global_model", {}).get("accuracy") or not history.get("local_models"):
            return transfer_analysis
        
        # Get global model data
        global_accuracy = history["global_model"]["accuracy"]
        global_runs = history["global_model"]["runs"]
        
        # Analyze global to local knowledge transfer
        for client_id, client_history in history.get("local_models", {}).items():
            if "accuracy" in client_history and len(client_history["accuracy"]) > 1:
                local_accuracy = client_history["accuracy"]
                local_runs = client_history["runs"]
                
                # Calculate correlation between global and local performance
                # We need to align the runs first
                common_runs = set(global_runs).intersection(set(local_runs))
                if common_runs:
                    global_values = []
                    local_values = []
                    
                    for run in sorted(common_runs):
                        global_idx = global_runs.index(run)
                        local_idx = local_runs.index(run)
                        
                        global_values.append(global_accuracy[global_idx])
                        local_values.append(local_accuracy[local_idx])
                    
                    if len(global_values) > 1:
                        correlation = np.corrcoef(global_values, local_values)[0, 1]
                        
                        transfer_analysis["global_to_local"][client_id] = {
                            "correlation": correlation,
                            "global_improvement": global_values[-1] - global_values[0],
                            "local_improvement": local_values[-1] - local_values[0],
                            "transfer_efficiency": (local_values[-1] - local_values[0]) / (global_values[-1] - global_values[0]) if global_values[-1] != global_values[0] else 0
                        }
        
        # Save analysis to CSV
        if transfer_analysis["global_to_local"]:
            transfer_data = []
            for client_id, analysis in transfer_analysis["global_to_local"].items():
                analysis["client_id"] = client_id
                transfer_data.append(analysis)
            
            transfer_df = pd.DataFrame(transfer_data)
            transfer_df.to_csv(os.path.join(self.output_dir, "knowledge_transfer_analysis.csv"), index=False)
        
        return transfer_analysis
    
    def plot_knowledge_transfer_analysis(self, transfer_analysis):
        """
        Create visualizations of knowledge transfer analysis.
        
        Args:
            transfer_analysis: Dictionary with knowledge transfer analysis
            
        Returns:
            Dictionary of paths to saved plots
        """
        plot_paths = {}
        
        # Plot global to local knowledge transfer correlation
        if transfer_analysis["global_to_local"]:
            plt.figure(figsize=(10, 6))
            
            client_ids = []
            correlations = []
            transfer_efficiencies = []
            
            for client_id, analysis in transfer_analysis["global_to_local"].items():
                client_ids.append(f'Client {client_id}')
                correlations.append(analysis["correlation"])
                transfer_efficiencies.append(analysis["transfer_efficiency"])
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Client': client_ids,
                'Correlation': correlations,
                'Transfer Efficiency': transfer_efficiencies
            })
            
            # Plot correlation
            plt.subplot(1, 2, 1)
            sns.barplot(x='Client', y='Correlation', data=df, palette='viridis')
            plt.title('Global-Local Performance Correlation')
            plt.ylabel('Correlation Coefficient')
            plt.ylim(-1, 1)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Plot transfer efficiency
            plt.subplot(1, 2, 2)
            sns.barplot(x='Client', y='Transfer Efficiency', data=df, palette='viridis')
            plt.title('Knowledge Transfer Efficiency')
            plt.ylabel('Transfer Efficiency')
            plt.axhline(y=1, color='g', linestyle='-', alpha=0.3)  # Efficiency = 1 means perfect transfer
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Efficiency = 0 means no transfer
            
            plt.tight_layout()
            
            transfer_path = os.path.join(self.visualizer.save_dir, "knowledge_transfer_analysis.png")
            plt.savefig(transfer_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths["transfer_analysis"] = transfer_path
        
        return plot_paths
    
    def run_full_analysis(self):
        """
        Run a complete analysis of the cumulative learning progress.
        
        Returns:
            Dictionary with all analysis results
        """
        print("Running full analysis of cumulative learning progress...")
        
        # Analyze performance trends
        trends = self.analyze_performance_trends()
        
        # Plot trend analysis
        trend_plots = self.plot_trend_analysis(trends)
        
        # Analyze knowledge transfer
        transfer_analysis = self.analyze_knowledge_transfer()
        
        # Plot knowledge transfer analysis
        transfer_plots = self.plot_knowledge_transfer_analysis(transfer_analysis)
        
        # Generate performance history plots
        history_plots = self.model_tracker.plot_performance_history(
            save_dir=os.path.join(self.visualizer.save_dir, "history")
        )
        
        # Combine all results
        results = {
            "trends": trends,
            "transfer_analysis": transfer_analysis,
            "plots": {
                "trends": trend_plots,
                "transfer": transfer_plots,
                "history": history_plots
            }
        }
        
        # Save summary to text file
        with open(os.path.join(self.output_dir, "analysis_summary.txt"), "w") as f:
            f.write("CUMULATIVE LEARNING ANALYSIS SUMMARY\n")
            f.write("===================================\n\n")
            
            f.write("Global Model Performance:\n")
            if trends["global_model"]:
                f.write(f"  Total runs: {trends['global_model']['total_runs']}\n")
                f.write(f"  Initial accuracy: {trends['global_model']['initial_accuracy']:.4f}\n")
                f.write(f"  Final accuracy: {trends['global_model']['final_accuracy']:.4f}\n")
                f.write(f"  Accuracy improvement: {trends['global_model']['accuracy_improvement']:.4f} ({trends['global_model']['accuracy_improvement_percent']:.2f}%)\n")
                f.write(f"  F1 improvement: {trends['global_model']['f1_improvement']:.4f} ({trends['global_model']['f1_improvement_percent']:.2f}%)\n")
                f.write(f"  Accuracy trend slope: {trends['global_model']['accuracy_trend']:.6f}\n\n")
            else:
                f.write("  No data available\n\n")
            
            f.write("Local Models Performance:\n")
            if trends["local_models"]:
                for client_id, client_trends in trends["local_models"].items():
                    f.write(f"  Client {client_id}:\n")
                    f.write(f"    Initial accuracy: {client_trends['initial_accuracy']:.4f}\n")
                    f.write(f"    Final accuracy: {client_trends['final_accuracy']:.4f}\n")
                    f.write(f"    Accuracy improvement: {client_trends['accuracy_improvement']:.4f} ({client_trends['accuracy_improvement_percent']:.2f}%)\n")
                    f.write(f"    F1 improvement: {client_trends['f1_improvement']:.4f} ({client_trends['f1_improvement_percent']:.2f}%)\n")
                    f.write(f"    Accuracy trend slope: {client_trends['accuracy_trend']:.6f}\n\n")
            else:
                f.write("  No data available\n\n")
            
            f.write("Knowledge Transfer Analysis:\n")
            if transfer_analysis["global_to_local"]:
                for client_id, analysis in transfer_analysis["global_to_local"].items():
                    f.write(f"  Client {client_id}:\n")
                    f.write(f"    Correlation with global model: {analysis['correlation']:.4f}\n")
                    f.write(f"    Transfer efficiency: {analysis['transfer_efficiency']:.4f}\n\n")
            else:
                f.write("  No data available\n\n")
            
            f.write("Analysis completed at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        
        print(f"Analysis completed. Results saved to: {self.output_dir}")
        
        return results


if __name__ == "__main__":
    # Run analysis when script is executed directly
    analyzer = LearningAnalyzer()
    analyzer.run_full_analysis() 