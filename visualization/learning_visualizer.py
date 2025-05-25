"""
Visualization module for tracking learning progress in the HETROFL system.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class LearningVisualizer:
    """
    Visualizes learning progress and model performance.
    """
    
    def __init__(self, save_dir=None):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def plot_learning_curves(self, history_dict, model_name=None, save_path=None):
        """
        Plot learning curves from training history.
        
        Args:
            history_dict: Dictionary containing training history
            model_name: Name of the model for the plot title
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        gs = GridSpec(2, 2, figure=plt.gcf())
        ax1 = plt.subplot(gs[0, :])  # Top row, spans both columns
        ax2 = plt.subplot(gs[1, 0])  # Bottom left
        ax3 = plt.subplot(gs[1, 1])  # Bottom right
        
        # Plot accuracy
        if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
            epochs = range(1, len(history_dict['accuracy']) + 1)
            ax1.plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy')
            ax1.plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy')
            ax1.set_title(f'Accuracy vs. Epochs {model_name or ""}')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot loss
        if 'loss' in history_dict and 'val_loss' in history_dict:
            epochs = range(1, len(history_dict['loss']) + 1)
            ax2.plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
            ax2.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss')
            ax2.set_title('Loss vs. Epochs')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot F1 score
        if 'f1_score' in history_dict and 'val_f1_score' in history_dict:
            epochs = range(1, len(history_dict['f1_score']) + 1)
            ax3.plot(epochs, history_dict['f1_score'], 'b-', label='Training F1')
            ax3.plot(epochs, history_dict['val_f1_score'], 'r-', label='Validation F1')
            ax3.set_title('F1 Score vs. Epochs')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('F1 Score')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_feature_importance(self, importance_df, model_name=None, top_n=20, save_path=None):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importances
            model_name: Name of the model for the plot title
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        if importance_df is None or len(importance_df) == 0:
            return None
        
        # Take top N features
        if len(importance_df) > top_n:
            importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Check if 'importance' or 'coefficient' is in the columns
        if 'importance' in importance_df.columns:
            importance_col = 'importance'
        elif 'coefficient' in importance_df.columns:
            importance_col = 'coefficient'
        elif 'mean_abs_coef' in importance_df.columns:
            importance_col = 'mean_abs_coef'
        else:
            importance_col = importance_df.columns[1]  # Assume second column is importance
        
        # Sort by importance
        importance_df = importance_df.sort_values(importance_col)
        
        # Create horizontal bar plot
        sns.barplot(x=importance_df[importance_col], y=importance_df['feature'], palette='viridis')
        
        title = f"Feature Importance" + (f" - {model_name}" if model_name else "")
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_embedding_visualization(self, X, y, model_name=None, method='tsne', save_path=None):
        """
        Visualize high-dimensional data in 2D using t-SNE or PCA.
        
        Args:
            X: Feature data
            y: Labels
            model_name: Name of the model for the plot title
            method: Dimensionality reduction method ('tsne' or 'pca')
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Convert to numpy arrays
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            title_prefix = 't-SNE'
        else:
            reducer = PCA(n_components=2, random_state=42)
            title_prefix = 'PCA'
        
        # Reduce dimensionality
        X_reduced = reducer.fit_transform(X_np)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Get unique labels
        unique_labels = np.unique(y_np)
        
        # Create a scatter plot for each class
        for label in unique_labels:
            idx = y_np == label
            plt.scatter(
                X_reduced[idx, 0], X_reduced[idx, 1],
                label=f'Class {label}',
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5
            )
        
        title = f"{title_prefix} Visualization" + (f" - {model_name}" if model_name else "")
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_model_comparison(self, metrics_dict, metric_name='accuracy', save_path=None):
        """
        Compare models based on a specific metric.
        
        Args:
            metrics_dict: Dictionary of model metrics
            metric_name: Name of the metric to compare
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Extract model names and metric values
        model_names = list(metrics_dict.keys())
        
        # Check if metrics_dict values are dictionaries or direct values
        if isinstance(list(metrics_dict.values())[0], dict):
            # If values are dictionaries, extract the specific metric
            metric_values = [metrics_dict[model][metric_name] for model in model_names]
        else:
            # If values are direct values (floats), use them directly
            metric_values = [metrics_dict[model] for model in model_names]
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Model': model_names,
            metric_name.capitalize(): metric_values
        })
        
        # Sort by metric value
        df = df.sort_values(metric_name.capitalize(), ascending=False)
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        ax = sns.barplot(x='Model', y=metric_name.capitalize(), data=df, palette='viridis')
        
        # Add value labels on top of bars
        for i, v in enumerate(df[metric_name.capitalize()]):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.title(f'Model Comparison by {metric_name.capitalize()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_learning_progression(self, rounds, values, metric_name='accuracy', model_name='Model', 
                            save_path=None):
        """
        Plot the progression of a metric across rounds.
        
        Args:
            rounds: List of round numbers
            values: List of metric values for each round
            metric_name: Name of the metric
            model_name: Name of the model
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, values, marker='o', linestyle='-', linewidth=2, markersize=8)
            
            # Add a trend line
            if len(rounds) > 2:
                z = np.polyfit(rounds, values, 1)
                p = np.poly1d(z)
                plt.plot(rounds, p(rounds), "r--", alpha=0.8, linewidth=1)
            
            plt.title(f"{model_name} {metric_name.capitalize()} Progression", fontsize=15)
            plt.xlabel("Round", fontsize=12)
            plt.ylabel(f"{metric_name.capitalize()}", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels at each point
            for i, j in zip(rounds, values):
                plt.annotate(f"{j:.4f}", xy=(i, j), xytext=(0, 5), 
                             textcoords='offset points', ha='center')
            
            # Add improvement percentages between consecutive rounds
            if len(rounds) > 1:
                for i in range(1, len(rounds)):
                    improvement = values[i] - values[i-1]
                    improvement_pct = improvement * 100
                    
                    # Only show if there's meaningful improvement
                    if abs(improvement) > 0.0001:
                        plt.annotate(f"{improvement_pct:+.2f}%", 
                                    xy=((rounds[i] + rounds[i-1])/2, (values[i] + values[i-1])/2), 
                                    xytext=(0, -15 if i % 2 == 0 else 15),
                                    textcoords='offset points', ha='center',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
            
            plt.tight_layout()
            
            # Save the figure if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved learning progression plot to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Error creating learning progression plot: {e}")
            return None
        finally:
            plt.close()
    
    def plot_multi_model_progression(self, rounds, model_data, metric_name='accuracy', 
                                    save_path=None):
        """
        Plot the progression of a metric across rounds for multiple models.
        
        Args:
            rounds: List of round numbers
            model_data: Dictionary mapping model names to lists of metric values
            metric_name: Name of the metric
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Set up a color cycle for different models
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            
            # Plot each model's progression
            for i, (model_name, values) in enumerate(model_data.items()):
                color = color_cycle[i % len(color_cycle)]
                plt.plot(rounds, values, marker='o', linestyle='-', 
                         linewidth=2, markersize=6, label=model_name, color=color)
                
                # Add final value annotation
                if len(values) > 0:
                    plt.annotate(f"{values[-1]:.4f}", xy=(rounds[-1], values[-1]), 
                                 xytext=(5, 0), textcoords='offset points', 
                                 color=color, fontweight='bold')
            
            plt.title(f"Model {metric_name.capitalize()} Comparison Across Rounds", fontsize=15)
            plt.xlabel("Round", fontsize=12)
            plt.ylabel(f"{metric_name.capitalize()}", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            # Add improvement indicator from first to last round
            for model_name, values in model_data.items():
                if len(values) > 1:
                    total_improvement = values[-1] - values[0]
                    total_improvement_pct = total_improvement * 100
                    
                    # Only show if there's meaningful improvement
                    if abs(total_improvement) > 0.001:
                        y_pos = (values[0] + values[-1]) / 2
                        plt.annotate(f"{model_name}: {total_improvement_pct:+.2f}%", 
                                    xy=(rounds[-1], y_pos), 
                                    xytext=(10, 0),
                                    textcoords='offset points', ha='left',
                                    bbox=dict(boxstyle="round,pad=0.3", 
                                              fc="yellow" if total_improvement > 0 else "red", 
                                              alpha=0.3))
            
            plt.tight_layout()
            
            # Save the figure if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved multi-model progression plot to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Error creating multi-model progression plot: {e}")
            return None
        finally:
            plt.close()
            
    def plot_knowledge_transfer_impact(self, pre_metrics, post_metrics, save_path=None):
        """
        Plot the impact of knowledge transfer on model performance.
        
        Args:
            pre_metrics: Dictionary of metrics before knowledge transfer
            post_metrics: Dictionary of metrics after knowledge transfer
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        try:
            # Extract accuracy values before and after knowledge transfer
            models = []
            pre_accuracies = []
            post_accuracies = []
            improvements = []
            
            for client_id in pre_metrics:
                if client_id in post_metrics:
                    pre_acc = pre_metrics[client_id].get('accuracy', 0)
                    post_acc = post_metrics[client_id].get('accuracy', 0)
                    
                    models.append(f"Client {client_id}")
                    pre_accuracies.append(pre_acc)
                    post_accuracies.append(post_acc)
                    improvements.append(post_acc - pre_acc)
            
            if not models:  # No data to plot
                print("No data for knowledge transfer impact visualization")
                return None
                
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Bar chart showing before/after accuracies
            x = np.arange(len(models))
            width = 0.35
            
            ax1.bar(x - width/2, pre_accuracies, width, label='Pre-Transfer', color='skyblue')
            ax1.bar(x + width/2, post_accuracies, width, label='Post-Transfer', color='lightcoral')
            
            ax1.set_title('Accuracy Before and After Knowledge Transfer', fontsize=15)
            ax1.set_xlabel('Model', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(pre_accuracies):
                ax1.text(i - width/2, v + 0.01, f"{v:.4f}", ha='center', fontsize=9)
            
            for i, v in enumerate(post_accuracies):
                ax1.text(i + width/2, v + 0.01, f"{v:.4f}", ha='center', fontsize=9)
            
            # Bar chart showing improvements
            colors = ['lightgreen' if i > 0 else 'lightcoral' for i in improvements]
            ax2.bar(x, improvements, width, color=colors)
            
            ax2.set_title('Accuracy Improvement from Knowledge Transfer', fontsize=15)
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_ylabel('Improvement', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(improvements):
                sign = '+' if v >= 0 else ''
                ax2.text(i, v + 0.002 if v >= 0 else v - 0.01, 
                       f"{sign}{v:.4f}", ha='center', fontsize=9)
            
            plt.tight_layout()
            
            # Save the figure if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved knowledge transfer impact plot to {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error creating knowledge transfer impact plot: {e}")
            return None
        finally:
            plt.close()
    
    def plot_cumulative_learning_progress(self, run_history, metric_name='accuracy', save_path=None):
        """
        Visualize the cumulative learning progress across runs.
        
        Args:
            run_history: Dictionary with run history data
            metric_name: Name of the metric to visualize
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        plt.figure(figsize=(14, 8))
        
        # Check if we have global model data
        if "global_model" in run_history and run_history["global_model"].get("runs"):
            global_runs = run_history["global_model"]["runs"]
            global_metrics = run_history["global_model"][metric_name]
            plt.plot(global_runs, global_metrics, 'b-o', linewidth=2, markersize=8, label='Global Model')
        
        # Plot local models
        if "local_models" in run_history:
            for client_id, history in run_history["local_models"].items():
                if history.get("runs"):
                    runs = history["runs"]
                    metrics = history[metric_name]
                    plt.plot(runs, metrics, 'o-', label=f'Client {client_id}')
        
        plt.title(f'Cumulative Learning Progress - {metric_name.capitalize()}')
        plt.xlabel('Run Number')
        plt.ylabel(metric_name.capitalize())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_dashboard(self, metrics_data, history_data, save_dir=None):
        """
        Create a comprehensive dashboard of visualizations.
        
        Args:
            metrics_data: Dictionary of metrics data
            history_data: Dictionary of training history data
            save_dir: Directory to save the dashboard
            
        Returns:
            Dictionary of paths to saved visualizations
        """
        if save_dir is None:
            save_dir = self.save_dir or 'dashboard'
        
        os.makedirs(save_dir, exist_ok=True)
        
        dashboard_paths = {}
        
        # Plot cumulative learning progress
        if history_data:
            progress_path = os.path.join(save_dir, 'cumulative_learning_progress.png')
            self.plot_cumulative_learning_progress(history_data, save_path=progress_path)
            dashboard_paths['progress'] = progress_path
        
        # Plot model comparison
        if metrics_data:
            comparison_path = os.path.join(save_dir, 'model_comparison.png')
            self.plot_model_comparison(metrics_data, save_path=comparison_path)
            dashboard_paths['comparison'] = comparison_path
        
        # Plot learning curves for each model
        if history_data and "local_models" in history_data:
            for client_id, model_history in history_data["local_models"].items():
                curves_path = os.path.join(save_dir, f'learning_curves_client_{client_id}.png')
                self.plot_learning_curves(model_history, model_name=f'Client {client_id}', save_path=curves_path)
                dashboard_paths[f'curves_client_{client_id}'] = curves_path
        
        return dashboard_paths 