"""
Metrics utility for model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import os
from pathlib import Path

class ModelEvaluator:
    """
    Class for evaluating and visualizing model performance.
    """
    
    def __init__(self, class_names=None):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class names for visualization
        """
        self.class_names = class_names
        self.metrics_history = {}
    
    def evaluate_classifier(self, y_true, y_pred, y_prob=None, model_name=None):
        """
        Evaluate a classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            model_name: Name of the model for tracking
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert inputs to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Per-class metrics if multiclass
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        for label in unique_labels:
            label_name = self.class_names[label] if self.class_names is not None and label < len(self.class_names) else f"class_{label}"
            metrics[f'precision_{label_name}'] = precision_score(y_true, y_pred, labels=[label], average='micro')
            metrics[f'recall_{label_name}'] = recall_score(y_true, y_pred, labels=[label], average='micro')
            metrics[f'f1_{label_name}'] = f1_score(y_true, y_pred, labels=[label], average='micro')
        
        # ROC AUC and PR AUC if probabilities are provided
        if y_prob is not None:
            # Handle both binary and multiclass cases
            if y_prob.ndim > 1 and y_prob.shape[1] > 2:  # multiclass
                # One-vs-Rest ROC AUC
                n_classes = y_prob.shape[1]
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    y_true_binary = (y_true == i).astype(int)
                    fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(
                    pd.get_dummies(y_true).values.ravel(), 
                    y_prob.ravel()
                )
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                metrics['roc_auc_micro'] = roc_auc["micro"]
                
                # Compute macro-average ROC AUC
                metrics['roc_auc_macro'] = np.mean(list(roc_auc.values())[:-1])  # Exclude the micro AUC
                
                # Store per-class AUC
                for i in range(n_classes):
                    class_name = self.class_names[i] if self.class_names is not None and i < len(self.class_names) else f"class_{i}"
                    metrics[f'roc_auc_{class_name}'] = roc_auc[i]
            
            else:  # binary
                if y_prob.ndim > 1:
                    # Take the probability of the positive class
                    y_prob_binary = y_prob[:, 1]
                else:
                    y_prob_binary = y_prob
                
                fpr, tpr, _ = roc_curve(y_true, y_prob_binary)
                metrics['roc_auc'] = auc(fpr, tpr)
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_true, y_prob_binary)
                metrics['pr_auc'] = auc(recall, precision)
                metrics['avg_precision'] = average_precision_score(y_true, y_prob_binary)
        
        # Store metrics for this model if a name is provided
        if model_name:
            self.metrics_history[model_name] = metrics
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name=None, save_path=None, figsize=(10, 8)):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for the plot title
            save_path: Path to save the plot (optional)
            figsize: Figure size
            
        Returns:
            Figure object
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create labels for the plot
        if self.class_names is not None:
            labels = [str(name) for name in self.class_names]
        else:
            labels = [str(i) for i in range(cm.shape[0])]
        
        # Create figure
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        title = f"Confusion Matrix" + (f" - {model_name}" if model_name else "")
        plt.title(title)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curves(self, y_true, y_prob, model_name=None, save_path=None, figsize=(10, 8)):
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model for the plot title
            save_path: Path to save the plot (optional)
            figsize: Figure size
            
        Returns:
            Figure object
        """
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Check dimensionality of y_prob
        if y_prob.ndim == 1 or y_prob.shape[1] == 2:
            # Binary classification
            if y_prob.ndim > 1:
                y_prob_binary = y_prob[:, 1]
            else:
                y_prob_binary = y_prob
                
            fpr, tpr, _ = roc_curve(y_true, y_prob_binary)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr, 
                label=f'ROC curve (area = {roc_auc:.3f})'
            )
        else:
            # Multi-class classification - One-vs-Rest ROC curves
            n_classes = y_prob.shape[1]
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                y_true_binary = (y_true == i).astype(int)
                fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
                class_name = self.class_names[i] if self.class_names is not None and i < len(self.class_names) else f"Class {i}"
                plt.plot(
                    fpr[i], tpr[i],
                    label=f'ROC curve - {class_name} (area = {roc_auc[i]:.3f})'
                )
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(
                pd.get_dummies(y_true).values.ravel(), 
                y_prob.ravel()
            )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.plot(
                fpr["micro"], tpr["micro"],
                label=f'ROC curve - micro-average (area = {roc_auc["micro"]:.3f})',
                linestyle=':', linewidth=4
            )
            
            # Compute macro-average ROC curve and ROC area
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            
            # Then interpolate all ROC curves at these points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
            plt.plot(
                fpr["macro"], tpr["macro"],
                label=f'ROC curve - macro-average (area = {roc_auc["macro"]:.3f})',
                linestyle='--', linewidth=4
            )
        
        # Add diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title = f"ROC Curves" + (f" - {model_name}" if model_name else "")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, y_true, y_prob, model_name=None, save_path=None, figsize=(10, 8)):
        """
        Plot Precision-Recall curves.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model for the plot title
            save_path: Path to save the plot (optional)
            figsize: Figure size
            
        Returns:
            Figure object
        """
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Check dimensionality of y_prob
        if y_prob.ndim == 1 or y_prob.shape[1] == 2:
            # Binary classification
            if y_prob.ndim > 1:
                y_prob_binary = y_prob[:, 1]
            else:
                y_prob_binary = y_prob
                
            precision, recall, _ = precision_recall_curve(y_true, y_prob_binary)
            pr_auc = auc(recall, precision)
            avg_precision = average_precision_score(y_true, y_prob_binary)
            
            plt.plot(
                recall, precision,
                label=f'PR curve (AP = {avg_precision:.3f}, AUC = {pr_auc:.3f})'
            )
            
            # Reference line (no-skill classifier)
            no_skill = len(y_true[y_true == 1]) / len(y_true)
            plt.plot([0, 1], [no_skill, no_skill], 'k--', label='No Skill')
            
        else:
            # Multi-class classification - One-vs-Rest PR curves
            n_classes = y_prob.shape[1]
            
            # Compute PR curve for each class
            precision = dict()
            recall = dict()
            avg_precision = dict()
            
            for i in range(n_classes):
                y_true_binary = (y_true == i).astype(int)
                precision[i], recall[i], _ = precision_recall_curve(y_true_binary, y_prob[:, i])
                avg_precision[i] = average_precision_score(y_true_binary, y_prob[:, i])
                
                pr_auc = auc(recall[i], precision[i])
                
                class_name = self.class_names[i] if self.class_names is not None and i < len(self.class_names) else f"Class {i}"
                plt.plot(
                    recall[i], precision[i],
                    label=f'PR - {class_name} (AP = {avg_precision[i]:.3f}, AUC = {pr_auc:.3f})'
                )
            
            # Compute micro-averaged PR curve
            y_true_one_hot = pd.get_dummies(y_true).values
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                y_true_one_hot.ravel(), y_prob.ravel()
            )
            avg_precision["micro"] = average_precision_score(
                y_true_one_hot, y_prob, average="micro"
            )
            pr_auc_micro = auc(recall["micro"], precision["micro"])
            
            plt.plot(
                recall["micro"], precision["micro"],
                label=f'PR - micro-average (AP = {avg_precision["micro"]:.3f}, AUC = {pr_auc_micro:.3f})',
                linestyle='--', linewidth=4
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        title = f"Precision-Recall Curves" + (f" - {model_name}" if model_name else "")
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_learning_curves(self, train_metrics, val_metrics, metric_name='accuracy', 
                           model_name=None, save_path=None, figsize=(10, 6)):
        """
        Plot learning curves (training and validation metrics over epochs).
        
        Args:
            train_metrics: List of training metrics for each epoch
            val_metrics: List of validation metrics for each epoch
            metric_name: Name of the metric to plot
            model_name: Name of the model for the plot title
            save_path: Path to save the plot (optional)
            figsize: Figure size
            
        Returns:
            Figure object
        """
        plt.figure(figsize=figsize)
        
        epochs = range(1, len(train_metrics) + 1)
        
        plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
        plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
        
        plt.xlabel('Epochs')
        plt.ylabel(metric_name.capitalize())
        title = f"Learning Curve - {metric_name}" + (f" - {model_name}" if model_name else "")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_model_comparison(self, metric_name='accuracy', models=None, save_path=None, figsize=(12, 8)):
        """
        Plot comparison of different models based on specified metric.
        
        Args:
            metric_name: Name of the metric to compare
            models: List of model names to include (None for all)
            save_path: Path to save the plot (optional)
            figsize: Figure size
            
        Returns:
            Figure object
        """
        if not self.metrics_history:
            raise ValueError("No metrics history available. Evaluate models first.")
        
        # Filter models if specified
        if models is not None:
            model_metrics = {model: self.metrics_history[model] for model in models if model in self.metrics_history}
        else:
            model_metrics = self.metrics_history
        
        if not model_metrics:
            raise ValueError("No matching models found in metrics history.")
        
        # Extract the specified metric for each model
        metric_values = []
        model_names = []
        
        for model, metrics in model_metrics.items():
            if metric_name in metrics:
                metric_values.append(metrics[metric_name])
                model_names.append(model)
        
        # Create the bar plot
        plt.figure(figsize=figsize)
        
        # Sort by metric value
        sorted_indices = np.argsort(metric_values)
        sorted_models = [model_names[i] for i in sorted_indices]
        sorted_values = [metric_values[i] for i in sorted_indices]
        
        bars = plt.barh(sorted_models, sorted_values, color='skyblue')
        
        # Add value labels
        for i, v in enumerate(sorted_values):
            plt.text(v + 0.01, i, f"{v:.4f}", va='center')
        
        plt.xlabel(metric_name.capitalize())
        plt.ylabel('Model')
        plt.title(f"Model Comparison - {metric_name.capitalize()}")
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def print_classification_report(self, y_true, y_pred, model_name=None):
        """
        Print a classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Classification report as a string
        """
        if model_name:
            print(f"Classification Report for {model_name}:")
            
        target_names = self.class_names if self.class_names is not None else None
        report = classification_report(y_true, y_pred, target_names=target_names)
        print(report)
        
        return report
    
    def save_metrics_to_csv(self, save_path):
        """
        Save all metrics history to a CSV file.
        
        Args:
            save_path: Path to save the CSV
            
        Returns:
            DataFrame with metrics
        """
        if not self.metrics_history:
            raise ValueError("No metrics history available. Evaluate models first.")
        
        # Convert metrics history to a DataFrame
        metrics_df = pd.DataFrame(self.metrics_history).T
        metrics_df.index.name = 'model'
        metrics_df.reset_index(inplace=True)
        
        # Save to CSV
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        metrics_df.to_csv(save_path, index=False)
        
        print(f"Metrics saved to {save_path}")
        
        return metrics_df


def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Convenience function to calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: List of class names (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = ModelEvaluator(class_names=class_names)
    return evaluator.evaluate_classifier(y_true, y_pred, y_prob)
