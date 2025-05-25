"""
Base model class for all local models in the HETROFL system.
"""

import os
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import optuna
from pathlib import Path
import pickle

from config.config import MODEL_SAVE_DIR, N_TRIALS, TIMEOUT


class BaseLocalModel(ABC):
    """
    Abstract base class for all local models.
    """
    
    def __init__(self, model_name, client_id, output_dim=None):
        """
        Initialize the base local model.
        
        Args:
            model_name: Name of the model
            client_id: Client ID
            output_dim: Number of output dimensions (for classification, number of classes)
        """
        self.model_name = model_name
        self.client_id = client_id
        self.model = None
        self.output_dim = output_dim
        self.best_params = None
        self.is_fitted = False
        self.training_history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    @abstractmethod
    def build_model(self, **kwargs):
        """
        Build the model architecture.
        
        Args:
            **kwargs: Additional model parameters
            
        Returns:
            Initialized model
        """
        pass
    
    @abstractmethod
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Tune model hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Best parameters
        """
        pass
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        pass
    
    def get_model_size(self):
        """
        Get the size of the model in bytes.
        
        Returns:
            Model size in bytes
        """
        if self.model is None:
            return 0
        
        # For scikit-learn based models
        if isinstance(self.model, BaseEstimator):
            return joblib.dump(self.model, os.devnull)[0]
        
        # Try model parameter count for other models
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            # Rough estimate: 4 bytes per parameter (float32)
            return param_count * 4
        except:
            return 0
    
    def save_model(self, save_dir=None, round_num=None, experiment_id=None, metadata=None):
        """
        Save the model to disk with optional round and experiment information.
        
        Args:
            save_dir: Directory to save the model (default: MODEL_SAVE_DIR)
            round_num: Round number for round-based saving
            experiment_id: Experiment ID for organized saving
            metadata: Additional metadata to save with the model
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Determine save directory
        if save_dir is None:
            if experiment_id and round_num is not None:
                save_dir = os.path.join(MODEL_SAVE_DIR, experiment_id, f"round_{round_num}", 
                                      "local_models", f"client_{self.client_id}_{self.model_name}")
            else:
                save_dir = os.path.join(MODEL_SAVE_DIR, f"client_{self.client_id}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f"{self.model_name}.joblib")
        
        # Save the model using joblib
        joblib.dump(self.model, model_path)
        
        # Save hyperparameters
        if self.best_params:
            params_path = os.path.join(save_dir, f"{self.model_name}_params.joblib")
            joblib.dump(self.best_params, params_path)
        
        # Save training history
        history_path = os.path.join(save_dir, f"{self.model_name}_history.joblib")
        joblib.dump(self.training_history, history_path)
        
        # Save metadata if provided
        if metadata or round_num is not None or experiment_id:
            metadata_dict = {
                "model_name": self.model_name,
                "client_id": self.client_id,
                "round_num": round_num,
                "experiment_id": experiment_id,
                "save_timestamp": pd.Timestamp.now().isoformat(),
                "is_fitted": self.is_fitted,
                "output_dim": self.output_dim
            }
            
            if metadata:
                metadata_dict.update(metadata)
            
            metadata_path = os.path.join(save_dir, "metadata.json")
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, load_dir=None):
        """
        Load the model from disk.
        
        Args:
            load_dir: Directory to load the model from (default: MODEL_SAVE_DIR)
            
        Returns:
            Loaded model
        """
        if load_dir is None:
            load_dir = os.path.join(MODEL_SAVE_DIR, f"client_{self.client_id}")
        
        model_path = os.path.join(load_dir, f"{self.model_name}.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model
        self.model = joblib.load(model_path)
        
        # Load hyperparameters if available
        params_path = os.path.join(load_dir, f"{self.model_name}_params.joblib")
        if os.path.exists(params_path):
            self.best_params = joblib.load(params_path)
        
        # Load training history if available
        history_path = os.path.join(load_dir, f"{self.model_name}_history.joblib")
        if os.path.exists(history_path):
            self.training_history = joblib.load(history_path)
        
        self.is_fitted = True
        
        print(f"Model loaded from {model_path}")
        
        return self.model
    
    def update_training_history(self, epoch_metrics):
        """
        Update the training history with metrics from an epoch.
        
        Args:
            epoch_metrics: Dictionary of metrics for the epoch
        """
        for metric_name, metric_value in epoch_metrics.items():
            if metric_name in self.training_history:
                self.training_history[metric_name].append(metric_value)
            else:
                self.training_history[metric_name] = [metric_value]
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance if the model supports it.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances, or None if not supported
        """
        if self.model is None or not self.is_fitted:
            return None
        
        # For tree-based models that have feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Create a DataFrame for better visualization
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
        
        # For linear models that have coef_
        elif hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
            
            # For binary classification, coef_ is a 1D array
            if len(coefs.shape) == 1:
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(coefs))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefs
                })
                
                # Sort by absolute coefficient value
                importance_df['abs_coefficient'] = np.abs(importance_df['coefficient'])
                importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
                
                return importance_df
            
            # For multiclass, coef_ is a 2D array
            else:
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(coefs.shape[1])]
                
                # Create a DataFrame with one column per class
                importance_df = pd.DataFrame(coefs.T, index=feature_names)
                
                # Add a column for mean absolute coefficient
                importance_df['mean_abs_coef'] = np.abs(coefs).mean(axis=0)
                
                # Sort by mean absolute coefficient
                importance_df = importance_df.sort_values('mean_abs_coef', ascending=False)
                
                return importance_df
        
        return None
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, param_space, objective_fn, direction='maximize', **kwargs):
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            param_space: Function that defines the parameter space
            objective_fn: Function that evaluates a trial
            direction: Optimization direction ('maximize' or 'minimize')
            **kwargs: Additional parameters for the optimization
            
        Returns:
            Best parameters
        """
        print(f"Optimizing hyperparameters for {self.model_name} (client {self.client_id})...")
        
        # Create an Optuna study
        study = optuna.create_study(direction=direction)
        
        # Start the optimization
        study.optimize(
            lambda trial: objective_fn(trial, X_train, y_train, X_val, y_val, **kwargs),
            n_trials=kwargs.get('n_trials', N_TRIALS),
            timeout=kwargs.get('timeout', TIMEOUT),
            show_progress_bar=True
        )
        
        # Get the best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"Best {direction} value: {best_value}")
        print(f"Best parameters: {best_params}")
        
        # Save the best parameters
        self.best_params = best_params
        
        return best_params
    
    def update_with_knowledge(self, X_data, y_data, global_soft_preds, alpha=0.3):
        """
        Update the local model with knowledge from the global model.
        
        Args:
            X_data: Feature data for knowledge transfer
            y_data: Target data for evaluation
            global_soft_preds: Soft predictions from the global model
            alpha: Weight for global knowledge (0.0-1.0)
            
        Returns:
            Updated model
        """
        # Handle shape mismatch between client data and global predictions
        if len(y_data) != len(global_soft_preds):
            print(f"Shape mismatch detected in {self.model_name} for client {self.client_id}:")
            print(f"Client data has {len(y_data)} samples, but global predictions have {len(global_soft_preds)} samples.")
            print("Using a subset of data that matches in size.")
            
            # Use only the first n samples where n is the minimum size
            n_samples = min(len(y_data), len(global_soft_preds))
            
            # Convert data if needed
            if isinstance(X_data, pd.DataFrame) or isinstance(X_data, pd.Series):
                X_data = X_data.iloc[:n_samples]
            else:
                X_data = X_data[:n_samples]
                
            if isinstance(y_data, pd.Series):
                y_data = y_data.iloc[:n_samples]
            else:
                y_data = y_data[:n_samples]
                
            global_soft_preds = global_soft_preds[:n_samples]
            
            print(f"Using {n_samples} samples for knowledge transfer.")
        
        # Default implementation - subclasses should override this method
        # with model-specific knowledge transfer implementations
        print(f"Knowledge transfer not implemented for {self.model_name}")
        return self.model 
