"""
LightGBM model implementation for HETROFL.
"""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import optuna
import pandas as pd

from local_models.base_model import BaseLocalModel
from config.config import RANDOM_STATE


class LightGBMModel(BaseLocalModel):
    """
    LightGBM classifier model for federated learning.
    """
    
    def __init__(self, client_id, output_dim=None):
        """
        Initialize the LightGBM model.
        
        Args:
            client_id: Client ID
            output_dim: Number of output classes
        """
        super().__init__("lightgbm", client_id, output_dim)
        self.param_grid = {
            'num_leaves': [31, 50, 100],
            'max_depth': [-1, 5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200, 300],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    
    def build_model(self, **kwargs):
        """
        Build the LightGBM model.
        
        Args:
            **kwargs: Additional model parameters
            
        Returns:
            LightGBM classifier
        """
        if self.output_dim is None:
            raise ValueError("output_dim must be specified for LightGBM model")
        
        # Default parameters
        params = {
            'objective': 'multiclass' if self.output_dim > 2 else 'binary',
            'num_class': self.output_dim if self.output_dim > 2 else None,
            'metric': 'multi_logloss' if self.output_dim > 2 else 'binary_logloss',
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': -1,
            'random_state': RANDOM_STATE,
            'class_weight': 'balanced'
        }
        
        # Update parameters with best params if available
        if self.best_params:
            params.update(self.best_params)
        
        # Update parameters with any provided kwargs
        params.update(kwargs)
        
        # Remove None values for LightGBM
        params = {k: v for k, v in params.items() if v is not None}
        
        # Create the LightGBM classifier
        self.model = lgb.LGBMClassifier(**params)
        
        return self.model
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Tune the LightGBM hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Best parameters
        """
        def param_space(trial):
            """Define the parameter space for Optuna"""
            return {
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 30, 500),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 0, 1.0)
            }
        
        def objective(trial, X_train, y_train, X_val, y_val, **kwargs):
            """Optuna objective function"""
            # Get parameters for this trial
            params = param_space(trial)
            
            # Build the model with these parameters
            model = self.build_model(**params)
            
            # Train the model
            callbacks = [lgb.early_stopping(10, verbose=False)]
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            
            # Get predictions on validation set
            y_pred = model.predict(X_val)
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Return the score to optimize
            return (accuracy + f1) / 2
        
        # Run the hyperparameter optimization
        best_params = self.optimize_hyperparameters(
            X_train, y_train, X_val, y_val,
            param_space=param_space,
            objective_fn=objective,
            direction='maximize'
        )
        
        return best_params
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        # If output_dim not set, determine it from the data
        if self.output_dim is None:
            self.output_dim = len(np.unique(y_train))
            print(f"Setting output_dim to {self.output_dim} based on training data")
        
        # Get early stopping rounds from kwargs if provided
        early_stopping_rounds = kwargs.pop('early_stopping_rounds', 10) if 'early_stopping_rounds' in kwargs else 10
        verbose = kwargs.pop('verbose', False) if 'verbose' in kwargs else False
        
        # Remove epochs if present (not used by LightGBM)
        if 'epochs' in kwargs:
            kwargs.pop('epochs')
        
        # Build the model if it hasn't been built yet
        if self.model is None:
            self.build_model(**kwargs)
        
        # Set up evaluation data
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # LightGBM uses callbacks for early stopping
        callbacks = []
        if eval_set:
            # Create early stopping callback
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose))
            callbacks.append(lgb.log_evaluation(period=10, show_stdv=False) if verbose else None)
            callbacks = [cb for cb in callbacks if cb is not None]  # Remove None values
        
        # Train the model without early_stopping_rounds argument
        fit_params = {
            'eval_set': eval_set,
            'callbacks': callbacks if callbacks else None
        }
        
        # Remove None values
        fit_params = {k: v for k, v in fit_params.items() if v is not None}
        
        # Train the model with the correct parameters
        self.model.fit(X_train, y_train, **fit_params)
        
        # Store evaluation metrics if available
        if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
            results = self.model.evals_result_
            if eval_set and 'valid_0' in results:
                for metric, values in results['valid_0'].items():
                    metric_name = 'val_' + metric
                    if metric_name not in self.training_history:
                        self.training_history[metric_name] = []
                    self.training_history[metric_name].extend(values)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        
        self.training_history['accuracy'] = [train_accuracy]
        self.training_history['f1_score'] = [train_f1]
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='weighted')
            
            if 'val_accuracy' not in self.training_history:
                self.training_history['val_accuracy'] = []
            if 'val_f1_score' not in self.training_history:
                self.training_history['val_f1_score'] = []
                
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['val_f1_score'].append(val_f1)
            
            print(f"Validation accuracy: {val_accuracy:.4f}, F1 score: {val_f1:.4f}")
        
        self.is_fitted = True
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions using the LightGBM model.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the LightGBM model.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def update_with_knowledge(self, X_data, y_data, global_soft_preds, alpha=0.3):
        """
        Update the LightGBM model with knowledge from the global model.
        
        Args:
            X_data: Feature data for knowledge transfer
            y_data: Target data for evaluation
            global_soft_preds: Soft predictions from the global model
            alpha: Weight for global knowledge (0.0-1.0)
            
        Returns:
            Updated model
        """
        if not self.is_fitted:
            print("LightGBM model not fitted. Cannot update with knowledge.")
            return self.model
            
        print(f"Updating LightGBM model for client {self.client_id} with global knowledge...")
        
        # Convert data if needed
        if isinstance(X_data, pd.DataFrame) or isinstance(X_data, pd.Series):
            X_data_np = X_data.values
        else:
            X_data_np = X_data
            
        if isinstance(y_data, pd.Series):
            y_data_np = y_data.values
        else:
            y_data_np = y_data
        
        # Check for shape mismatch
        if len(y_data_np) != len(global_soft_preds):
            print(f"Shape mismatch detected: Client data has {len(y_data_np)} samples, but global predictions have {len(global_soft_preds)} samples.")
            print("Using a subset of client data that matches the size of global predictions.")
            
            # Use only the first n samples where n is the number of global predictions
            n_samples = min(len(y_data_np), len(global_soft_preds))
            X_data_np = X_data_np[:n_samples]
            y_data_np = y_data_np[:n_samples]
            global_soft_preds = global_soft_preds[:n_samples]
            
            print(f"Using {n_samples} samples for knowledge transfer.")
        
        # Create a blend of original labels and global predictions
        num_classes = self.output_dim
        
        # Convert hard labels to one-hot encoding
        y_one_hot = np.zeros((len(y_data_np), num_classes))
        for i, label in enumerate(y_data_np):
            y_one_hot[i, int(label)] = 1
            
        # Create blended targets
        blended_targets = (1 - alpha) * y_one_hot + alpha * global_soft_preds
        
        # Get the most likely class from the blended targets
        y_blend = np.argmax(blended_targets, axis=1)
        
        # Create a dataset with blended targets
        lgb_train = lgb.Dataset(X_data_np, y_blend)
        
        # Get the current model parameters
        params = self.model.get_params()
        
        # Extract number of trees and learning rate for incremental learning
        n_estimators = 10  # Small number for incremental update
        learning_rate = params.get('learning_rate', 0.1) * 0.5  # Halve the learning rate for fine-tuning
        
        # Update only key parameters for incremental learning
        update_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'objective': 'multiclass' if self.output_dim > 2 else 'binary',
            'num_class': self.output_dim if self.output_dim > 2 else None,
            'metric': 'multi_logloss' if self.output_dim > 2 else 'binary_logloss',
            'boosting_type': params.get('boosting_type', 'gbdt'),
            'verbose': -1
        }
        
        # Remove None values
        update_params = {k: v for k, v in update_params.items() if v is not None}
        
        # For LightGBM, create a new model with the updated parameters and train from scratch
        # with the existing model as initial model
        update_model = lgb.LGBMClassifier(**update_params)
        
        # For LightGBM, incremental learning is achieved by init_model parameter
        # We need to save the current model to a temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_model_file = f.name
        
        # Save current model to temp file
        self.model.booster_.save_model(temp_model_file)
        
        try:
            # Fit the update model using the current model as starting point
            update_model.fit(
                X_data_np, y_blend,
                init_model=temp_model_file,  # Use existing model as starting point
                callbacks=[lgb.log_evaluation(period=0)]  # Disable verbose output
            )
            
            # Update the model
            self.model = update_model
            
            # Evaluate updated model
            predictions = self.predict(X_data_np)
            accuracy = accuracy_score(y_data_np, predictions)
            print(f"Updated LightGBM model accuracy: {accuracy:.4f}")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_model_file):
                os.remove(temp_model_file)
        
        return self.model 