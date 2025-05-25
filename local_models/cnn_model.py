"""
CNN model implementation for HETROFL.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from local_models.base_model import BaseLocalModel
from config.config import RANDOM_STATE, BATCH_SIZE, LEARNING_RATE


class CNN(nn.Module):
    """
    Convolutional Neural Network for tabular data.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2, kernel_size=3):
        """
        Initialize the CNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout_rate: Dropout rate
            kernel_size: Kernel size for convolutional layers
        """
        super(CNN, self).__init__()
        
        # Reshape tabular data into a 2D format suitable for CNN
        self.width = int(np.sqrt(input_dim)) + 1
        self.height = int(np.ceil(input_dim / self.width))
        self.padded_input_dim = self.width * self.height
        
        # Initial layers - reshape and add channel dimension
        self.pre_cnn_layers = nn.Sequential(
            nn.Linear(input_dim, self.padded_input_dim),
            nn.BatchNorm1d(self.padded_input_dim),
            nn.LeakyReLU()
        )
        
        # Build CNN layers
        in_channels = 1
        cnn_layers = []
        current_dim = self.height
        
        for i, hidden_dim in enumerate(hidden_dims):
            out_channels = hidden_dim
            
            # Add CNN block
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size//2)),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ])
            
            in_channels = out_channels
        
        self.cnn_layers = nn.Sequential(*cnn_layers)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output predictions
        """
        # Preprocess input
        x = self.pre_cnn_layers(x)
        
        # Reshape for CNN
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.padded_input_dim)
        
        # Pass through CNN layers
        x = self.cnn_layers(x)
        
        # Global average pooling
        x = self.gap(x).squeeze(-1)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


class CNNModel(BaseLocalModel):
    """
    CNN classifier model for federated learning.
    """
    
    def __init__(self, client_id, input_dim=None, output_dim=None):
        """
        Initialize the CNN model.
        
        Args:
            client_id: Client ID
            input_dim: Input feature dimension
            output_dim: Number of output classes
        """
        super().__init__("cnn", client_id, output_dim)
        self.input_dim = input_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = None
        self.optimizer = None
        self.scaler = None
        
    def build_model(self, **kwargs):
        """
        Build the CNN model.
        
        Args:
            **kwargs: Additional model parameters
            
        Returns:
            CNN model
        """
        if self.input_dim is None:
            raise ValueError("input_dim must be specified for CNN model")
        
        if self.output_dim is None:
            raise ValueError("output_dim must be specified for CNN model")
        
        # Default parameters
        params = {
            'hidden_dims': [32, 64, 128],
            'dropout_rate': 0.2,
            'kernel_size': 3,
            'learning_rate': LEARNING_RATE
        }
        
        # Update parameters with best params if available
        if self.best_params:
            params.update(self.best_params)
        
        # Update parameters with any provided kwargs
        params.update(kwargs)
        
        # Create the CNN model
        self.model = CNN(
            input_dim=self.input_dim,
            hidden_dims=params['hidden_dims'],
            output_dim=self.output_dim,
            dropout_rate=params['dropout_rate'],
            kernel_size=params['kernel_size']
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=params['learning_rate']
        )
        
        return self.model
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Tune the CNN hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Best parameters
        """
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        
        def param_space(trial):
            """Define the parameter space for Optuna"""
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_dims = []
            
            for i in range(n_layers):
                hidden_dims.append(trial.suggest_categorical(f'hidden_dim_{i}', [16, 32, 64, 128]))
            
            return {
                'hidden_dims': hidden_dims,
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256])
            }
        
        def objective(trial, X_train, y_train, X_val, y_val, **kwargs):
            """Optuna objective function"""
            # Get parameters for this trial
            params = param_space(trial)
            
            # Set input dimension if not set
            if self.input_dim is None:
                self.input_dim = X_train.shape[1]
            
            # Build the model with these parameters
            model = self.build_model(**params)
            
            # Create dataloaders with the trial's batch size
            train_loader = DataLoader(
                train_dataset,
                batch_size=params['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=params['batch_size'],
                shuffle=False
            )
            
            # Train for a few epochs
            n_epochs = 5
            for epoch in range(n_epochs):
                # Training
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(val_targets, val_preds)
            f1 = f1_score(val_targets, val_preds, average='weighted')
            
            # Return the score to optimize
            return (accuracy + f1) / 2
        
        # Set input dimension if not set
        if self.input_dim is None:
            self.input_dim = X_train.shape[1]
        
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
        Train the CNN model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        # If dimensions not set, determine them from the data
        if self.input_dim is None:
            self.input_dim = X_train.shape[1]
            print(f"Setting input_dim to {self.input_dim} based on training data")
        
        if self.output_dim is None:
            self.output_dim = len(np.unique(y_train))
            print(f"Setting output_dim to {self.output_dim} based on training data")
        
        # Build the model if it hasn't been built yet
        if self.model is None:
            self.build_model(**kwargs)
        
        # Convert data to tensors
        # Convert pandas Series to numpy arrays if needed
        if isinstance(X_train, pd.Series) or isinstance(X_train, pd.DataFrame):
            X_train_np = X_train.values
        else:
            X_train_np = X_train
            
        if isinstance(y_train, pd.Series):
            y_train_np = y_train.values
        else:
            y_train_np = y_train
            
        X_train_tensor = torch.FloatTensor(X_train_np)
        y_train_tensor = torch.LongTensor(y_train_np)
        
        # Create training dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=kwargs.get('batch_size', BATCH_SIZE),
            shuffle=True
        )
        
        # Create validation dataset and dataloader if validation data provided
        val_loader = None
        if X_val is not None and y_val is not None:
            # Convert pandas Series to numpy arrays if needed
            if isinstance(X_val, pd.Series) or isinstance(X_val, pd.DataFrame):
                X_val_np = X_val.values
            else:
                X_val_np = X_val
                
            if isinstance(y_val, pd.Series):
                y_val_np = y_val.values
            else:
                y_val_np = y_val
                
            X_val_tensor = torch.FloatTensor(X_val_np)
            y_val_tensor = torch.LongTensor(y_val_np)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=kwargs.get('batch_size', BATCH_SIZE),
                shuffle=False
            )
        
        # Training parameters
        n_epochs = kwargs.get('epochs', 10)
        patience = kwargs.get('patience', 5)  # Early stopping patience
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        patience_count = 0
        
        # Training loop
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            # Calculate training metrics
            train_loss = train_loss / train_total
            train_accuracy = train_correct / train_total
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_preds = []
                val_targets_list = []
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        
                        # Track statistics
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += targets.size(0)
                        val_correct += (predicted == targets).sum().item()
                        
                        # Store predictions for F1 score calculation
                        val_preds.extend(predicted.cpu().numpy())
                        val_targets_list.extend(targets.cpu().numpy())
                
                # Calculate validation metrics
                val_loss = val_loss / val_total
                val_accuracy = val_correct / val_total
                val_f1 = f1_score(val_targets_list, val_preds, average='weighted')
                
                # Update training history
                epoch_metrics = {
                    'loss': train_loss,
                    'accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_f1_score': val_f1
                }
                
                # Print progress
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                      f"Val F1: {val_f1:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                    # Save best model (optional)
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # No validation data, just update with training metrics
                epoch_metrics = {
                    'loss': train_loss,
                    'accuracy': train_accuracy
                }
                
                # Print progress
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            
            # Update training history
            self.update_training_history(epoch_metrics)
        
        self.is_fitted = True
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions using the CNN model.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert data to tensor
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the CNN model.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert data to tensor
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def update_with_knowledge(self, X_data, y_data, global_soft_preds, alpha=0.3):
        """
        Legacy method for backward compatibility.
        
        Args:
            X_data: Feature data for knowledge transfer
            y_data: Target data for evaluation
            global_soft_preds: Soft predictions from the global model
            alpha: Weight for global knowledge (0.0-1.0)
            
        Returns:
            Updated model
        """
        return self.update_with_global_knowledge(
            global_preds_proba=global_soft_preds,
            X_val=X_data,
            y_val=y_data,
            alpha=alpha
        )
        
    def update_with_global_knowledge(self, global_preds_proba, X_val, y_val, 
                                   global_feature_importance=None, alpha=0.2, **kwargs):
        """
        Update the CNN model with knowledge from the global model using an adaptive
        approach that ensures performance improvement across rounds.
        
        Args:
            global_preds_proba: Soft predictions from the global model
            X_val: Validation features
            y_val: Validation labels
            global_feature_importance: Feature importance from global model (not used for CNN)
            alpha: Weight for global knowledge (0.0-1.0)
            **kwargs: Additional parameters including:
                - round_number: Current communication round
                - current_performance: Model's current performance metrics
                - global_performance: Global model performance metrics
            
        Returns:
            Updated model
        """
        if not self.is_fitted:
            print("CNN model not fitted. Cannot update with knowledge.")
            return self.model
        
        # Get additional parameters
        round_number = kwargs.get('round_number', 1)
        current_performance = kwargs.get('current_performance', {})
        global_performance = kwargs.get('global_performance', {})
        
        print(f"Updating CNN model for client {self.client_id} with knowledge distillation (round={round_number}, alpha={alpha:.3f})...")
        
        # Check model performance before update
        current_preds = self.predict(X_val)
        current_accuracy = accuracy_score(y_val, current_preds)
        current_f1 = f1_score(y_val, current_preds, average='weighted')
        print(f"CNN model before update - Accuracy: {current_accuracy:.4f}, F1 score: {current_f1:.4f}")
        
        # Handle shape mismatch between client data and global predictions
        if len(y_val) != len(global_preds_proba):
            print(f"Shape mismatch detected: Client data has {len(y_val)} samples, but global predictions have {len(global_preds_proba)} samples.")
            print("Using a subset of client data that matches the size of global predictions.")
            
            # Use only the first n samples where n is the minimum size
            n_samples = min(len(y_val), len(global_preds_proba))
            
            # Convert data if needed
            if isinstance(X_val, pd.DataFrame) or isinstance(X_val, pd.Series):
                X_val = X_val.iloc[:n_samples]
            else:
                X_val = X_val[:n_samples]
            
            if isinstance(y_val, pd.Series):
                y_val = y_val.iloc[:n_samples]
            else:
                y_val = y_val[:n_samples]
            
            global_preds_proba = global_preds_proba[:n_samples]
            
            print(f"Using {n_samples} samples for knowledge transfer.")
            
        # Store original model state for possible rollback
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val).to(self.device)
        y_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create TensorDataset and DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = min(32, len(X_tensor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create global soft targets tensor
        global_preds_tensor = torch.FloatTensor(global_preds_proba).to(self.device)
        
        # Set model to training mode
        self.model.train()
        
        # Adaptive fine-tuning parameters based on round
        # As rounds increase, we fine-tune more aggressively
        temperature = max(2.0, 4.0 - (round_number * 0.3))  # Start higher and decrease with rounds
        learning_rate = max(0.00005, 0.0001 / round_number)  # Decrease with rounds
        weight_decay = min(0.001, 0.0001 * round_number)  # Increase with rounds
        n_epochs = min(3, 1 + round_number // 2)  # Increase epochs with rounds
        
        print(f"Using adaptive parameters: temp={temperature:.2f}, lr={learning_rate:.6f}, " 
              f"weight_decay={weight_decay:.6f}, epochs={n_epochs}")
        
        # Use Adam optimizer with weight decay for regularization
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop with knowledge distillation
        epoch_losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            # Process each batch
            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                # Get global predictions for this batch
                batch_indices = batch_idx * batch_size
                batch_end_idx = min((batch_idx + 1) * batch_size, len(global_preds_tensor))
                batch_global_preds = global_preds_tensor[batch_indices:batch_end_idx]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                
                # Hard target loss (cross entropy with true labels)
                hard_loss = self.criterion(outputs, batch_y)
                
                # Soft target loss (KL divergence with global model predictions)
                log_outputs = nn.functional.log_softmax(outputs / temperature, dim=1)
                soft_targets = nn.functional.softmax(batch_global_preds / temperature, dim=1)
                soft_loss = nn.functional.kl_div(log_outputs, soft_targets, reduction='batchmean') * (temperature ** 2)
                
                # Combined loss with alpha weighting
                loss = (1 - alpha) * hard_loss + alpha * soft_loss
                
                # Backward pass and optimizer step
                loss.backward()
                
                # Gradient clipping to prevent drastic updates
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
            
            # Check if loss is diverging
            if epoch > 0 and avg_loss > epoch_losses[0] * 1.5:
                print("Warning: Loss is diverging. Stopping early.")
                break
        
        # Set model back to eval mode
        self.model.eval()
        
        # Evaluate updated model
        updated_preds = self.predict(X_val)
        updated_accuracy = accuracy_score(y_val, updated_preds)
        updated_f1 = f1_score(y_val, updated_preds, average='weighted')
        
        # Performance threshold varies based on round number
        # Higher rounds have higher expectations for improvement
        min_accepted_ratio = 0.98 if round_number <= 2 else (0.99 if round_number <= 4 else 0.995)
        
        # Check if performance degraded significantly
        if updated_accuracy < current_accuracy * min_accepted_ratio:  
            print(f"Warning: Performance degraded significantly after update. "
                 f"Reverting to original model state.")
            
            # If the performance degraded, restore from original state
            self.model.load_state_dict(original_state)
            print(f"Model reverted - Accuracy: {current_accuracy:.4f}, F1 score: {current_f1:.4f}")
            
            # Try again with a much smaller alpha if this is not the first round
            if round_number > 1 and alpha > 0.05:
                tiny_alpha = alpha * 0.25  # Very small alpha
                print(f"Attempting transfer with tiny alpha={tiny_alpha:.4f}")
                
                # Recursively call with reduced alpha
                return self.update_with_global_knowledge(
                    global_preds_proba=global_preds_proba,
                    X_val=X_val,
                    y_val=y_val,
                    alpha=tiny_alpha,
                    **kwargs
                )
        else:
            # If model improved or maintained performance
            improvement = updated_accuracy - current_accuracy
            print(f"Updated CNN model - Accuracy: {updated_accuracy:.4f} (change: {improvement:.4f}), "
                 f"F1 score: {updated_f1:.4f} (change: {updated_f1-current_f1:.4f})")
            
            # If very little improvement, apply an additional fine-tuning step on hard labels
            if 0 <= improvement < 0.002 and round_number > 1:
                print("Minimal improvement detected. Applying additional fine-tuning on hard labels.")
                
                # Fine-tune with just hard labels
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate*0.5)
                
                # Short fine-tuning loop
                self.model.train()
                for _ in range(1):
                    for batch_x, batch_y in dataloader:
                        optimizer.zero_grad()
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                self.model.eval()
                
                # Check final performance
                final_preds = self.predict(X_val)
                final_accuracy = accuracy_score(y_val, final_preds)
                final_f1 = f1_score(y_val, final_preds, average='weighted')
                
                print(f"After additional fine-tuning - Accuracy: {final_accuracy:.4f}, F1 score: {final_f1:.4f}")
        
        return self.model 