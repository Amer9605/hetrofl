"""
Knowledge distillation for federated learning.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from config.config import (
    GLOBAL_MODEL_HIDDEN_LAYERS,
    KL_TEMPERATURE,
    DISTILLATION_ALPHA,
    LEARNING_RATE,
    BATCH_SIZE,
    RANDOM_STATE
)


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron Classifier model.
    """
    
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate=0.2):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout_rate: Dropout rate
        """
        super(MLPClassifier, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output predictions
        """
        return self.model(x)


class KnowledgeDistillation:
    """
    Knowledge distillation for aggregating knowledge from local models.
    """
    
    def __init__(self, input_dim, output_dim, temperature=2.0, device=None):
        """
        Initialize the knowledge distillation model.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            temperature: Temperature for knowledge distillation
            device: Device to use for PyTorch
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_models = []
        self.is_fitted = False
        self.model = None
        
        # Set up training parameters
        self.epochs = 20
        self.patience = 5
        self.batch_size = 32
        
        # Initialize history dictionary
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': []
        }
        
        # Build the model
        if input_dim is not None and output_dim is not None:
            self.build_model()
        
        # Parameters
        self.hidden_layers = GLOBAL_MODEL_HIDDEN_LAYERS
        self.learning_rate = LEARNING_RATE
        self.criterion_hard = None
        self.criterion_soft = None
    
    def build_model(self):
        """
        Build the global model.
        
        Returns:
            PyTorch model
        """
        # Initialize is_fitted to False
        self.is_fitted = False
        
        # Create a simple neural network
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.output_dim)
        ).to(self.device)
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()
        
        return self.model
    
    def add_local_model(self, local_model):
        """
        Add a local model to the list of local models.
        
        Args:
            local_model: Local model object
        """
        self.local_models.append(local_model)
    
    def _get_ensemble_predictions(self, X):
        """
        Get predictions from all local models.
        
        Args:
            X: Features
            
        Returns:
            Ensemble of predictions (soft targets)
        """
        if not self.local_models:
            print("Warning: No local models available for ensemble predictions")
            # Return uniform distribution as fallback
            num_samples = X.shape[0]
            num_classes = self.output_dim
            return np.ones((num_samples, num_classes)) / num_classes
        
        # Get predictions from all local models
        all_probs = []
        for model in self.local_models:
            try:
                probs = model.predict_proba(X)
                
                # If binary classification with single probability, convert to 2-class format
                if len(probs.shape) == 1 or probs.shape[1] == 1:
                    probs_2class = np.zeros((probs.shape[0], 2))
                    probs_2class[:, 1] = probs
                    probs_2class[:, 0] = 1 - probs
                    probs = probs_2class
                
                all_probs.append(probs)
            except Exception as e:
                print(f"Error getting predictions from model {model.model_name}: {e}")
                continue
        
        if not all_probs:
            print("Warning: Failed to get predictions from any local model")
            # Return uniform distribution as fallback
            num_samples = X.shape[0]
            num_classes = self.output_dim
            return np.ones((num_samples, num_classes)) / num_classes
        
        # Average predictions
        ensemble_probs = np.mean(all_probs, axis=0)
        
        return ensemble_probs
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=20, patience=5):
        """
        Train the global model using knowledge distillation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Trained model
        """
        if self.model is None:
            self.build_model()
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train.values if isinstance(y_train, pd.Series) else y_train).to(self.device)
        
        # Get soft targets from local models
        soft_targets = self._get_ensemble_predictions(X_train)
        soft_targets_tensor = torch.FloatTensor(soft_targets).to(self.device)
        
        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val.values if isinstance(y_val, pd.Series) else y_val).to(self.device)
            
            # Get soft targets for validation data
            soft_targets_val = self._get_ensemble_predictions(X_val)
            soft_targets_val_tensor = torch.FloatTensor(soft_targets_val).to(self.device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor, soft_targets_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor, soft_targets_val_tensor)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Progressive distillation: gradually increase the importance of soft targets
        # Start with more focus on hard targets and gradually shift to soft targets
        alpha_schedule = np.linspace(0.1, DISTILLATION_ALPHA, epochs)
        
        for epoch in range(epochs):
            # Set current alpha
            current_alpha = alpha_schedule[epoch]
            
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch, soft_targets_batch in train_loader:
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Hard targets
                hard_targets = y_batch
                
                # Hard target loss (cross-entropy)
                hard_loss = self.criterion(outputs, hard_targets)
                
                # Soft target loss (KL divergence)
                if soft_targets_tensor is not None:
                    log_softmax_outputs = nn.functional.log_softmax(outputs / self.temperature, dim=1)
                    soft_targets_scaled = nn.functional.softmax(soft_targets_batch / self.temperature, dim=1)
                    soft_loss = nn.functional.kl_div(log_softmax_outputs, soft_targets_scaled, reduction='batchmean') * (self.temperature ** 2)
                    
                    # Combined loss with progressive alpha
                    loss = (1 - current_alpha) * hard_loss + current_alpha * soft_loss
                else:
                    loss = hard_loss
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Accumulate loss
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0.0
            if X_val is not None and y_val is not None:
                self.model.eval()
                
                with torch.no_grad():
                    for X_batch, y_batch, soft_targets_batch in val_loader:
                        # Forward pass
                        outputs = self.model(X_batch)
                        
                        # Hard targets
                        hard_targets = y_batch
                        
                        # Hard target loss
                        hard_loss = self.criterion(outputs, hard_targets)
                        
                        # Soft target loss
                        if 'soft_targets_val_tensor' in locals():
                            log_softmax_outputs = nn.functional.log_softmax(outputs / self.temperature, dim=1)
                            soft_targets_scaled = nn.functional.softmax(soft_targets_batch / self.temperature, dim=1)
                            soft_loss = nn.functional.kl_div(log_softmax_outputs, soft_targets_scaled, reduction='batchmean') * (self.temperature ** 2)
                            
                            # Combined loss with progressive alpha
                            loss = (1 - current_alpha) * hard_loss + current_alpha * soft_loss
                        else:
                            loss = hard_loss
                        
                        # Accumulate loss
                        val_loss += loss.item()
                    
                    # Calculate average validation loss
                    val_loss /= len(val_loader)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Alpha: {current_alpha:.2f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            if X_val is not None and y_val is not None:
                self.history['val_loss'].append(val_loss)
        
        # Load best model if available
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Set is_fitted flag
        self.is_fitted = True
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions using the global model.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to tensor
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
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        # Check if the model is fitted
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            # If not fitted, use ensemble predictions from local models
            print("Global model not fitted yet. Using ensemble predictions from local models...")
            return self._get_ensemble_predictions(X)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Convert to numpy array
        probabilities = probabilities.cpu().numpy()
        
        return probabilities
    
    def save_model(self, save_dir=None, filename="global_model.pt"):
        """
        Save the global model.
        
        Args:
            save_dir: Directory to save the model (default: MODEL_SAVE_DIR/global)
            filename: Name of the model file
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        if save_dir is None:
            save_dir = os.path.join(MODEL_SAVE_DIR, "global")
        
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, filename)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers,
            'history': self.history,
            'temperature': self.temperature,
            'alpha': DISTILLATION_ALPHA
        }, model_path)
        
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """
        Load the global model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Set parameters
        self.input_dim = checkpoint['input_dim']
        self.output_dim = checkpoint['output_dim']
        self.hidden_layers = checkpoint['hidden_layers']
        self.history = checkpoint['history']
        self.temperature = checkpoint.get('temperature', self.temperature)
        self.alpha = DISTILLATION_ALPHA
        
        # Build model
        self.build_model()
        
        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.is_fitted = True
        
        print(f"Model loaded from {model_path}")
        
        return self.model
    
    def update_local_models(self, local_models, X_data, y_data):
        """
        Send updates to local models based on global model knowledge.
        
        Args:
            local_models: Dictionary of local model objects
            X_data: Feature data to use for knowledge transfer
            y_data: Target data for evaluation
            
        Returns:
            Dictionary of updated local models
        """
        if not BIDIRECTIONAL_LEARNING:
            print("Bidirectional learning is disabled. Skipping local model updates.")
            return local_models
        
        print("\nUpdating local models with global knowledge...")
        
        # Ensure model is trained
        if not self.is_fitted:
            print("Global model not fitted. Cannot update local models.")
            return local_models
        
        # Get global model predictions (soft targets)
        X_tensor = torch.FloatTensor(X_data.values if isinstance(X_data, pd.DataFrame) else X_data)
        X_tensor = X_tensor.to(self.device)
        
        # Get global model soft predictions
        self.model.eval()
        with torch.no_grad():
            global_logits = self.model(X_tensor)
            global_soft_preds = torch.softmax(global_logits / self.temperature, dim=1).cpu().numpy()
        
        # Update each local model
        updated_models = {}
        for client_id, model in local_models.items():
            if hasattr(model, 'update_with_knowledge') and callable(getattr(model, 'update_with_knowledge')):
                try:
                    # Apply global knowledge transfer
                    print(f"Updating local model for client {client_id}...")
                    model.update_with_knowledge(X_data, y_data, global_soft_preds, GLOBAL_TO_LOCAL_ALPHA)
                    updated_models[client_id] = model
                except Exception as e:
                    print(f"Error updating local model for client {client_id}: {e}")
                    updated_models[client_id] = model
            else:
                print(f"Local model for client {client_id} does not support knowledge updates.")
                updated_models[client_id] = model
        
        print("Local models updated with global knowledge.")
        return updated_models 