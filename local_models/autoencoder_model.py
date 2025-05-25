"""
Autoencoder model implementation for HETROFL.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from local_models.base_model import BaseLocalModel
from config.config import RANDOM_STATE, BATCH_SIZE, LEARNING_RATE


class Autoencoder(nn.Module):
    """
    Autoencoder for tabular data with classification head.
    """
    
    def __init__(self, input_dim, encoder_dims, latent_dim, decoder_dims=None, output_dim=None, dropout_rate=0.2):
        """
        Initialize the Autoencoder model.
        
        Args:
            input_dim: Input feature dimension
            encoder_dims: List of encoder layer dimensions
            latent_dim: Dimension of the latent space
            decoder_dims: List of decoder layer dimensions (if None, mirror of encoder)
            output_dim: Output dimension for classification (optional)
            dropout_rate: Dropout rate
        """
        super(Autoencoder, self).__init__()
        
        # Mirror encoder dimensions for decoder if not provided
        if decoder_dims is None:
            decoder_dims = encoder_dims[::-1]
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Final encoder layer to latent space
        encoder_layers.extend([
            nn.Linear(prev_dim, latent_dim),
            nn.BatchNorm1d(latent_dim)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        
        for dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Final decoder layer to output
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Classification head (if output_dim provided)
        self.classifier = None
        if output_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, output_dim)
            )
    
    def encode(self, x):
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representation to reconstructed input.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed input, class predictions)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        
        if self.classifier is not None:
            class_pred = self.classifier(z)
            return x_recon, class_pred
        else:
            return x_recon, None


class AutoencoderModel(BaseLocalModel):
    """
    Autoencoder model for federated learning.
    """
    
    def __init__(self, client_id, input_dim=None, output_dim=None):
        """
        Initialize the Autoencoder model.
        
        Args:
            client_id: Client ID
            input_dim: Input feature dimension
            output_dim: Number of output classes
        """
        super().__init__("autoencoder", client_id, output_dim)
        self.input_dim = input_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reconstruction_criterion = None
        self.classification_criterion = None
        self.optimizer = None
        self.latent_dim = None
        
        # Additional classifier for extracted features
        self.classifier = None
    
    def build_model(self, **kwargs):
        """
        Build the Autoencoder model.
        
        Args:
            **kwargs: Additional model parameters
            
        Returns:
            Autoencoder model
        """
        if self.input_dim is None:
            raise ValueError("input_dim must be specified for Autoencoder model")
        
        # Default parameters
        params = {
            'encoder_dims': [256, 128],
            'latent_dim': 64,
            'decoder_dims': None,
            'dropout_rate': 0.2,
            'learning_rate': LEARNING_RATE,
            'classification_weight': 0.5  # Weight for classification loss
        }
        
        # Update parameters with best params if available
        if self.best_params:
            params.update(self.best_params)
        
        # Update parameters with any provided kwargs
        params.update(kwargs)
        
        # Store latent dimension
        self.latent_dim = params['latent_dim']
        
        # Create the Autoencoder model
        self.model = Autoencoder(
            input_dim=self.input_dim,
            encoder_dims=params['encoder_dims'],
            latent_dim=params['latent_dim'],
            decoder_dims=params['decoder_dims'],
            output_dim=self.output_dim,
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
        # Loss functions
        self.reconstruction_criterion = nn.MSELoss()
        if self.output_dim is not None:
            self.classification_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=params['learning_rate']
        )
        
        # Classification weight
        self.classification_weight = params['classification_weight']
        
        return self.model
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Tune the Autoencoder hyperparameters using Optuna.
        
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
            encoder_dims = []
            
            for i in range(n_layers):
                encoder_dims.append(trial.suggest_categorical(
                    f'encoder_dim_{i}',
                    [64, 128, 256, 512]
                ))
            
            return {
                'encoder_dims': encoder_dims,
                'latent_dim': trial.suggest_categorical('latent_dim', [32, 64, 128]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'classification_weight': trial.suggest_float('classification_weight', 0.1, 0.9)
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
                    reconstructed, class_pred = model(inputs)
                    
                    # Reconstruction loss
                    recon_loss = self.reconstruction_criterion(reconstructed, inputs)
                    
                    # Classification loss (if applicable)
                    if class_pred is not None:
                        class_loss = self.classification_criterion(class_pred, targets)
                        total_loss = (1 - self.classification_weight) * recon_loss + \
                                     self.classification_weight * class_loss
                    else:
                        total_loss = recon_loss
                    
                    total_loss.backward()
                    self.optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            val_reconstructions = []
            val_features = []
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Get latent representations and reconstructions
                    latent_repr = model.encode(inputs)
                    reconstructed = model.decode(latent_repr)
                    
                    # If model has classifier, use it
                    if model.classifier is not None:
                        class_pred = model.classifier(latent_repr)
                        _, predicted = torch.max(class_pred.data, 1)
                        val_preds.extend(predicted.cpu().numpy())
                    else:
                        # Store latent features for external classifier
                        val_features.append(latent_repr.cpu().numpy())
                    
                    val_reconstructions.append(reconstructed.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            # If we used external classifier, train it on latent features
            if model.classifier is None and val_features:
                val_features = np.vstack(val_features)
                clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                clf.fit(val_features, val_targets)
                val_preds = clf.predict(val_features)
            
            # Calculate evaluation metrics
            reconstruction_error = np.mean(
                np.square(np.vstack(val_reconstructions) - X_val_tensor.numpy())
            )
            
            if val_preds:
                accuracy = accuracy_score(val_targets, val_preds)
                f1 = f1_score(val_targets, val_preds, average='weighted')
                
                # Combined score: reconstruction quality and classification performance
                score = 0.5 * (1 - reconstruction_error) + 0.5 * (accuracy + f1) / 2
            else:
                score = 1 - reconstruction_error
            
            return score
        
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
        Train the Autoencoder model.
        
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
        
        if self.output_dim is None and y_train is not None:
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
            
        X_train_tensor = torch.FloatTensor(X_train_np)
        
        if y_train is not None:
            if isinstance(y_train, pd.Series):
                y_train_np = y_train.values
            else:
                y_train_np = y_train
            y_train_tensor = torch.LongTensor(y_train_np)
        else:
            y_train_tensor = None
        
        # Create training dataset and dataloader
        if y_train_tensor is not None:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        else:
            train_dataset = TensorDataset(X_train_tensor, X_train_tensor)  # Dummy targets
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=kwargs.get('batch_size', BATCH_SIZE),
            shuffle=True
        )
        
        # Create validation dataset and dataloader if validation data provided
        val_loader = None
        if X_val is not None:
            # Convert pandas Series to numpy arrays if needed
            if isinstance(X_val, pd.Series) or isinstance(X_val, pd.DataFrame):
                X_val_np = X_val.values
            else:
                X_val_np = X_val
                
            X_val_tensor = torch.FloatTensor(X_val_np)
            
            if y_val is not None:
                if isinstance(y_val, pd.Series):
                    y_val_np = y_val.values
                else:
                    y_val_np = y_val
                y_val_tensor = torch.LongTensor(y_val_np)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            else:
                val_dataset = TensorDataset(X_val_tensor, X_val_tensor)  # Dummy targets
                
            val_loader = DataLoader(
                val_dataset,
                batch_size=kwargs.get('batch_size', BATCH_SIZE),
                shuffle=False
            )
        
        # Training parameters
        n_epochs = kwargs.get('epochs', 20)
        patience = kwargs.get('patience', 5)  # Early stopping patience
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        patience_count = 0
        
        # Training loop
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_recon_loss = 0.0
            train_class_loss = 0.0
            train_total_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
                inputs = inputs.to(self.device)
                if y_train_tensor is not None:
                    targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstructed, class_pred = self.model(inputs)
                
                # Reconstruction loss
                recon_loss = self.reconstruction_criterion(reconstructed, inputs)
                
                # Classification loss if applicable
                if class_pred is not None and y_train_tensor is not None:
                    class_loss = self.classification_criterion(class_pred, targets)
                    total_loss = (1 - self.classification_weight) * recon_loss + \
                                 self.classification_weight * class_loss
                    
                    # Track accuracy
                    _, predicted = torch.max(class_pred.data, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
                    train_class_loss += class_loss.item() * inputs.size(0)
                else:
                    total_loss = recon_loss
                
                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_recon_loss += recon_loss.item() * inputs.size(0)
                train_total_loss += total_loss.item() * inputs.size(0)
            
            # Calculate training metrics
            train_recon_loss = train_recon_loss / len(train_loader.dataset)
            train_total_loss = train_total_loss / len(train_loader.dataset)
            
            if y_train_tensor is not None and class_pred is not None:
                train_class_loss = train_class_loss / len(train_loader.dataset)
                train_accuracy = train_correct / train_total
                
                # Add to training history
                epoch_metrics = {
                    'loss': train_total_loss,
                    'recon_loss': train_recon_loss,
                    'class_loss': train_class_loss,
                    'accuracy': train_accuracy
                }
                
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Loss: {train_total_loss:.4f}, Recon Loss: {train_recon_loss:.4f}, "
                      f"Class Loss: {train_class_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            else:
                # Add to training history
                epoch_metrics = {
                    'loss': train_total_loss,
                    'recon_loss': train_recon_loss
                }
                
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Loss: {train_total_loss:.4f}, Recon Loss: {train_recon_loss:.4f}")
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_recon_loss = 0.0
                val_class_loss = 0.0
                val_total_loss = 0.0
                val_correct = 0
                val_total = 0
                val_preds = []
                val_targets_list = []
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        if y_val is not None:
                            targets = targets.to(self.device)
                        
                        # Forward pass
                        reconstructed, class_pred = self.model(inputs)
                        
                        # Reconstruction loss
                        recon_loss = self.reconstruction_criterion(reconstructed, inputs)
                        
                        # Classification loss if applicable
                        if class_pred is not None and y_val is not None:
                            class_loss = self.classification_criterion(class_pred, targets)
                            total_loss = (1 - self.classification_weight) * recon_loss + \
                                         self.classification_weight * class_loss
                            
                            # Track accuracy
                            _, predicted = torch.max(class_pred.data, 1)
                            val_total += targets.size(0)
                            val_correct += (predicted == targets).sum().item()
                            val_class_loss += class_loss.item() * inputs.size(0)
                            
                            # Store predictions for F1 score calculation
                            val_preds.extend(predicted.cpu().numpy())
                            val_targets_list.extend(targets.cpu().numpy())
                        else:
                            total_loss = recon_loss
                        
                        # Track statistics
                        val_recon_loss += recon_loss.item() * inputs.size(0)
                        val_total_loss += total_loss.item() * inputs.size(0)
                
                # Calculate validation metrics
                val_recon_loss = val_recon_loss / len(val_loader.dataset)
                val_total_loss = val_total_loss / len(val_loader.dataset)
                
                if y_val is not None and class_pred is not None:
                    val_class_loss = val_class_loss / len(val_loader.dataset)
                    val_accuracy = val_correct / val_total
                    val_f1 = f1_score(val_targets_list, val_preds, average='weighted')
                    
                    # Update epoch metrics
                    epoch_metrics.update({
                        'val_loss': val_total_loss,
                        'val_recon_loss': val_recon_loss,
                        'val_class_loss': val_class_loss,
                        'val_accuracy': val_accuracy,
                        'val_f1_score': val_f1
                    })
                    
                    print(f"Validation: Loss: {val_total_loss:.4f}, Recon Loss: {val_recon_loss:.4f}, "
                          f"Class Loss: {val_class_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
                else:
                    # Update epoch metrics
                    epoch_metrics.update({
                        'val_loss': val_total_loss,
                        'val_recon_loss': val_recon_loss
                    })
                    
                    print(f"Validation: Loss: {val_total_loss:.4f}, Recon Loss: {val_recon_loss:.4f}")
                
                # Early stopping
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    patience_count = 0
                    # Save best model (optional)
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Update training history
            self.update_training_history(epoch_metrics)
        
        # Train classifier if model doesn't have one but we need classification
        if self.model.classifier is None and y_train is not None:
            self._train_external_classifier(X_train, y_train)
        
        self.is_fitted = True
        
        return self.model
    
    def _train_external_classifier(self, X, y):
        """
        Train an external classifier on the encoded features.
        
        Args:
            X: Features
            y: Labels
        """
        print("Training external classifier on encoded features...")
        
        # Get encoded features
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            features = self.model.encode(X_tensor).cpu().numpy()
        
        # Train a logistic regression classifier
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        self.classifier.fit(features, y)
        
        # Evaluate on training data
        y_pred = self.classifier.predict(features)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        
        print(f"External classifier - Training accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
    
    def predict(self, X):
        """
        Make predictions using the Autoencoder model.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get encoded features
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Get latent representation
            latent_repr = self.model.encode(X_tensor)
            
            # If model has classifier, use it
            if self.model.classifier is not None:
                outputs = self.model.classifier(latent_repr)
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
            # Otherwise use external classifier
            elif self.classifier is not None:
                features = latent_repr.cpu().numpy()
                return self.classifier.predict(features)
            else:
                raise ValueError("No classifier available for prediction")
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Autoencoder model.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get encoded features
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Get latent representation
            latent_repr = self.model.encode(X_tensor)
            
            # If model has classifier, use it
            if self.model.classifier is not None:
                outputs = self.model.classifier(latent_repr)
                probabilities = nn.functional.softmax(outputs, dim=1)
                return probabilities.cpu().numpy()
            # Otherwise use external classifier
            elif self.classifier is not None:
                features = latent_repr.cpu().numpy()
                return self.classifier.predict_proba(features)
            else:
                raise ValueError("No classifier available for prediction")
    
    def encode(self, X):
        """
        Encode data to latent representation.
        
        Args:
            X: Features
            
        Returns:
            Latent representation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Encode data
        with torch.no_grad():
            latent_repr = self.model.encode(X_tensor)
        
        return latent_repr.cpu().numpy()
    
    def reconstruct(self, X):
        """
        Reconstruct input data.
        
        Args:
            X: Features
            
        Returns:
            Reconstructed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before reconstruction")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Reconstruct data
        with torch.no_grad():
            latent_repr = self.model.encode(X_tensor)
            reconstructed = self.model.decode(latent_repr)
        
        return reconstructed.cpu().numpy()
    
    def update_with_knowledge(self, X_data, y_data, global_soft_preds, alpha=0.3):
        """
        Update the Autoencoder model with knowledge from the global model.
        
        Args:
            X_data: Feature data for knowledge transfer
            y_data: Target data for evaluation
            global_soft_preds: Soft predictions from the global model
            alpha: Weight for global knowledge (0.0-1.0)
            
        Returns:
            Updated model
        """
        if not self.is_fitted:
            print("Autoencoder model not fitted. Cannot update with knowledge.")
            return self.model
        
        print(f"Updating Autoencoder model for client {self.client_id} with global knowledge...")
        
        # Handle shape mismatch between client data and global predictions
        if len(y_data) != len(global_soft_preds):
            print(f"Shape mismatch detected: Client data has {len(y_data)} samples, but global predictions have {len(global_soft_preds)} samples.")
            print("Using a subset of client data that matches the size of global predictions.")
            
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
        
        # For neural network models like Autoencoder, we can fine-tune the classifier part
        # using a blend of hard labels and soft predictions from the global model
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_data.values if isinstance(X_data, pd.DataFrame) else X_data).to(self.device)
        
        # Convert labels to one-hot encoding
        num_classes = self.output_dim
        y_one_hot = np.zeros((len(y_data), num_classes))
        for i, label in enumerate(y_data):
            y_one_hot[i, int(label)] = 1
        
        # Create blended targets
        blended_targets = (1 - alpha) * y_one_hot + alpha * global_soft_preds
        blended_targets_tensor = torch.FloatTensor(blended_targets).to(self.device)
        
        # If the model has a classifier, fine-tune it
        if self.model.classifier is not None:
            print("Fine-tuning classifier with knowledge distillation...")
            
            # Set up optimizer for fine-tuning just the classifier
            optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=0.001)
            
            # Set model to training mode
            self.model.eval()  # Keep encoder/decoder in eval mode
            self.model.classifier.train()  # Only train classifier
            
            # Fine-tune for a few epochs
            n_epochs = 5
            batch_size = 32
            
            # Create dataset and dataloader
            dataset = torch.utils.data.TensorDataset(X_tensor, blended_targets_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(n_epochs):
                total_loss = 0.0
                
                for batch_X, batch_targets in dataloader:
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass - only through the encoder and classifier
                    with torch.no_grad():
                        latent_repr = self.model.encode(batch_X)
                    
                    outputs = self.model.classifier(latent_repr)
                    
                    # KL divergence loss between outputs and blended targets
                    log_softmax_outputs = nn.functional.log_softmax(outputs, dim=1)
                    loss = nn.functional.kl_div(log_softmax_outputs, batch_targets, reduction='batchmean')
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
            
            # Set model back to eval mode
            self.model.eval()
            
        # If using external classifier, retrain it with blended targets
        elif self.classifier is not None:
            print("Retraining external classifier with knowledge distillation...")
            
            # Get encoded features
            self.model.eval()
            with torch.no_grad():
                features = self.model.encode(X_tensor).cpu().numpy()
            
            # Get hard labels from blended targets
            y_blend = np.argmax(blended_targets, axis=1)
            
            # Retrain the classifier
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            )
            self.classifier.fit(features, y_blend)
            
            # Evaluate
            y_pred = self.classifier.predict(features)
            accuracy = accuracy_score(y_data, y_pred)
            print(f"Updated external classifier accuracy: {accuracy:.4f}")
        
        else:
            print("No classifier component found. Cannot update with knowledge.")
        
        return self.model 