"""
Data loading and preprocessing module for the HETROFL system.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

from config.config import (
    DATASET_PATH, 
    TARGET_COLUMN, 
    TEST_SIZE, 
    VAL_SIZE, 
    RANDOM_STATE,
    SMOTE_SAMPLING_STRATEGY,
    SMOTE_K_NEIGHBORS,
    RESULTS_DIR,
    DATASET_SAMPLE_SIZE
)

class DataLoader:
    """
    Handles loading, preprocessing, and partitioning of the dataset for federated learning.
    """
    
    def __init__(self, sample_size=None):
        """
        Initialize the DataLoader.
        
        Args:
            sample_size: Number of samples to use from the dataset (None for all data)
        """
        self.sample_size = sample_size
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        self.label_encoder = None
        self.scaler = None
        self.class_names = None
        self.class_distribution = None
        
    def load_data(self):
        """
        Load the dataset from the specified path.
        """
        print(f"Loading dataset from {DATASET_PATH}...")
        # Check if file exists
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
            
        # Load the full dataset - handle both Parquet and CSV files
        if DATASET_PATH.endswith('.parquet'):
            full_data = pd.read_parquet(DATASET_PATH)
        elif DATASET_PATH.endswith('.csv'):
            full_data = pd.read_csv(DATASET_PATH)
        else:
            raise ValueError(f"Unsupported file format: {DATASET_PATH}")
        
        # Print columns to check if target column exists
        print(f"Available columns: {full_data.columns.tolist()}")
        if TARGET_COLUMN not in full_data.columns:
            print(f"WARNING: Target column '{TARGET_COLUMN}' not found in dataset!")
            print("Available columns that might be the target:")
            for col in full_data.columns:
                if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower() or 'attack' in col.lower():
                    print(f"  - {col}")
        
        # Use the full dataset or sample it based on configuration
        if self.sample_size is None:
            self.sample_size = DATASET_SAMPLE_SIZE
        
        # If sample_size is -1 or exceeds the dataset size, use the full dataset
        if self.sample_size == -1 or self.sample_size >= len(full_data):
            self.data = full_data
            print(f"Using the full dataset with {len(full_data)} rows")
        else:
            self.data = full_data.sample(n=self.sample_size, random_state=RANDOM_STATE)
            print(f"Using a sample of {self.sample_size} rows for processing")
        
        print(f"Dataset loaded with shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """
        Perform basic data exploration and return summary statistics.
        """
        if self.data is None:
            self.load_data()
            
        # Get basic info
        summary = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "class_distribution": self.data[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in self.data.columns else None
        }
        
        print("Data Summary:")
        print(f"Shape: {summary['shape']}")
        print(f"Missing values: {sum(summary['missing_values'].values())}")
        
        if TARGET_COLUMN in self.data.columns:
            print(f"Class distribution: {summary['class_distribution']}")
            self.class_distribution = summary['class_distribution']
        
        return summary
    
    def preprocess_data(self):
        """
        Preprocess the dataset: handle missing values, encode categorical variables, and scale features.
        """
        if self.data is None:
            self.load_data()
            
        print("Preprocessing data...")
        
        # Make a copy to avoid modifying the original data
        df = self.data.copy()
        
        # Handle missing values
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                # For categorical columns, we need to add missing to categories first
                categories = df[col].cat.categories.tolist()
                if 'missing' not in categories:
                    df[col] = df[col].cat.add_categories(['missing'])
                df[col] = df[col].fillna('missing')
            elif df[col].dtype == 'object':
                # For object types, simply fill with 'missing'
                df[col] = df[col].fillna('missing')
            else:
                # For numeric types, fill with median
                df[col] = df[col].fillna(df[col].median())
        
        # Encode categorical features and ensure all columns are numeric
        for col in df.columns:
            if col != TARGET_COLUMN and (df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])):
                # Use LabelEncoder for all categorical/string columns
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Encode target variable
        if TARGET_COLUMN in df.columns:
            self.label_encoder = LabelEncoder()
            df[TARGET_COLUMN] = self.label_encoder.fit_transform(df[TARGET_COLUMN].astype(str))
            self.class_names = self.label_encoder.classes_
            
        # Separate features and target
        if TARGET_COLUMN in df.columns:
            self.X = df.drop(TARGET_COLUMN, axis=1)
            self.y = df[TARGET_COLUMN]
        else:
            self.X = df
            self.y = None
            
        self.feature_names = self.X.columns.tolist()
        
        # Verify all columns are numeric before scaling
        for col in self.X.columns:
            if not pd.api.types.is_numeric_dtype(self.X[col]):
                print(f"Warning: Column {col} is not numeric. Converting to numeric.")
                self.X[col] = pd.to_numeric(self.X[col], errors='coerce').fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.feature_names
        )
        
        print("Data preprocessing completed.")
        return self.X, self.y
    
    def split_data(self):
        """
        Split the dataset into train, validation, and test sets.
        """
        if self.X is None or self.y is None:
            self.preprocess_data()
            
        print("Splitting data into train, validation, and test sets...")
        
        # First split: training + validation, and test
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=self.y
        )
        
        # Second split: training and validation
        val_ratio = VAL_SIZE / (1 - TEST_SIZE)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_train_val
        )
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)
    
    def apply_smote(self, X=None, y=None):
        """
        Apply SMOTE to handle class imbalance on the training data.
        """
        X_input = X if X is not None else self.X_train
        y_input = y if y is not None else self.y_train
        
        print("Applying SMOTE to handle class imbalance...")
        # Dynamically adjust k_neighbors based on smallest class count
        from collections import Counter
        counts = Counter(y_input)
        min_count = min(counts.values())
        if min_count < 2:
            print(f"SMOTE skipped: only {min_count} sample(s) in smallest class")
            return X_input, y_input
        # ensure at least 1 neighbor
        k_neighbors = min(SMOTE_K_NEIGHBORS, min_count - 1)
        k_neighbors = max(1, k_neighbors)
        # apply SMOTE with adjusted neighbors
        smote = SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            k_neighbors=k_neighbors,
            random_state=RANDOM_STATE
        )
        try:
            X_resampled, y_resampled = smote.fit_resample(X_input, y_input)
        except ValueError as e:
            print(f"SMOTE error: {e}. Skipping SMOTE.")
            return X_input, y_input
        
        # Update original data if no specific data provided
        if X is None and y is None:
            self.X_train, self.y_train = X_resampled, y_resampled
            
        # Print class distribution after SMOTE
        unique, counts = np.unique(y_resampled, return_counts=True)
        print("Class distribution after SMOTE:")
        class_dict = dict(zip(unique, counts))
        for class_idx, count in class_dict.items():
            if self.class_names is not None and len(self.class_names) > class_idx:
                class_name = self.class_names[class_idx]
                print(f"  {class_name}: {count}")
            else:
                print(f"  Class {class_idx}: {count}")
        
        return X_resampled, y_resampled
    
    def create_non_iid_partitions(self, distribution_type="label_skew", num_clients=5):
        """
        Create non-IID data partitions for federated learning.
        
        Args:
            distribution_type: Type of non-IID distribution ('label_skew' or 'feature_skew')
            num_clients: Number of clients to create partitions for
            
        Returns:
            List of (X, y) tuples for each client
        """
        if self.X_train is None or self.y_train is None:
            self.split_data()
        
        print(f"Creating non-IID partitions with {distribution_type} distribution...")
        X_train_np = self.X_train.to_numpy()
        y_train_np = self.y_train.to_numpy()
        
        partitions = []
        
        if distribution_type == "label_skew":
            # Create label-skewed partitions (each client has different class distributions)
            unique_labels = np.unique(y_train_np)
            num_labels = len(unique_labels)
            
            # Assign dominant labels to each client
            dominant_labels_per_client = max(1, num_labels // num_clients)
            
            # Create client partitions
            for i in range(num_clients):
                # Assign dominant labels to this client (with some overlap between clients)
                start_idx = (i * dominant_labels_per_client) % num_labels
                dominant_labels = unique_labels[start_idx:start_idx + dominant_labels_per_client]
                if start_idx + dominant_labels_per_client > num_labels:
                    dominant_labels = np.append(
                        dominant_labels, 
                        unique_labels[:(start_idx + dominant_labels_per_client) % num_labels]
                    )
                
                # Select samples for dominant classes (higher probability)
                dominant_mask = np.isin(y_train_np, dominant_labels)
                dominant_indices = np.where(dominant_mask)[0]
                
                # Select some samples from other classes (lower probability)
                other_mask = ~dominant_mask
                other_indices = np.where(other_mask)[0]
                
                # Sample size for this client (roughly equal partition size)
                total_samples = len(X_train_np) // num_clients
                
                # 70% dominant classes, 30% other classes
                n_dominant = min(int(total_samples * 0.7), len(dominant_indices))
                n_other = min(total_samples - n_dominant, len(other_indices))
                
                # Randomly select indices
                selected_dominant = np.random.choice(dominant_indices, n_dominant, replace=False)
                selected_other = np.random.choice(other_indices, n_other, replace=False)
                selected_indices = np.concatenate([selected_dominant, selected_other])
                
                # Create client dataset
                X_client = X_train_np[selected_indices]
                y_client = y_train_np[selected_indices]
                
                # Convert back to DataFrame for consistency
                X_client_df = pd.DataFrame(X_client, columns=self.feature_names)
                
                partitions.append((X_client_df, y_client))
                
                print(f"Client {i} partition size: {len(X_client)} samples")
                unique, counts = np.unique(y_client, return_counts=True)
                print(f"Client {i} class distribution: {dict(zip(unique, counts))}")
        
        elif distribution_type == "feature_skew":
            # Feature-skewed partitions (each client has different feature distributions)
            # Divide features into partially overlapping groups
            n_features = X_train_np.shape[1]
            features_per_client = int(n_features * 0.6)  # Each client gets 60% of features
            
            # Create client partitions
            for i in range(num_clients):
                # Create a mask for feature selection with some overlap
                start_idx = (i * features_per_client // 2) % n_features
                end_idx = (start_idx + features_per_client) % n_features
                
                feature_mask = np.zeros(n_features, dtype=bool)
                if end_idx > start_idx:
                    feature_mask[start_idx:end_idx] = True
                else:
                    feature_mask[start_idx:] = True
                    feature_mask[:end_idx] = True
                
                # Add some random features for diversity
                remaining_features = np.where(~feature_mask)[0]
                random_features = np.random.choice(
                    remaining_features,
                    size=min(int(n_features * 0.1), len(remaining_features)),
                    replace=False
                )
                feature_mask[random_features] = True
                
                # Get indices for this client (random subset of data)
                client_size = len(X_train_np) // num_clients
                client_indices = np.random.choice(
                    len(X_train_np), size=client_size, replace=False
                )
                
                # Create masked version of X where only selected features have true values
                X_client = X_train_np[client_indices].copy()
                y_client = y_train_np[client_indices].copy()
                
                # Zero out non-selected features (simulate missing or non-informative features)
                non_selected = ~feature_mask
                # Add small noise to non-selected features instead of zeroing them out
                X_client[:, non_selected] = np.random.normal(
                    0, 0.01, size=(X_client.shape[0], np.sum(non_selected))
                )
                
                # Convert back to DataFrame
                selected_features = [f for f, m in zip(self.feature_names, feature_mask) if m]
                X_client_df = pd.DataFrame(X_client, columns=self.feature_names)
                
                partitions.append((X_client_df, y_client))
                
                print(f"Client {i} partition size: {len(X_client)} samples")
                print(f"Client {i} has {sum(feature_mask)} informative features")
        
        else:  # IID partitions
            # Just randomly split the data equally
            indices = np.random.permutation(len(X_train_np))
            partition_size = len(indices) // num_clients
            
            for i in range(num_clients):
                start_idx = i * partition_size
                end_idx = (i + 1) * partition_size if i < num_clients - 1 else len(indices)
                client_indices = indices[start_idx:end_idx]
                
                X_client = X_train_np[client_indices]
                y_client = y_train_np[client_indices]
                
                # Convert back to DataFrame
                X_client_df = pd.DataFrame(X_client, columns=self.feature_names)
                
                partitions.append((X_client_df, y_client))
                
                print(f"Client {i} partition size: {len(X_client)} samples")
        
        return partitions
    
    def create_iid_partitions(self, num_clients=5):
        """
        Create IID data partitions for federated learning.
        
        Args:
            num_clients: Number of clients to create partitions for
            
        Returns:
            List of (X, y) tuples for each client
        """
        return self.create_non_iid_partitions(distribution_type="iid", num_clients=num_clients)
    
    def save_preprocessor(self, save_dir=None):
        """
        Save the preprocessing objects for later use.
        """
        if save_dir is None:
            save_dir = os.path.join(RESULTS_DIR, "preprocessor")
            
        os.makedirs(save_dir, exist_ok=True)
        
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(save_dir, "scaler.joblib"))
            
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, os.path.join(save_dir, "label_encoder.joblib"))
            
        # Save feature names
        if self.feature_names is not None:
            pd.Series(self.feature_names).to_csv(
                os.path.join(save_dir, "feature_names.csv"), index=False
            )
            
        print(f"Preprocessor objects saved to {save_dir}")
        
    def load_preprocessor(self, load_dir=None):
        """
        Load the preprocessing objects from disk.
        """
        if load_dir is None:
            load_dir = os.path.join(RESULTS_DIR, "preprocessor")
            
        if not os.path.exists(load_dir):
            print(f"Preprocessor directory not found: {load_dir}")
            return False
            
        try:
            scaler_path = os.path.join(load_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            encoder_path = os.path.join(load_dir, "label_encoder.joblib")
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                self.class_names = self.label_encoder.classes_
                
            feature_names_path = os.path.join(load_dir, "feature_names.csv")
            if os.path.exists(feature_names_path):
                self.feature_names = pd.read_csv(feature_names_path).iloc[:, 0].tolist()
                
            print(f"Preprocessor objects loaded from {load_dir}")
            return True
        except Exception as e:
            print(f"Error loading preprocessor objects: {e}")
            return False 