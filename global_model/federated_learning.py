"""
Federated learning system for the HETROFL project.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import (
    COMMUNICATION_ROUNDS,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    CLIENT_MODELS,
    RANDOM_STATE,
    CUMULATIVE_LEARNING,
    GLOBAL_TO_LOCAL_ALPHA
)
from utils.metrics import ModelEvaluator
from utils.logger import ExperimentLogger
from utils.model_persistence import ModelTracker
from visualization.learning_visualizer import LearningVisualizer
from global_model.knowledge_distillation import KnowledgeDistillation


class HeterogeneousFederatedLearning:
    """
    Heterogeneous Federated Learning system that manages local models and global aggregation.
    """
    
    def __init__(self, data_loader, local_model_classes, experiment_name=None):
        """
        Initialize the federated learning system.
        
        Args:
            data_loader: DataLoader object for handling dataset
            local_model_classes: Dictionary mapping model names to model classes
            experiment_name: Name for the experiment (optional)
        """
        self.data_loader = data_loader
        self.local_model_classes = local_model_classes
        self.local_models = {}
        self.global_model = None
        self.input_dim = None
        self.output_dim = None
        
        # Set up logger
        self.logger = ExperimentLogger(experiment_name)
        
        # Set up model tracker for cumulative learning
        self.model_tracker = ModelTracker()
        
        # Set up visualizer
        self.visualizer = LearningVisualizer(save_dir=os.path.join(self.logger.experiment_dir, "plots"))
        
        # Set up evaluator
        self.evaluator = None
        
        # Metrics for tracking knowledge transfer impact
        self.pre_transfer_metrics = {}
        self.post_transfer_metrics = {}
    
    def initialize_system(self, data_distribution="iid", load_previous_models=CUMULATIVE_LEARNING):
        """
        Initialize the federated learning system.
        
        Args:
            data_distribution: Type of data distribution ('iid', 'non_iid_label_skew', 'non_iid_feature_skew')
            load_previous_models: Whether to load previously trained models for cumulative learning
            
        Returns:
            Initialized system
        """
        print("Initializing Heterogeneous Federated Learning system...")
        
        # Load and preprocess data
        self.data_loader.load_data()
        self.data_loader.explore_data()
        self.data_loader.preprocess_data()
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.data_loader.split_data()
        
        # Set dimensions
        self.input_dim = X_train.shape[1]
        self.output_dim = len(np.unique(y_train))
        
        # Set up evaluator with class names
        self.evaluator = ModelEvaluator(class_names=self.data_loader.class_names)
        
        # Create data partitions
        if data_distribution == "iid":
            client_partitions = self.data_loader.create_iid_partitions(num_clients=len(CLIENT_MODELS))
        else:
            client_partitions = self.data_loader.create_non_iid_partitions(
                distribution_type=data_distribution.replace("non_iid_", ""),
                num_clients=len(CLIENT_MODELS)
            )
        
        # Initialize local models
        for i, (model_name, model_class) in enumerate(self.local_model_classes.items()):
            client_id = i
            
            # Get client data
            X_client, y_client = client_partitions[i]
            
            # Apply SMOTE to handle class imbalance - use all data for training
            X_client_resampled, y_client_resampled = self.data_loader.apply_smote(X_client, y_client)
            
            # Initialize model
            model = model_class(client_id=client_id, output_dim=self.output_dim)
            if hasattr(model, 'input_dim') and model.input_dim is None:
                model.input_dim = self.input_dim
            
            # Load previous model if available and cumulative learning is enabled
            if load_previous_models:
                try:
                    model_path = self.model_tracker.get_latest_model_path('local', client_id)
                    if model_path:
                        print(f"Loading previous model for client {client_id}...")
                        model.load_model(model_path)
                        print(f"Previous model loaded successfully for client {client_id}")
                except Exception as e:
                    print(f"Error loading previous model for client {client_id}: {e}")
                    print("Initializing new model instead.")
            
            # Store model and data
            self.local_models[client_id] = {
                'model': model,
                'X_train': X_client_resampled,
                'y_train': y_client_resampled,
                'X_val': X_val,
                'y_val': y_val
            }
        
        # Initialize global model
        self.global_model = KnowledgeDistillation(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        # Load previous global model if available and cumulative learning is enabled
        if load_previous_models:
            try:
                global_model_path = self.model_tracker.get_latest_model_path('global')
                if global_model_path:
                    print("Loading previous global model...")
                    self.global_model.load_model(global_model_path)
                    print("Previous global model loaded successfully")
            except Exception as e:
                print(f"Error loading previous global model: {e}")
                print("Initializing new global model instead.")
        
        # Log system parameters
        self.logger.log_parameters({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_clients': len(self.local_models),
            'data_distribution': data_distribution,
            'client_models': list(self.local_model_classes.keys()),
            'communication_rounds': COMMUNICATION_ROUNDS,
            'local_epochs': LOCAL_EPOCHS,
            'cumulative_learning': load_previous_models
        })
        
        self.logger.log_stage("initialization", {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_clients': len(self.local_models),
            'data_distribution': data_distribution,
            'cumulative_learning': load_previous_models
        })
        
        print("System initialized successfully.")
        
        return self
    
    def train_local_models(self, hyperparameter_tuning=True, local_epochs=None):
        """
        Train all local models.
        
        Args:
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            local_epochs: Number of epochs for local training (overrides default)
            
        Returns:
            Dictionary of trained local models
        """
        print("Training local models...")
        
        # Use specified local_epochs or default from config
        if local_epochs is None:
            from config.config import LOCAL_EPOCHS
            local_epochs = LOCAL_EPOCHS
            
        print(f"Training local models for {local_epochs} epochs each")
        
        for client_id, client_data in self.local_models.items():
            model = client_data['model']
            X_train = client_data['X_train']
            y_train = client_data['y_train']
            X_val = client_data['X_val']
            y_val = client_data['y_val']
            
            print(f"\nTraining {model.model_name} model for client {client_id}...")
            
            # Hyperparameter tuning
            if hyperparameter_tuning:
                print(f"Performing hyperparameter tuning for {model.model_name}...")
                try:
                    best_params = model.tune_hyperparameters(X_train, y_train, X_val, y_val)
                    print(f"Best parameters for {model.model_name}: {best_params}")
                except Exception as e:
                    print(f"Error during hyperparameter tuning: {e}")
                    print("Continuing with default parameters.")
            
            # Train the model
            try:
                model.fit(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    epochs=local_epochs
                )
                
                # Evaluate the model
                y_val_pred = model.predict(X_val)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                val_f1 = f1_score(y_val, y_val_pred, average='weighted')
                
                print(f"{model.model_name} validation - Accuracy: {val_accuracy:.4f}, F1 score: {val_f1:.4f}")
                
                # Log metrics
                metrics = {
                    'val_accuracy': val_accuracy,
                    'val_f1_score': val_f1
                }
                
                self.logger.log_model_performance(
                    model_name=f"{model.model_name}_client_{client_id}",
                    metrics=metrics,
                    model_type="local"
                )
                
            except Exception as e:
                print(f"Error training {model.model_name}: {e}")
                self.logger.log_exception(e, f"training_{model.model_name}_client_{client_id}")
        
        self.logger.log_stage("local_training_completed")
        
        print("Local model training completed.")
        
        return self.local_models
    
    def aggregate_knowledge(self):
        """
        Aggregate knowledge from local models using knowledge distillation.
        
        Returns:
            Trained global model
        """
        print("\nAggregating knowledge from local models...")
        
        # Get list of trained local models
        trained_models = []
        for client_id, client_data in self.local_models.items():
            model = client_data['model']
            if model.is_fitted:
                trained_models.append(model)
        
        if not trained_models:
            print("Warning: No trained local models available for aggregation")
            return self.global_model
        
        # Initialize global model if it doesn't exist
        if self.global_model is None:
            print("Initializing global model...")
            self.global_model = KnowledgeDistillation(
                input_dim=self.input_dim,
                output_dim=self.output_dim
            )
            self.global_model.build_model()
        
        # Add local models to the global model
        self.global_model.local_models = []  # Reset local models list
        for model in trained_models:
            self.global_model.add_local_model(model)
        
        # Get training and validation data
        X_train = self.data_loader.X_train
        y_train = self.data_loader.y_train
        X_val = self.data_loader.X_val
        y_val = self.data_loader.y_val
        
        # Train global model
        try:
            print(f"Training global model with {len(trained_models)} local models...")
            self.global_model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate global model
            y_val_pred = self.global_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            print(f"Global model validation - Accuracy: {val_accuracy:.4f}, F1 score: {val_f1:.4f}")
            
            # Log metrics
            metrics = {
                'val_accuracy': val_accuracy,
                'val_f1_score': val_f1
            }
            
            self.logger.log_model_performance(
                model_name="global_model",
                metrics=metrics,
                model_type="global"
            )
            
        except Exception as e:
            print(f"Error training global model: {e}")
            import traceback
            traceback.print_exc()
            self.logger.log_exception(e, "training_global_model")
        
        self.logger.log_stage("knowledge_aggregation_completed")
        
        print("Knowledge aggregation completed.")
        
        return self.global_model
    
    def run_federated_learning(self, communication_rounds=None, hyperparameter_tuning=True, 
                             data_distribution="iid", load_previous_models=CUMULATIVE_LEARNING,
                             local_epochs=None):
        """
        Run the federated learning process.
        
        Args:
            communication_rounds: Number of communication rounds
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            data_distribution: Type of data distribution
            load_previous_models: Whether to load previously trained models for cumulative learning
            local_epochs: Number of epochs for local training (overrides default)
            
        Returns:
            Trained global model
        """
        # Set number of communication rounds
        if communication_rounds is None:
            from config.config import COMMUNICATION_ROUNDS
            communication_rounds = COMMUNICATION_ROUNDS
            
        # Set number of local epochs
        if local_epochs is None:
            from config.config import LOCAL_EPOCHS
            local_epochs = LOCAL_EPOCHS
        
        print(f"Starting federated learning with {communication_rounds} communication rounds...")
        print(f"Local models will train for {local_epochs} epochs per round")
        print(f"Cumulative learning is {'enabled' if load_previous_models else 'disabled'}")
        
        # Initialize the system
        self.initialize_system(data_distribution=data_distribution, load_previous_models=load_previous_models)
        
        # Ensure the global model is built
        if self.global_model and not self.global_model.is_fitted:
            print("Building global model architecture...")
            self.global_model.build_model()
        
        # Log the start of federated learning
        self.logger.log_stage("federated_learning_started", {
            'communication_rounds': communication_rounds,
            'hyperparameter_tuning': hyperparameter_tuning,
            'data_distribution': data_distribution,
            'cumulative_learning': load_previous_models
        })
        
        # Run communication rounds
        for round_num in range(1, communication_rounds + 1):
            print(f"\n--- Communication Round {round_num}/{communication_rounds} ---")
            
            # Train local models
            self.train_local_models(
                hyperparameter_tuning=(round_num == 1 and hyperparameter_tuning),
                local_epochs=local_epochs
            )
            
            # Use the enhanced transfer_knowledge method for bidirectional knowledge transfer
            self.transfer_knowledge(round_num)
            
            # Evaluate and log round results
            self.evaluate_round(round_num)
        
        # Final evaluation
        test_metrics = self.evaluate_global_model()
        
        # Save models for cumulative learning
        self.save_models()
        
        # Generate final visualizations
        self.generate_visualizations()
        
        # Log the completion of federated learning
        self.logger.log_stage("federated_learning_completed", {
            'final_test_metrics': test_metrics
        })
        
        # Close logger and save all data
        self.logger.close()
        
        print("Federated learning completed.")
        
        return self.global_model
    
    def evaluate_local_models(self, stage_name=""):
        """
        Evaluate all local models.
        
        Args:
            stage_name: Name of the evaluation stage
            
        Returns:
            Dictionary of evaluation metrics for each local model
        """
        print(f"\nEvaluating local models ({stage_name})...")
        
        # Validation data
        X_val = self.data_loader.X_val
        y_val = self.data_loader.y_val
        
        # Evaluate local models
        local_metrics = {}
        for client_id, client_data in self.local_models.items():
            model = client_data['model']
            if model.is_fitted:
                try:
                    y_val_pred = model.predict(X_val)
                    y_val_proba = model.predict_proba(X_val)
                    
                    # Calculate metrics
                    metrics = self.evaluator.evaluate_classifier(
                        y_val, y_val_pred, y_val_proba, 
                        model_name=f"client_{client_id}_{model.model_name}"
                    )
                    
                    # include confusion matrix and class names for local plotting
                    metrics['confusion_matrix'] = confusion_matrix(y_val, y_val_pred)
                    metrics['class_names'] = self.data_loader.class_names
                    
                    local_metrics[client_id] = metrics
                    
                    print(f"Client {client_id} ({model.model_name}) - "
                          f"Accuracy: {metrics['accuracy']:.4f}, F1 score: {metrics['f1_weighted']:.4f}")
                except Exception as e:
                    print(f"Error evaluating client {client_id} ({model.model_name}): {e}")
        
        return local_metrics
    
    def evaluate_round(self, round_num):
        """
        Evaluate the current round.
        
        Args:
            round_num: Current round number
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating Round {round_num}...")
        
        # Validation data
        X_val = self.data_loader.X_val
        y_val = self.data_loader.y_val
        
        # Evaluate local models
        local_metrics = {}
        for client_id, client_data in self.local_models.items():
            model = client_data['model']
            if model.is_fitted:
                try:
                    y_val_pred = model.predict(X_val)
                    
                    # compute metrics
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
                    # include confusion matrix and class names
                    cm = confusion_matrix(y_val, y_val_pred)
                    metrics = {'accuracy': val_accuracy, 'f1_score': val_f1,
                               'confusion_matrix': cm,
                               'class_names': self.data_loader.class_names}
                    # save local confusion plot
                    fig, ax = plt.subplots(figsize=(6,5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=self.data_loader.class_names,
                                yticklabels=self.data_loader.class_names, ax=ax)
                    fig.savefig(os.path.join(self.visualizer.save_dir, f"local_client_{client_id}_round_{round_num}.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    local_metrics[client_id] = metrics
                    
                    print(f"Client {client_id} ({model.model_name}) - Accuracy: {val_accuracy:.4f}, F1 score: {val_f1:.4f}")
                except Exception as e:
                    print(f"Error evaluating client {client_id} ({model.model_name}): {e}")
        
        # Evaluate global model
        try:
            y_val_global_pred = self.global_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_global_pred)
            val_f1 = f1_score(y_val, y_val_global_pred, average='weighted')
            cm = confusion_matrix(y_val, y_val_global_pred)
            global_metrics = {'accuracy': val_accuracy, 'f1_score': val_f1,
                               'confusion_matrix': cm,
                               'class_names': self.data_loader.class_names}
            # save global confusion plot
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.data_loader.class_names,
                        yticklabels=self.data_loader.class_names, ax=ax)
            fig.savefig(os.path.join(self.visualizer.save_dir, f"global_round_{round_num}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Global model - Accuracy: {val_accuracy:.4f}, F1 score: {val_f1:.4f}")
        except Exception as e:
            print(f"Error evaluating global model: {e}")
    
        # Log round metrics
        self.logger.log_round(round_num, global_metrics, local_metrics)
        
        # Return metrics
        return {
            'global_metrics': global_metrics,
            'local_metrics': local_metrics
        }
    
    def evaluate_global_model(self):
        """
        Evaluate the global model on the test set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating global model on test set...")
        
        # Test data
        X_test = self.data_loader.X_test
        y_test = self.data_loader.y_test
        
        if not self.global_model or not self.global_model.is_fitted:
            print("Global model not fitted. Cannot evaluate.")
            return None
        
        try:
            # Make predictions
            y_test_pred = self.global_model.predict(X_test)
            y_test_proba = self.global_model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self.evaluator.evaluate_classifier(
                y_test, y_test_pred, y_test_proba, model_name="global_model"
            )
            
            # Print classification report
            print("\nClassification Report:")
            print(self.evaluator.print_classification_report(y_test, y_test_pred, "Global Model"))
            
            # Generate plots
            print("Generating evaluation plots...")
            
            # Confusion matrix
            self.evaluator.plot_confusion_matrix(
                y_test, y_test_pred, 
                model_name="Global Model", 
                save_path=os.path.join(self.logger.experiment_dir, "confusion_matrix.png")
            )
            
            # ROC curves
            self.evaluator.plot_roc_curves(
                y_test, y_test_proba, 
                model_name="Global Model", 
                save_path=os.path.join(self.logger.experiment_dir, "roc_curves.png")
            )
            
            # Precision-Recall curves
            self.evaluator.plot_precision_recall_curves(
                y_test, y_test_proba, 
                model_name="Global Model", 
                save_path=os.path.join(self.logger.experiment_dir, "pr_curves.png")
            )
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(os.path.join(self.logger.experiment_dir, "global_model_metrics.csv"), index=False)
            
            print(f"Test accuracy: {metrics['accuracy']:.4f}")
            print(f"Test F1 score (weighted): {metrics['f1_weighted']:.4f}")
            
            # Log final evaluation
            self.logger.log_stage("final_evaluation", metrics)
            
            # Save the global model
            try:
                global_model_path = self.global_model.save_model(
                    save_dir=os.path.join(self.logger.experiment_dir, "models/global")
                )
            except Exception as e:
                print(f"Error saving global model: {e}")
                global_model_path = None
            
            # Register global model performance in the model tracker
            if global_model_path:
                self.model_tracker.register_model_paths(
                    model_type='global',
                    client_id=None,
                    model_paths=global_model_path,
                    metrics=metrics
                )
                print(f"Global model saved and registered at: {global_model_path}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating global model: {e}")
            self.logger.log_exception(e, "evaluating_global_model")
            return None
    
    def save_models(self, save_dir=None):
        """
        Save all models.
        
        Args:
            save_dir: Custom directory to save models (optional)
            
        Returns:
            Dictionary of saved model paths
        """
        print("\nSaving models...")
        
        saved_paths = {}
        
        # Save local models
        for client_id, client_data in self.local_models.items():
            model = client_data['model']
            if model.is_fitted:
                try:
                    # Pass save_dir to model.save_model if specified
                    if save_dir:
                        client_save_dir = os.path.join(save_dir, f"client_{client_id}")
                        path = model.save_model(save_dir=client_save_dir)
                    else:
                        path = model.save_model()
                        
                    saved_paths[f"client_{client_id}"] = path
                    
                    # Register local model in the model tracker
                    X_val = client_data['X_val']
                    y_val = client_data['y_val']
                    y_val_pred = model.predict(X_val)
                    accuracy = accuracy_score(y_val, y_val_pred)
                    f1 = f1_score(y_val, y_val_pred, average='weighted')
                    
                    self.model_tracker.register_model_paths(
                        model_type='local',
                        client_id=client_id,
                        model_paths=path,
                        metrics={'accuracy': accuracy, 'f1_weighted': f1}
                    )
                except Exception as e:
                    print(f"Error saving model for client {client_id}: {e}")
        
        # Save global model
        if self.global_model and self.global_model.is_fitted:
            try:
                # Pass save_dir to global_model.save_model if specified
                if save_dir:
                    global_save_dir = os.path.join(save_dir, "global")
                    path = self.global_model.save_model(save_dir=global_save_dir)
                else:
                    path = self.global_model.save_model()
                    
                saved_paths["global_model"] = path
            except Exception as e:
                print(f"Error saving global model: {e}")
        
        self.logger.log_stage("models_saved", saved_paths)
        
        return saved_paths
    
    def update_local_models_with_global_knowledge(self):
        """
        Update local models with knowledge from the global model.
        
        Returns:
            Updated local models
        """
        print("\nUpdating local models with global knowledge...")
        
        if not self.global_model or not hasattr(self.global_model, 'update_local_models'):
            print("Global model does not support updating local models.")
            return self.local_models
        
        # Get training data for knowledge transfer
        X_train = self.data_loader.X_train
        y_train = self.data_loader.y_train
        
        # Extract just the model objects from the local_models dictionary
        local_model_objects = {
            client_id: client_data['model'] 
            for client_id, client_data in self.local_models.items()
        }
        
        # Update local models with global knowledge
        updated_models = self.global_model.update_local_models(
            local_model_objects, X_train, y_train
        )
        
        # Update the local models in our system
        for client_id, updated_model in updated_models.items():
            if client_id in self.local_models:
                self.local_models[client_id]['model'] = updated_model
        
        self.logger.log_stage("local_models_updated_with_global_knowledge")
        
        return self.local_models
    
    def generate_visualizations(self):
        """
        Generate comprehensive visualizations of the federated learning process.
        
        Returns:
            Dictionary of visualization paths
        """
        print("\nGenerating visualizations...")
        
        visualization_paths = {}
        
        # Plot performance history
        history_plots = self.model_tracker.plot_performance_history(
            save_dir=os.path.join(self.logger.experiment_dir, "plots/history")
        )
        visualization_paths.update(history_plots)
        
        # Get performance summary
        performance_summary = self.model_tracker.get_performance_summary()
        
        # Save performance summary to CSV
        summary_df = pd.DataFrame([performance_summary["global_model"]])
        summary_df.to_csv(
            os.path.join(self.logger.experiment_dir, "global_model_performance_summary.csv"),
            index=False
        )
        
        # Save local models performance summary
        local_summary_data = []
        for client_id, summary in performance_summary["local_models"].items():
            summary["client_id"] = client_id
            local_summary_data.append(summary)
        
        if local_summary_data:
            local_summary_df = pd.DataFrame(local_summary_data)
            local_summary_df.to_csv(
                os.path.join(self.logger.experiment_dir, "local_models_performance_summary.csv"),
                index=False
            )
        
        # Plot feature importance for each model if available
        for client_id, client_data in self.local_models.items():
            model = client_data['model']
            if hasattr(model, 'get_feature_importance'):
                try:
                    importance_df = model.get_feature_importance(feature_names=self.data_loader.feature_names)
                    if importance_df is not None:
                        importance_path = os.path.join(
                            self.visualizer.save_dir,
                            f"feature_importance_client_{client_id}.png"
                        )
                        self.visualizer.plot_feature_importance(
                            importance_df,
                            model_name=f"Client {client_id} ({model.model_name})",
                            save_path=importance_path
                        )
                        visualization_paths[f"importance_client_{client_id}"] = importance_path
                except Exception as e:
                    print(f"Error generating feature importance for client {client_id}: {e}")
        
        # Generate t-SNE visualization of the data
        try:
            X_sample = self.data_loader.X_train.sample(min(1000, len(self.data_loader.X_train)))
            y_sample = self.data_loader.y_train.loc[X_sample.index]
            
            tsne_path = os.path.join(self.visualizer.save_dir, "tsne_visualization.png")
            self.visualizer.plot_embedding_visualization(
                X_sample, y_sample,
                method='tsne',
                save_path=tsne_path
            )
            visualization_paths["tsne"] = tsne_path
        except Exception as e:
            print(f"Error generating t-SNE visualization: {e}")
        
        print(f"Visualizations saved to: {self.visualizer.save_dir}")
        
        return visualization_paths
    
    def transfer_knowledge(self, communication_round):
        """
        Transfer knowledge between global and local models through bidirectional knowledge transfer
        with adaptive weights based on round number and model performance.
        
        Args:
            communication_round: Current communication round number
        
        Returns:
            Dictionary of updated local models
        """
        print("\nTransferring knowledge between models...")
        
        # Aggregate knowledge from local models to global model
        self.aggregate_knowledge()
        
        # Store pre-transfer metrics
        pre_transfer_metrics = self.evaluate_local_models(stage_name=f"pre_transfer_round_{communication_round}")
        self.pre_transfer_metrics[communication_round] = pre_transfer_metrics
        
        # Calculate global adaptive alpha based on round number
        # This increases knowledge transfer impact as rounds progress
        # The idea is to share more knowledge as the global model improves
        base_alpha = GLOBAL_TO_LOCAL_ALPHA
        round_factor = min(0.5, 0.1 * communication_round)  # Caps at +50% after 5 rounds
        global_adaptive_alpha = min(0.7, base_alpha * (1 + round_factor))  # Cap at 0.7
        
        print(f"Using adaptive knowledge transfer weight (base={base_alpha:.2f}, round={communication_round}, adaptive={global_adaptive_alpha:.2f})")
        
        # Check which models should receive knowledge transfer based on model type, current performance and history
        should_update = {}
        alpha_values = {}  # Store model-specific alpha values
        
        # Calculate average global model performance
        global_model_accuracy = 0
        global_model_f1 = 0
        
        if self.global_model and self.global_model.is_fitted:
            X_val = self.data_loader.X_val
            y_val = self.data_loader.y_val
            y_val_pred = self.global_model.predict(X_val)
            global_model_accuracy = accuracy_score(y_val, y_val_pred)
            global_model_f1 = f1_score(y_val, y_val_pred, average='weighted')
            print(f"Global model performance - Accuracy: {global_model_accuracy:.4f}, F1 score: {global_model_f1:.4f}")
        
        for client_id, client_data in self.local_models.items():
            model = client_data['model']
            model_type = model.model_name
            
            # By default, receive knowledge transfer
            should_update[client_id] = True
            
            # Get current model performance
            if client_id in pre_transfer_metrics:
                current_accuracy = pre_transfer_metrics[client_id].get('accuracy', 0)
                current_f1 = pre_transfer_metrics[client_id].get('f1_weighted', 0)
                
                # Check performance relative to global model
                if global_model_accuracy > 0:
                    # If local model is better than global model, apply less knowledge transfer
                    if current_accuracy >= global_model_accuracy * 1.02:  # Local model is 2% better
                        model_alpha = global_adaptive_alpha * 0.3  # Reduce alpha significantly
                        print(f"Model {model_type}_client_{client_id} is performing better than global model. Using reduced alpha={model_alpha:.3f}")
                    # If local model is significantly worse, apply more knowledge transfer
                    elif current_accuracy < global_model_accuracy * 0.95:  # Local model is 5% worse
                        model_alpha = min(0.8, global_adaptive_alpha * 1.5)  # Increase alpha, but cap at 0.8
                        print(f"Model {model_type}_client_{client_id} is performing worse than global model. Using increased alpha={model_alpha:.3f}")
                    else:
                        # Similar performance, use standard alpha
                        model_alpha = global_adaptive_alpha
                
                # Special handling for CNN and other neural network models
                if model_type in ['cnn', 'autoencoder']:
                    # For neural networks, we need to be more conservative to prevent catastrophic forgetting
                    model_alpha = model_alpha * 0.5
                    
                    # Check if this is the first round (always transfer in first round)
                    if communication_round > 1 and client_id in pre_transfer_metrics:
                        current_accuracy = pre_transfer_metrics[client_id].get('accuracy', 0)
                        
                        # Get previous round metrics if available
                        prev_round = communication_round - 1
                        if prev_round in self.post_transfer_metrics and client_id in self.post_transfer_metrics[prev_round]:
                            prev_accuracy = self.post_transfer_metrics[prev_round][client_id].get('accuracy', 0)
                            
                            # If current performance is better than previous round by a threshold, keep training without transfer
                            if current_accuracy > prev_accuracy * 1.05:  # 5% improvement threshold
                                should_update[client_id] = False
                                print(f"Skipping knowledge transfer for client {client_id} ({model_type}) "
                                     f"as it's improving well on its own")
                            # If performance is getting worse, apply more aggressive knowledge transfer
                            elif current_accuracy < prev_accuracy * 0.98:  # Performance degrading
                                model_alpha = min(0.6, model_alpha * 1.5)  # Increase alpha but cap at 0.6 for neural nets
                                print(f"Model {model_type}_client_{client_id} is degrading. Using increased alpha={model_alpha:.3f}")
            
            # Store the final alpha value for this model
            alpha_values[client_id] = model_alpha
        
        # Transfer knowledge from global to local models
        print("Transferring knowledge from global to local models...")
        
        for client_id, client_data in self.local_models.items():
            if should_update[client_id]:
                model = client_data['model']
                model_type = model.model_name
                
                # Get data for this client
                X_val = client_data['X_val'] 
                y_val = client_data['y_val']
                alpha = alpha_values[client_id]
                
                try:
                    # Get global model predictions on validation data
                    global_preds_proba = self.global_model.predict_proba(X_val)
                    
                    # Get global model feature importance if available
                    global_feature_importance = None
                    if hasattr(self.global_model, 'get_feature_importance') and callable(getattr(self.global_model, 'get_feature_importance')):
                        try:
                            global_feature_importance = self.global_model.get_feature_importance()
                        except Exception as e:
                            print(f"Could not get global feature importance: {e}")
                    
                    # Update local model with global knowledge
                    if hasattr(model, 'update_with_global_knowledge') and callable(getattr(model, 'update_with_global_knowledge')):
                        # Progressive learning approach - use additional parameters to guide the model
                        extra_params = {
                            'round_number': communication_round,
                            'current_performance': pre_transfer_metrics.get(client_id, {}),
                            'global_performance': {'accuracy': global_model_accuracy, 'f1_score': global_model_f1}
                        }
                        
                        model.update_with_global_knowledge(
                            global_preds_proba=global_preds_proba,
                            X_val=X_val,
                            y_val=y_val,
                            global_feature_importance=global_feature_importance,
                            alpha=alpha,
                            **extra_params
                        )
                        print(f"Updated {model_type}_client_{client_id} with global knowledge using alpha={alpha:.3f}")
                    else:
                        print(f"Model {model_type}_client_{client_id} does not support knowledge transfer")
                except Exception as e:
                    print(f"Error updating {model_type}_client_{client_id} with global knowledge: {e}")
            else:
                model = client_data['model']
                print(f"Skipped knowledge transfer for {model.model_name}_client_{client_id}")
        
        # Store post-transfer metrics
        post_transfer_metrics = self.evaluate_local_models(stage_name=f"post_transfer_round_{communication_round}")
        self.post_transfer_metrics[communication_round] = post_transfer_metrics
        
        # Log knowledge transfer results
        self.log_knowledge_transfer_results(pre_transfer_metrics, post_transfer_metrics, communication_round)
        
        # Additional step: retrain models that didn't improve from knowledge transfer
        # This gives a second chance to models that may have regressed
        self._fine_tune_underperforming_models(pre_transfer_metrics, post_transfer_metrics, communication_round)
        
        return self.local_models
    
    def _fine_tune_underperforming_models(self, pre_metrics, post_metrics, communication_round):
        """
        Fine-tune models that didn't improve after knowledge transfer to ensure
        continuous improvement across rounds.
        
        Args:
            pre_metrics: Metrics before knowledge transfer
            post_metrics: Metrics after knowledge transfer
            communication_round: Current round number
        """
        print("\nFine-tuning models that didn't improve after knowledge transfer...")
        
        underperforming_models = []
        
        # Identify models that didn't improve or regressed
        for client_id in pre_metrics.keys():
            if client_id in post_metrics:
                pre_acc = pre_metrics[client_id].get('accuracy', 0)
                post_acc = post_metrics[client_id].get('accuracy', 0)
                
                # If accuracy decreased or stayed the same
                if post_acc <= pre_acc and client_id in self.local_models:
                    model = self.local_models[client_id]['model']
                    model_type = model.model_name
                    
                    # Only apply to non-neural network models for efficiency
                    if model_type not in ['cnn', 'autoencoder']:
                        underperforming_models.append(client_id)
                        print(f"Model {model_type}_client_{client_id} didn't improve. Will fine-tune.")
        
        # Fine-tune underperforming models with a different approach
        for client_id in underperforming_models:
            client_data = self.local_models[client_id]
            model = client_data['model']
            X_train = client_data['X_train']
            y_train = client_data['y_train']
            X_val = client_data['X_val']
            y_val = client_data['y_val']
            
            print(f"\nFine-tuning {model.model_name} model for client {client_id}...")
            
            try:
                # For tree-based models, we can train with different parameters
                if model.model_name in ['xgboost', 'random_forest', 'lightgbm']:
                    # Use a shortened training process with different focus
                    fine_tune_params = {
                        'epochs': 1,  # Short training
                        'learning_rate': 0.01,  # Slightly higher learning rate
                        'fine_tuning': True,  # Flag to indicate fine-tuning
                        'round_number': communication_round  # Pass round number for adaptive behavior
                    }
                    
                    model.fit(
                        X_train, y_train,
                        X_val=X_val, y_val=y_val,
                        **fine_tune_params
                    )
                    
                    # Evaluate after fine-tuning
                    y_val_pred = model.predict(X_val)
                    new_accuracy = accuracy_score(y_val, y_val_pred)
                    new_f1 = f1_score(y_val, y_val_pred, average='weighted')
                    
                    print(f"After fine-tuning: Accuracy: {new_accuracy:.4f}, F1 score: {new_f1:.4f}")
                    print(f"Change from post-transfer: Accuracy: {new_accuracy - post_metrics[client_id].get('accuracy', 0):.4f}")
                    
                    # Update post-transfer metrics
                    post_metrics[client_id]['accuracy'] = new_accuracy
                    post_metrics[client_id]['f1_weighted'] = new_f1
                    
            except Exception as e:
                print(f"Error fine-tuning model for client {client_id}: {e}")
        
        # Update stored post-transfer metrics if we made changes
        if underperforming_models:
            self.post_transfer_metrics[communication_round] = post_metrics
        
        return self.local_models
    
    def log_knowledge_transfer_results(self, pre_metrics, post_metrics, communication_round):
        """
        Log the results of knowledge transfer.
        
        Args:
            pre_metrics: Metrics before knowledge transfer
            post_metrics: Metrics after knowledge transfer
            communication_round: Current round number
        """
        print("\nKnowledge Transfer Results:")
        
        # Create a comparison table
        comparison_data = []
        underperforming_models = []
        
        # Identify models that didn't improve or regressed
        for client_id in pre_metrics.keys():
            if client_id in post_metrics:
                pre_acc = pre_metrics[client_id].get('accuracy', 0)
                post_acc = post_metrics[client_id].get('accuracy', 0)
                
                # If accuracy decreased or stayed the same
                if post_acc <= pre_acc and client_id in self.local_models:
                    model = self.local_models[client_id]['model']
                    model_type = model.model_name
                    
                    # Only apply to non-neural network models for efficiency
                    if model_type not in ['cnn', 'autoencoder']:
                        underperforming_models.append(client_id)
                        print(f"Model {model_type}_client_{client_id} didn't improve. Will fine-tune.")
        
        # Fine-tune underperforming models with a different approach
        for client_id in underperforming_models:
            client_data = self.local_models[client_id]
            model = client_data['model']
            X_train = client_data['X_train']
            y_train = client_data['y_train']
            X_val = client_data['X_val']
            y_val = client_data['y_val']
            
            print(f"\nFine-tuning {model.model_name} model for client {client_id}...")
            
            try:
                # For tree-based models, we can train with different parameters
                if model.model_name in ['xgboost', 'random_forest', 'lightgbm']:
                    # Use a shortened training process with different focus
                    fine_tune_params = {
                        'epochs': 1,  # Short training
                        'learning_rate': 0.01,  # Slightly higher learning rate
                        'fine_tuning': True,  # Flag to indicate fine-tuning
                        'round_number': communication_round  # Pass round number for adaptive behavior
                    }
                    
                    model.fit(
                        X_train, y_train,
                        X_val=X_val, y_val=y_val,
                        **fine_tune_params
                    )
                    
                    # Evaluate after fine-tuning
                    y_val_pred = model.predict(X_val)
                    new_accuracy = accuracy_score(y_val, y_val_pred)
                    new_f1 = f1_score(y_val, y_val_pred, average='weighted')
                    
                    print(f"After fine-tuning: Accuracy: {new_accuracy:.4f}, F1 score: {new_f1:.4f}")
                    print(f"Change from post-transfer: Accuracy: {new_accuracy - post_metrics[client_id].get('accuracy', 0):.4f}")
                    
                    # Update post-transfer metrics
                    post_metrics[client_id]['accuracy'] = new_accuracy
                    post_metrics[client_id]['f1_weighted'] = new_f1
                    
            except Exception as e:
                print(f"Error fine-tuning model for client {client_id}: {e}")
        
        # Update stored post-transfer metrics if we made changes
        if underperforming_models:
            self.post_transfer_metrics[communication_round] = post_metrics
        
        return self.local_models
    
    def run_federated_learning_round(self, round_num, hyperparameter_tuning=False, 
                                data_distribution="iid", local_epochs=None, 
                                adaptive_alpha=None, model_params=None, 
                                learning_rates=None, reinit_probability=0):
        """
        Run a single round of federated learning with round-specific parameters.
        
        Args:
            round_num: Current round number
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            data_distribution: Type of data distribution
            local_epochs: Number of epochs for local training (overrides default)
            adaptive_alpha: Adaptive alpha value for knowledge transfer
            model_params: Dictionary of model-specific parameters
            learning_rates: Dictionary of model-specific learning rates
            reinit_probability: Probability to reinitialize underperforming models
            
        Returns:
            Dictionary of evaluation metrics for this round
        """
        # Log the start of this round
        print(f"\nStarting federated learning round {round_num}")
        self.logger.log_stage(f"round_{round_num}_started", {
            'adaptive_alpha': adaptive_alpha,
            'local_epochs': local_epochs
        })
        
        # Apply round-specific learning rates if provided
        if learning_rates:
            print(f"Applying round-specific learning rates: {learning_rates}")
            for client_id, client_data in self.local_models.items():
                model = client_data['model']
                model_name = model.model_name
                
                if model_name in learning_rates:
                    lr = learning_rates[model_name]
                    print(f"Setting learning rate for {model_name} to {lr}")
                    if hasattr(model, 'set_learning_rate'):
                        model.set_learning_rate(lr)
        
        # Apply round-specific model parameters if provided
        if model_params:
            print(f"Applying round-specific model parameters")
            for client_id, client_data in self.local_models.items():
                model = client_data['model']
                model_name = model.model_name
                
                if model_name in model_params:
                    params = model_params[model_name]
                    print(f"Setting params for {model_name}: {params}")
                    if hasattr(model, 'set_model_params'):
                        model.set_model_params(params)
        
        # Potentially reinitialize underperforming models
        if reinit_probability > 0 and round_num > 1:
            self._consider_model_reinitialization(reinit_probability)
        
        # Train local models for this round
        self.train_local_models(
            hyperparameter_tuning=hyperparameter_tuning,
            local_epochs=local_epochs
        )
        
        # Transfer knowledge with adaptive alpha if provided
        if adaptive_alpha is not None:
            print(f"Using adaptive alpha for knowledge transfer: {adaptive_alpha}")
            # Save original alpha
            from config.config import GLOBAL_TO_LOCAL_ALPHA
            original_alpha = GLOBAL_TO_LOCAL_ALPHA
            
            # Temporarily override the alpha value
            import config.config
            config.config.GLOBAL_TO_LOCAL_ALPHA = adaptive_alpha
            
            # Run knowledge transfer
            self.transfer_knowledge(round_num)
            
            # Restore original alpha
            config.config.GLOBAL_TO_LOCAL_ALPHA = original_alpha
        else:
            # Use default alpha
            self.transfer_knowledge(round_num)
        
        # Evaluate and log round results
        results = self.evaluate_round(round_num)
        
        # Log the completion of this round
        self.logger.log_stage(f"round_{round_num}_completed", results)
        
        return results
    
    def _consider_model_reinitialization(self, probability):
        """
        Consider reinitializing underperforming models based on probability.
        This helps escape local optima.
        
        Args:
            probability: Probability of reinitializing a model (0-1)
        """
        import random
        
        if not hasattr(self, 'post_transfer_metrics') or not self.post_transfer_metrics:
            print("No performance history available for model reinitialization assessment")
            return
        
        # Get the latest metrics
        latest_round = max(self.post_transfer_metrics.keys())
        latest_metrics = self.post_transfer_metrics[latest_round]
        
        # Find models that might benefit from reinitialization
        for client_id, client_data in self.local_models.items():
            model = client_data['model']
            
            # Skip neural network models as they're more expensive to retrain from scratch
            if model.model_name in ['cnn', 'autoencoder']:
                continue
            
            # Get current performance
            current_accuracy = latest_metrics.get(client_id, {}).get('accuracy', 0)
            
            # Compare with global model if available
            global_better = False
            if hasattr(self, 'global_model') and self.global_model and self.global_model.is_fitted:
                X_val = client_data['X_val']
                y_val = client_data['y_val']
                
                global_preds = self.global_model.predict(X_val)
                global_accuracy = accuracy_score(y_val, global_preds)
                
                # If global model is significantly better, consider reinitializing
                if global_accuracy > current_accuracy * 1.05:  # 5% better
                    global_better = True
            
            # Apply random chance based on probability
            if (global_better or current_accuracy < 0.6) and random.random() < probability:
                print(f"Reinitializing underperforming model: {model.model_name}_client_{client_id}")
                
                # Save input/output dimensions
                input_dim = model.input_dim if hasattr(model, 'input_dim') else None
                output_dim = model.output_dim if hasattr(model, 'output_dim') else None
                
                # Create a new instance of the same model class
                model_class = type(model)
                new_model = model_class(client_id=client_id, input_dim=input_dim, output_dim=output_dim)
                
                # Replace the model in our dictionary
                self.local_models[client_id]['model'] = new_model 