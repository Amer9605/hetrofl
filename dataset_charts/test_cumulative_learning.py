#!/usr/bin/env python3
"""
Test script for the cumulative learning system.
"""

import os
import time
import argparse
from datetime import datetime

from config.config import CLIENT_MODELS
from data.data_loader import DataLoader
from local_models.xgboost_model import XGBoostModel
from local_models.random_forest_model import RandomForestModel
from local_models.lightgbm_model import LightGBMModel
from local_models.cnn_model import CNNModel
from local_models.autoencoder_model import AutoencoderModel
from global_model.federated_learning import HeterogeneousFederatedLearning
from utils.model_persistence import ModelTracker
from utils.analyze_learning import LearningAnalyzer


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test the cumulative learning system")
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run"
    )
    
    parser.add_argument(
        "--communication_rounds",
        type=int,
        default=2,
        help="Number of communication rounds per iteration"
    )
    
    parser.add_argument(
        "--data_distribution",
        type=str,
        default="iid",
        choices=["iid", "non_iid_label_skew", "non_iid_feature_skew"],
        help="Data distribution type"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis after all iterations"
    )
    
    return parser.parse_args()


def run_iteration(iteration, args):
    """
    Run a single iteration of the federated learning process.
    
    Args:
        iteration: Iteration number
        args: Command line arguments
        
    Returns:
        Tuple of (experiment_name, duration)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"test_cumulative_learning_iter{iteration}_{timestamp}"
    
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}/{args.iterations}")
    print(f"{'='*80}")
    
    print(f"Running experiment: {experiment_name}")
    print(f"Data distribution: {args.data_distribution}")
    print(f"Communication rounds: {args.communication_rounds}")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Initialize local model classes
    local_model_classes = {
        "xgboost": XGBoostModel,
        "random_forest": RandomForestModel,
        "lightgbm": LightGBMModel,
        "cnn": CNNModel,
        "autoencoder": AutoencoderModel
    }
    
    # Initialize federated learning system
    fl_system = HeterogeneousFederatedLearning(
        data_loader=data_loader,
        local_model_classes=local_model_classes,
        experiment_name=experiment_name
    )
    
    # Start timer
    start_time = time.time()
    
    # Run federated learning
    global_model = fl_system.run_federated_learning(
        communication_rounds=args.communication_rounds,
        hyperparameter_tuning=(iteration == 1),  # Only tune hyperparameters in first iteration
        data_distribution=args.data_distribution,
        load_previous_models=True  # Always enable cumulative learning
    )
    
    # End timer
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nIteration {iteration} completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Results saved to: {fl_system.logger.experiment_dir}")
    
    return experiment_name, duration


def main():
    """
    Main function to test the cumulative learning system.
    """
    # Parse arguments
    args = parse_args()
    
    print(f"Starting cumulative learning test with {args.iterations} iterations")
    
    # Run iterations
    total_start_time = time.time()
    experiments = []
    
    for i in range(1, args.iterations + 1):
        experiment_name, duration = run_iteration(i, args)
        experiments.append((experiment_name, duration))
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print("CUMULATIVE LEARNING TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Iterations: {args.iterations}")
    print(f"Communication rounds per iteration: {args.communication_rounds}")
    print(f"Data distribution: {args.data_distribution}")
    
    print("\nExperiments:")
    for i, (name, duration) in enumerate(experiments, 1):
        print(f"  Iteration {i}: {name} - {duration:.2f} seconds")
    
    # Run analysis if requested
    if args.analyze:
        print("\nRunning cumulative learning analysis...")
        analyzer = LearningAnalyzer()
        results = analyzer.run_full_analysis()
        
        print(f"\nAnalysis completed. Results saved to: {analyzer.output_dir}")
    
    print("\nTest completed successfully.")


if __name__ == "__main__":
    main() 