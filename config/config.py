"""
Configuration settings for the HETROFL system.
"""

# Dataset configuration
DATASET_PATH = "CIC-ToN-IoT-V2.parquet/NF-ToN-IoT-v3-train.csv"
TARGET_COLUMN = "Attack"  # Updated to match the actual column name in the dataset
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42
DATASET_SAMPLE_SIZE = 250000  # Default sample size, -1 for using all data

# Client configuration
NUM_CLIENTS = 5
CLIENT_MODELS = ["xgboost", "random_forest", "lightgbm", "cnn", "autoencoder"]

# Training configuration
COMMUNICATION_ROUNDS = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.001
GLOBAL_EPOCHS = 10

# SMOTE configuration
SMOTE_SAMPLING_STRATEGY = "auto"
SMOTE_K_NEIGHBORS = 5

# Hyperparameter tuning
N_TRIALS = 20
TIMEOUT = 3600  # seconds

# Paths for saving results
RESULTS_DIR = "results"
MODEL_SAVE_DIR = "results/models"
PLOT_SAVE_DIR = "results/plots"
LOG_DIR = "results/logs"

# Data distribution types
DATA_DISTRIBUTIONS = ["iid", "non_iid_label_skew", "non_iid_feature_skew"]

# Federated learning parameters
FL_AGGREGATION_METHOD = "knowledge_distillation"
BIDIRECTIONAL_LEARNING = True  # Enable global to local knowledge transfer

# Modified knowledge transfer parameters for improved performance
GLOBAL_TO_LOCAL_ALPHA = 0.25  # Initial weight for global knowledge (reduced from 0.3)
ALPHA_DECAY_RATE = 0.9  # Alpha decay rate per round
ALPHA_MIN_VALUE = 0.1  # Minimum alpha value

# Learning rate schedules for local models
LEARNING_RATE_SCHEDULE = {
    "xgboost": [0.1, 0.08, 0.05, 0.03, 0.01],
    "random_forest": [0.1, 0.1, 0.1, 0.1, 0.1],  # RF less affected by LR
    "lightgbm": [0.1, 0.08, 0.05, 0.03, 0.01],
    "cnn": [0.001, 0.0008, 0.0005, 0.0003, 0.0001],
    "autoencoder": [0.001, 0.0008, 0.0005, 0.0003, 0.0001]
}

# Global model configuration
GLOBAL_MODEL_HIDDEN_LAYERS = [128, 64, 32]
KL_TEMPERATURE = 2.0
DISTILLATION_ALPHA = 0.5  # Weight between soft and hard targets

# Round-specific model complexity (increases with rounds)
MODEL_COMPLEXITY_SCHEDULE = {
    "xgboost": {"n_estimators": [100, 120, 150, 180, 200], "max_depth": [3, 4, 5, 6, 6]},
    "random_forest": {"n_estimators": [100, 120, 150, 180, 200], "max_depth": [5, 6, 7, 8, 10]},
    "lightgbm": {"n_estimators": [100, 120, 150, 180, 200], "num_leaves": [31, 35, 40, 45, 50]}
}

# Model re-initialization probability per round (to escape local minima)
REINIT_PROBABILITY = 0.05  # 5% chance to reinitialize a model

# Cumulative learning configuration
CUMULATIVE_LEARNING = True  # Enable cumulative learning across runs
LOAD_BEST_MODELS = False  # If True, load best models instead of latest models
PERFORMANCE_HISTORY_WINDOW = 10  # Number of past runs to track in performance history
VISUALIZATION_ENABLED = True  # Enable comprehensive visualizations

# Enhanced knowledge retention settings
KNOWLEDGE_RETENTION_FACTOR = 0.85  # Weight for retaining knowledge from previous runs (increased from 0.8)
MODEL_ENSEMBLE_THRESHOLD = 0.7  # Threshold for model ensemble decisions 