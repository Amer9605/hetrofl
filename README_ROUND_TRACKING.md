# Round-by-Round Model Tracking and Testing System

## Overview

The HETROFL system has been enhanced with comprehensive round-by-round model saving, testing, and improvement tracking capabilities. This enhancement allows users to:

- **Automatically save models after each communication round**
- **Test models from specific rounds on custom datasets**
- **Track performance improvements across all rounds**
- **Generate comprehensive improvement analysis and visualizations**
- **Compare model performance between rounds**

## 🚀 Key Features

### 1. Automatic Round-by-Round Model Saving
- Models are automatically saved after each communication round
- Organized directory structure with experiment tracking
- Metadata saved with each model (round number, metrics, timestamps)
- Both local and global models are saved with complete information

### 2. Round-Specific Testing
- Test models from any specific round on custom datasets
- Test all rounds at once for comprehensive analysis
- Support for multiple data formats (CSV, Parquet, JSON)
- Detailed performance metrics for each round

### 3. Improvement Tracking and Analysis
- Track accuracy and F1 score improvements across rounds
- Identify best performing rounds for each model
- Calculate improvement deltas between first and last rounds
- Generate trend analysis with visualizations

### 4. Enhanced GUI Interface
- New `RoundTestingWidget` for interactive round-by-round testing
- Experiment selection and round browsing
- Real-time progress tracking and results display
- Export capabilities for test results and analysis

## 📁 Directory Structure

The enhanced system creates the following organized structure:

```
results/
├── models/
│   └── experiment_YYYY-MM-DD_HH-MM-SS/
│       ├── round_1/
│       │   ├── local_models/
│       │   │   ├── client_0_xgboost/
│       │   │   │   ├── xgboost.joblib
│       │   │   │   ├── metadata.json
│       │   │   │   └── xgboost_history.joblib
│       │   │   ├── client_1_random_forest/
│       │   │   └── ...
│       │   ├── global_model/
│       │   │   ├── global_model.joblib
│       │   │   └── metadata.json
│       │   └── round_summary.json
│       ├── round_2/
│       └── ...
├── test_results/
│   └── experiment_YYYY-MM-DD_HH-MM-SS_test_results.json
├── improvement_plots/
│   └── experiment_YYYY-MM-DD_HH-MM-SS/
│       ├── improvement_trends.png
│       ├── client_0_improvement.png
│       └── ...
└── history/
    ├── round_history.json
    ├── performance_history.json
    └── model_registry.json
```

## 🔧 Usage Examples

### Basic Usage (Automatic Round Saving)

```python
from data.data_loader import DataLoader
from global_model.federated_learning import HeterogeneousFederatedLearning
from local_models.xgboost_model import XGBoostModel
from local_models.random_forest_model import RandomForestModel

# Initialize system
data_loader = DataLoader(dataset_path="your_dataset.csv", target_column="target")
local_model_classes = {
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel
}

fl_system = HeterogeneousFederatedLearning(
    data_loader=data_loader,
    local_model_classes=local_model_classes
)

# Run federated learning (models automatically saved after each round)
global_model = fl_system.run_federated_learning(
    communication_rounds=5,
    hyperparameter_tuning=True
)
```

### Testing Specific Rounds

```python
# Test models from round 3
round_3_results = fl_system.test_round_models(round_num=3)
print(f"Round 3 global model accuracy: {round_3_results['global_model']['accuracy']:.4f}")

# Test all rounds
all_results = fl_system.test_all_rounds(save_results=True)
print(f"Tested {len(all_results['rounds'])} rounds")
```

### Improvement Analysis

```python
# Generate comprehensive improvement analysis
analysis = fl_system.generate_improvement_analysis(save_plots=True)

# Access improvement metrics
global_improvement = analysis['improvement_summary']['global_model']
print(f"Global model accuracy improvement: {global_improvement['accuracy_improvement']:.4f}")
print(f"Best accuracy achieved in round: {global_improvement['best_accuracy_round']}")
```

### Using ModelTracker Directly

```python
from utils.model_persistence import ModelTracker

tracker = ModelTracker()

# List all experiments
experiments = tracker.list_experiments()
print(f"Available experiments: {experiments}")

# Get latest experiment
latest_exp = tracker.get_latest_experiment()

# Test models across rounds with custom data
test_results = tracker.test_models_across_rounds(
    experiment_id=latest_exp,
    test_data=custom_test_data,
    test_labels=custom_test_labels,
    save_results=True
)

# Analyze improvement trends
trends = tracker.analyze_improvement_trends(latest_exp, save_plots=True)
```

## 🖥️ GUI Usage

### Round Testing Widget

The new `RoundTestingWidget` provides an intuitive interface for round-by-round testing:

1. **Select Experiment**: Choose from available experiments
2. **Select Round**: Test specific rounds or all rounds
3. **Load Test Data**: Browse and load custom test datasets
4. **Run Tests**: Execute testing with progress tracking
5. **View Results**: Comprehensive results display with multiple tabs
6. **Generate Analysis**: Create improvement analysis with visualizations
7. **Export Results**: Save results to JSON or CSV files

### Integration with Main GUI

```python
from gui.gui_test_interface import RoundTestingWidget

# Add to your main GUI
round_testing_widget = RoundTestingWidget()
# Add to your tab widget or main layout
```

## 📊 Metrics and Analysis

### Tracked Metrics
- **Accuracy**: Classification accuracy for each model
- **F1 Score**: Weighted F1 score for multi-class problems
- **Improvement Deltas**: Change from first to last round
- **Best Performance**: Highest metrics achieved and in which round
- **Round-by-Round Trends**: Performance progression across rounds

### Visualizations Generated
- **Improvement Trends**: Line plots showing performance across rounds
- **Individual Model Progress**: Separate plots for each client model
- **Comparative Analysis**: Side-by-side comparison of all models
- **Global vs Local Performance**: Comparison between global and local models

## 🔍 Testing and Validation

### Test Script

Run the comprehensive test script to validate functionality:

```bash
python test_round_tracking.py
```

This script will:
- Create a sample dataset
- Run a short federated learning experiment
- Test round-by-round functionality
- Generate improvement analysis
- Validate all new features

### Expected Output

The test script validates:
- ✅ Automatic round-by-round model saving
- ✅ Round-specific directory structure creation
- ✅ Metadata saving with each model
- ✅ Round-specific model testing
- ✅ Cross-round performance analysis
- ✅ Improvement trend visualization
- ✅ Comprehensive experiment tracking

## 🛠️ Technical Implementation

### Enhanced Components

1. **ModelTracker** (`utils/model_persistence.py`)
   - Added round-based tracking capabilities
   - Experiment management with unique IDs
   - Cross-round testing methods
   - Improvement analysis functions

2. **HeterogeneousFederatedLearning** (`global_model/federated_learning.py`)
   - Integrated automatic round saving
   - Added round testing methods
   - Enhanced with improvement analysis
   - Experiment lifecycle management

3. **BaseLocalModel** (`local_models/base_model.py`)
   - Enhanced save method with round information
   - Metadata saving capabilities
   - Round-specific directory organization

4. **RoundTestingWidget** (`gui/gui_test_interface.py`)
   - Complete GUI for round-by-round testing
   - Experiment and round selection
   - Progress tracking and results display
   - Export functionality

### Key Methods Added

- `ModelTracker.start_experiment()`: Initialize experiment tracking
- `ModelTracker.register_round_models()`: Register models for a round
- `ModelTracker.test_models_across_rounds()`: Test all rounds
- `ModelTracker.analyze_improvement_trends()`: Generate improvement analysis
- `HeterogeneousFederatedLearning.save_round_models()`: Save models after each round
- `HeterogeneousFederatedLearning.test_round_models()`: Test specific round
- `HeterogeneousFederatedLearning.generate_improvement_analysis()`: Create analysis

## 🎯 Benefits

### For Researchers
- **Detailed Analysis**: Track model performance evolution across rounds
- **Reproducibility**: Complete experiment tracking with metadata
- **Comparison**: Easy comparison between different rounds and models
- **Visualization**: Comprehensive plots for publication and analysis

### For Practitioners
- **Model Selection**: Identify best performing rounds for deployment
- **Debugging**: Understand when and why performance changes
- **Optimization**: Fine-tune based on round-by-round insights
- **Validation**: Test models on custom datasets at any round

### For System Users
- **User-Friendly**: Intuitive GUI for non-technical users
- **Flexible**: Support for various data formats and testing scenarios
- **Comprehensive**: All-in-one solution for model testing and analysis
- **Export**: Easy export of results for further analysis

## 🔮 Future Enhancements

Potential future improvements:
- **Real-time Monitoring**: Live performance tracking during training
- **Advanced Visualizations**: Interactive plots with drill-down capabilities
- **Model Comparison**: Side-by-side comparison of different experiments
- **Automated Reporting**: Generate PDF reports with analysis
- **Cloud Integration**: Save and share experiments in cloud storage
- **Performance Prediction**: Predict future round performance based on trends

## 📝 Notes

- All round-by-round functionality is enabled by default
- Models are saved automatically without impacting training performance
- The system maintains backward compatibility with existing code
- Test data should be preprocessed similarly to training data for accurate results
- Large experiments may generate significant storage requirements

## 🤝 Contributing

When contributing to the round tracking system:
1. Ensure all new methods include comprehensive docstrings
2. Add appropriate error handling and logging
3. Update tests to cover new functionality
4. Maintain consistency with existing code style
5. Document any new configuration options

---

This enhanced round-by-round tracking system provides comprehensive insights into the federated learning process, enabling better understanding, debugging, and optimization of heterogeneous federated learning systems.