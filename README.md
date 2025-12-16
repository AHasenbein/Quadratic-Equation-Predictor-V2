# Quadratic Equation Predictor - Hyperparameter Optimization Research

A research-focused application for systematic hyperparameter optimization of neural networks for quadratic root prediction. This project provides practical insights for optimizing small ML models on mathematical function approximation tasks.

## Research Focus

**Main Research Question**: *"What are the optimal hyperparameters for small neural networks approximating mathematical functions, and what practical insights can we derive for similar problems?"*

This project provides:
- **Systematic Hyperparameter Optimization**: Bayesian optimization (Optuna), grid search, and random search
- **Architecture Analysis**: Compare different neural network configurations
- **Efficiency Trade-offs**: Analyze accuracy vs model size vs speed
- **Practical Guidelines**: Actionable insights for optimizing small ML models

See [RESEARCH.md](RESEARCH.md) for detailed research documentation.

## Features

### Core Functionality
- **Page 1: Equation Entry**
  - Enter quadratic equations with LaTeX display
  - Real-time graph visualization
  - Automatic root calculation

- **Page 2: Data Generation & Model Presets**
  - Generate synthetic training data (uniform, stratified, edge cases)
  - Data quality analysis and visualization
  - Multiple model presets

- **Page 3: Training & Optimization**
  - Real-time training metrics
  - **Hyperparameter Optimization**: Run Optuna-based optimization studies
  - Best models tracker with hyperparameter importance
  - Training loss and accuracy plots
  - Model architecture visualization

- **Page 4: Results & Analysis**
  - Optimization results dashboard
  - Pareto frontier visualization (multi-objective optimization)
  - Architecture comparison
  - Trade-off analysis (accuracy vs size vs speed)
  - Practical insights and recommendations
  - Comprehensive error analysis by root type

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

## Quick Start: Running Optimization

1. **Generate Data** (Page 2):
   - Set coefficient ranges
   - Choose data generation strategy (uniform, stratified, edge cases)
   - Generate training dataset

2. **Run Optimization** (Page 3):
   - Select optimization preset (quick, full, architecture study, etc.)
   - Choose optimization method (Optuna, grid search, random search)
   - Set number of trials and epochs
   - Start optimization

3. **Analyze Results** (Page 4):
   - View optimization results table
   - Examine hyperparameter importance
   - Analyze Pareto frontiers (for multi-objective)
   - Review trade-off analysis
   - Read practical insights

## Project Structure

```
QEPV2/
├── main.py                 # Main application entry point
├── config.py              # Configuration (colors, presets, etc.)
├── config/
│   └── optimization_config.py  # Optimization search spaces and presets
├── RESEARCH.md            # Research documentation
├── requirements.txt       # Python dependencies
├── models/
│   ├── __init__.py
│   └── quadratic_model.py # Enhanced neural network model with callbacks
├── pages/
│   ├── __init__.py
│   ├── base_page.py       # Base page class
│   ├── page1_equation.py  # Equation entry page
│   ├── page2_data_generation.py  # Data generation page
│   ├── page3_training.py  # Training & optimization page
│   └── page4_results.py   # Results & analysis page
└── utils/
    ├── __init__.py
    ├── quadratic_utils.py # Quadratic equation utilities
    ├── data_generator.py  # Enhanced data generation (stratified, edge cases)
    ├── evaluation.py      # Comprehensive evaluation framework
    ├── experiment_runner.py  # Experiment execution and tracking
    ├── hyperparameter_optimization.py  # Optuna, grid, random search
    └── visualization.py   # Advanced plotting for optimization results
```

## Key Research Contributions

1. **Systematic Optimization Framework**: Comprehensive hyperparameter search with multiple strategies
2. **Architecture Insights**: Identification of optimal architectures for mathematical function approximation
3. **Efficiency Analysis**: Trade-off curves between accuracy, model size, and inference speed
4. **Practical Guidelines**: Actionable recommendations for optimizing small ML models

## Results Interpretation

### Optimization Results
- **Best Parameters**: Optimal hyperparameter configuration found
- **Hyperparameter Importance**: Which parameters matter most for performance
- **Convergence Plots**: How optimization improved over trials
- **Pareto Frontiers**: Best models for different objectives (accuracy vs efficiency)

### Practical Insights
The system automatically generates insights such as:
- "For 95% accuracy, use architecture X with learning rate Y"
- "Model size can be reduced by 60% with only 2% accuracy loss"
- "ReLU outperforms Tanh for this mathematical function"

## Customization

- **Search Spaces**: Modify `config/optimization_config.py` to define custom search spaces
- **Experiment Presets**: Add new presets for different optimization studies
- **Evaluation Metrics**: Configure metric priorities in optimization config
- **Colors & Theme**: Edit `config.py` for UI customization

## Requirements

- Python 3.8+
- PyQt6 >= 6.6.0
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- Matplotlib >= 3.8.0
- SymPy >= 1.12
- scikit-learn >= 1.3.0
- Optuna >= 3.4.0 (for hyperparameter optimization)
- pandas >= 2.1.0

## License

MIT License
