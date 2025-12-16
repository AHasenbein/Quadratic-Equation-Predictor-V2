# Hyperparameter Optimization Research: Quadratic Root Prediction

A research-focused project for systematic hyperparameter optimization of neural networks for quadratic root prediction. This project provides practical insights for optimizing small ML models on mathematical function approximation tasks.

## Research Question

**"What are the optimal hyperparameters for small neural networks approximating mathematical functions, and what practical insights can we derive for similar problems?"**

## Features

- **Bayesian Optimization (Optuna)**: Intelligent search using TPE algorithm
- **Random Search**: Baseline comparison method
- **Multi-objective Optimization**: Pareto frontier for accuracy vs efficiency trade-offs
- **Stratified Data Generation**: Balanced datasets by root type (real, complex, single)
- **Comprehensive Evaluation**: MAE, RMSE, R², accuracy, error analysis by root type
- **Model Efficiency Metrics**: Parameter count, model size, inference speed
- **Reproducible Experiments**: Fixed random seeds for reproducibility

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick optimization (20 trials, ~5 minutes)
python main.py --preset quick

# Run full optimization (100 trials, ~30-60 minutes)
python main.py --preset full

# Custom experiment
python main.py --preset quick --trials 50 --epochs 100 --samples 20000
```

## Available Presets

- `quick` - Fast test (20 trials, 50 epochs, 5000 samples)
- `full` - Comprehensive search (100 trials, 100 epochs, 20000 samples)
- `architecture_study` - Focus on architecture hyperparameters
- `training_study` - Focus on training hyperparameters
- `random_search` - Random search baseline
- `multi_objective` - Pareto frontier optimization (loss vs model size)

## Project Structure

```
QEPV2/
├── main.py                    # CLI entry point
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── models/
│   ├── __init__.py
│   └── model.py               # Neural network model
├── utils/
│   ├── __init__.py
│   ├── data.py                # Data generation (uniform, stratified)
│   ├── evaluation.py          # Evaluation metrics
│   ├── optimizer.py           # Optimization methods (Optuna, random, multi-objective)
│   └── runner.py              # Experiment runner
└── config/
    ├── __init__.py
    └── search_spaces.py       # Search spaces and presets
```

## Usage Examples

### Basic Optimization

```bash
# Quick test
python main.py --preset quick

# Full study
python main.py --preset full --device cuda  # Use GPU if available
```

### Custom Configuration

```bash
# Override preset values
python main.py --preset quick --trials 50 --epochs 150

# Use different data strategy
python main.py --preset full --data-strategy uniform

# Multi-objective optimization
python main.py --preset multi_objective
```

## Results

Results are saved as JSON files in the `results/` directory. Each file contains:

- Best hyperparameters found
- Best loss/objective value
- All trial results (for random search)
- Pareto front solutions (for multi-objective)
- Experiment metadata (trials, epochs, seed, etc.)

## Requirements

- Python 3.8+
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- Optuna >= 3.4.0
- scikit-learn >= 1.3.0
- Matplotlib >= 3.8.0
- pandas >= 2.1.0

## License

MIT
