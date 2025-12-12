# Quadratic Equation Predictor - AI Hyper-Optimization

A modern, expandable GUI application for predicting quadratic equation roots using AI, with a focus on hyper-optimization for training time and memory usage.

## Features

- **Page 1: Equation Entry**
  - Enter quadratic equations with LaTeX display
  - Real-time graph visualization
  - Automatic root calculation

- **Page 2: Data Generation & Model Presets**
  - Generate synthetic training data
  - Multiple model presets (Tiny, Small, Medium, Large)
  - Data distribution visualization

- **Page 3: Training**
  - Real-time training metrics
  - Current and best model comparison
  - Color-coded success rates
  - Training loss and accuracy plots
  - Model architecture visualization

- **Page 4: Results**
  - Test model predictions
  - Comprehensive statistics
  - Error analysis and visualization

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

## Project Structure

```
QEPV2/
├── main.py                 # Main application entry point
├── config.py              # Configuration (colors, presets, etc.)
├── requirements.txt       # Python dependencies
├── models/
│   ├── __init__.py
│   └── quadratic_model.py # Neural network model
├── pages/
│   ├── __init__.py
│   ├── base_page.py       # Base page class
│   ├── page1_equation.py  # Equation entry page
│   ├── page2_data_generation.py  # Data generation page
│   ├── page3_training.py  # Training page
│   └── page4_results.py   # Results page
└── utils/
    ├── __init__.py
    ├── quadratic_utils.py # Quadratic equation utilities
    └── data_generator.py  # Data generation utilities
```

## Customization

The application is designed to be easily expandable:

- **Colors & Theme**: Edit `config.py` to change colors, sizes, and styling
- **Model Presets**: Modify `MODEL_PRESETS` in `config.py`
- **Pages**: Each page is a separate module in `pages/` directory
- **Utilities**: Add new utility functions in `utils/` directory

## Requirements

- Python 3.8+
- PyQt6
- PyTorch
- NumPy
- Matplotlib
- SymPy
- scikit-learn
- Optuna (for future hyperparameter optimization)

## License

MIT License
