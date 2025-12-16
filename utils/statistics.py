"""
Statistical analysis utilities for model evaluation.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import optuna


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a dataset.
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95)
    
    Returns:
        (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)
    
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return (mean - h, mean + h)


def compare_models_statistically(metrics1: Dict, metrics2: Dict, n_samples: int) -> Dict:
    """
    Compare two models statistically.
    
    Args:
        metrics1: Metrics from first model
        metrics2: Metrics from second model
        n_samples: Number of samples used
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    for metric in ['loss', 'mae', 'rmse']:
        if metric in metrics1 and metric in metrics2:
            diff = metrics1[metric] - metrics2[metric]
            improvement_pct = (diff / metrics1[metric]) * 100 if metrics1[metric] != 0 else 0
            comparison[metric] = {
                'difference': diff,
                'improvement_pct': improvement_pct,
                'better': 'model2' if diff > 0 else 'model1'
            }
    
    for metric in ['r2', 'accuracy']:
        if metric in metrics1 and metric in metrics2:
            diff = metrics2[metric] - metrics1[metric]
            improvement_pct = (diff / metrics1[metric]) * 100 if metrics1[metric] != 0 else 0
            comparison[metric] = {
                'difference': diff,
                'improvement_pct': improvement_pct,
                'better': 'model2' if diff > 0 else 'model1'
            }
    
    return comparison


def analyze_hyperparameter_sensitivity(study, param_name: str) -> Dict:
    """
    Analyze sensitivity of a hyperparameter.
    
    Args:
        study: Optuna study
        param_name: Name of parameter to analyze
    
    Returns:
        Dictionary with sensitivity metrics
    """
    param_values = []
    losses = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and param_name in trial.params:
            param_values.append(trial.params[param_name])
            losses.append(trial.value)
    
    if not param_values:
        return {'sensitivity': 0, 'correlation': 0}
    
    # Calculate correlation
    correlation = np.corrcoef(param_values, losses)[0, 1] if len(param_values) > 1 else 0
    
    # Calculate sensitivity (how much loss changes per unit change in param)
    if len(param_values) > 1:
        param_range = max(param_values) - min(param_values)
        loss_range = max(losses) - min(losses)
        sensitivity = loss_range / param_range if param_range > 0 else 0
    else:
        sensitivity = 0
    
    return {
        'sensitivity': sensitivity,
        'correlation': correlation,
        'param_range': (min(param_values), max(param_values)) if param_values else (0, 0),
        'loss_range': (min(losses), max(losses)) if losses else (0, 0)
    }
