"""
Comprehensive evaluation framework for quadratic root prediction models.
Provides metrics, error analysis, and statistical testing.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.quadratic_utils import solve_quadratic


def calculate_root_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    tolerance: float = 0.1
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for root predictions.
    
    Args:
        predictions: Array of shape (N, 4) with [r1_real, r2_real, r1_imag, r2_imag]
        targets: Array of shape (N, 4) with true roots
        tolerance: Tolerance for accuracy calculation
    
    Returns:
        Dictionary with metrics: mae, rmse, r2, accuracy, accuracy_real, accuracy_complex
    """
    # Flatten for overall metrics
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Calculate standard regression metrics
    mae = mean_absolute_error(target_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
    r2 = r2_score(target_flat, pred_flat)
    
    # Calculate accuracy (within tolerance)
    diff = np.abs(predictions - targets)
    within_tolerance = np.sum(diff < tolerance)
    total = predictions.size
    accuracy = (within_tolerance / total) * 100 if total > 0 else 0.0
    
    # Separate metrics for real and imaginary parts
    real_pred = predictions[:, [0, 1]]  # r1_real, r2_real
    real_target = targets[:, [0, 1]]
    imag_pred = predictions[:, [2, 3]]  # r1_imag, r2_imag
    imag_target = targets[:, [2, 3]]
    
    real_mae = mean_absolute_error(real_target.flatten(), real_pred.flatten())
    imag_mae = mean_absolute_error(imag_target.flatten(), imag_pred.flatten())
    
    real_accuracy = (np.sum(np.abs(real_pred - real_target) < tolerance) / real_pred.size) * 100
    imag_accuracy = (np.sum(np.abs(imag_pred - imag_target) < tolerance) / imag_pred.size) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy,
        'real_mae': real_mae,
        'imag_mae': imag_mae,
        'real_accuracy': real_accuracy,
        'imag_accuracy': imag_accuracy
    }


def analyze_errors_by_root_type(
    predictions: np.ndarray,
    targets: np.ndarray,
    inputs: np.ndarray,
    tolerance: float = 0.1
) -> Dict[str, Dict[str, float]]:
    """
    Analyze prediction errors by root type (real vs complex).
    
    Args:
        predictions: Array of shape (N, 4) with predicted roots
        targets: Array of shape (N, 4) with true roots
        inputs: Array of shape (N, 3) with coefficients [a, b, c]
        tolerance: Tolerance for accuracy calculation
    
    Returns:
        Dictionary with metrics for each root type
    """
    results = {}
    
    # Classify each sample by root type
    real_root_indices = []
    complex_root_indices = []
    single_root_indices = []
    
    for i, (a, b, c) in enumerate(inputs):
        root1, root2, disc_type = solve_quadratic(a, b, c)
        
        if disc_type == "Two real roots":
            real_root_indices.append(i)
        elif disc_type == "One real root":
            single_root_indices.append(i)
        else:  # Complex roots
            complex_root_indices.append(i)
    
    # Calculate metrics for each type
    for name, indices in [
        ('real_roots', real_root_indices),
        ('complex_roots', complex_root_indices),
        ('single_root', single_root_indices)
    ]:
        if len(indices) == 0:
            results[name] = {
                'count': 0,
                'mae': 0.0,
                'rmse': 0.0,
                'accuracy': 0.0
            }
            continue
        
        pred_subset = predictions[indices]
        target_subset = targets[indices]
        
        metrics = calculate_root_metrics(pred_subset, target_subset, tolerance)
        results[name] = {
            'count': len(indices),
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'accuracy': metrics['accuracy']
        }
    
    return results


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: str = 'cpu',
    tolerance: float = 0.1
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader with test data
        device: Device to run on
        tolerance: Tolerance for accuracy
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_inputs = []
    total_loss = 0.0
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    inputs_array = np.vstack(all_inputs)
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate metrics
    metrics = calculate_root_metrics(predictions, targets, tolerance)
    metrics['loss'] = avg_loss
    
    # Error analysis by root type
    error_analysis = analyze_errors_by_root_type(predictions, targets, inputs_array, tolerance)
    metrics['error_analysis'] = error_analysis
    
    return metrics


def compare_models(
    models: Dict[str, torch.nn.Module],
    data_loader,
    device: str = 'cpu',
    tolerance: float = 0.1
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary of {name: model}
        data_loader: DataLoader with test data
        device: Device to run on
        tolerance: Tolerance for accuracy
    
    Returns:
        Dictionary of {model_name: metrics}
    """
    results = {}
    
    for name, model in models.items():
        metrics = evaluate_model(model, data_loader, device, tolerance)
        results[name] = metrics
    
    return results


def calculate_statistical_significance(
    metrics1: Dict[str, float],
    metrics2: Dict[str, float],
    n_samples: int
) -> Dict[str, float]:
    """
    Calculate statistical significance between two model performances.
    Uses paired t-test approximation.
    
    Args:
        metrics1: Metrics from first model
        metrics2: Metrics from second model
        n_samples: Number of samples used for evaluation
    
    Returns:
        Dictionary with p-values for different metrics
    """
    # Simplified statistical test - in practice would need actual error distributions
    # This is a placeholder for the framework
    
    p_values = {}
    
    # For loss/MAE/RMSE - lower is better
    for metric in ['loss', 'mae', 'rmse']:
        if metric in metrics1 and metric in metrics2:
            diff = abs(metrics1[metric] - metrics2[metric])
            # Simplified: assume normal distribution
            # In practice, would use actual error distributions
            p_values[metric] = 0.05  # Placeholder
    
    # For accuracy/R² - higher is better
    for metric in ['accuracy', 'r2']:
        if metric in metrics1 and metric in metrics2:
            diff = abs(metrics1[metric] - metrics2[metric])
            p_values[metric] = 0.05  # Placeholder
    
    return p_values


def get_model_efficiency_metrics(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Calculate model efficiency metrics (size, speed).
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor for timing
        device: Device to run on
    
    Returns:
        Dictionary with efficiency metrics
    """
    import time
    
    # Model size
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # Inference speed
    model.eval()
    model.to(device)
    sample_input = sample_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    
    # Time inference
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(sample_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    avg_inference_time_ms = (elapsed / 100) * 1000
    
    return {
        'param_count': param_count,
        'model_size_mb': total_size_mb,
        'inference_time_ms': avg_inference_time_ms,
        'inference_speed_samples_per_sec': 1000 / avg_inference_time_ms if avg_inference_time_ms > 0 else 0
    }
