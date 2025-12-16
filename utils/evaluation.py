"""
Evaluation metrics for model performance.
"""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data import solve_quadratic


def evaluate(model, data_loader, device='cpu', tolerance=0.1):
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_preds, all_targets, all_inputs = [], [], []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())
    
    preds, targets, inputs_array = np.vstack(all_preds), np.vstack(all_targets), np.vstack(all_inputs)
    
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    diff = np.abs(preds - targets)
    accuracy = (np.sum(diff < tolerance) / diff.size) * 100
    
    error_analysis = analyze_errors_by_root_type(preds, targets, inputs_array, tolerance)
    
    return {'loss': rmse**2, 'mae': mae, 'rmse': rmse, 'r2': r2, 'accuracy': accuracy, 'error_analysis': error_analysis}


def analyze_errors_by_root_type(predictions, targets, inputs, tolerance=0.1):
    """Analyze errors by root type."""
    results = {}
    real_indices, complex_indices, single_indices = [], [], []
    
    for i, (a, b, c) in enumerate(inputs):
        _, _, disc_type = solve_quadratic(a, b, c)
        if disc_type == "Two real roots":
            real_indices.append(i)
        elif disc_type == "One real root":
            single_indices.append(i)
        else:
            complex_indices.append(i)
    
    for name, indices in [('real_roots', real_indices), ('complex_roots', complex_indices), ('single_root', single_indices)]:
        if len(indices) == 0:
            results[name] = {'count': 0, 'mae': 0.0, 'rmse': 0.0, 'accuracy': 0.0}
            continue
        
        pred_subset, target_subset = predictions[indices], targets[indices]
        mae = mean_absolute_error(target_subset, pred_subset)
        rmse = np.sqrt(mean_squared_error(target_subset, pred_subset))
        diff = np.abs(pred_subset - target_subset)
        accuracy = (np.sum(diff < tolerance) / diff.size) * 100
        results[name] = {'count': len(indices), 'mae': mae, 'rmse': rmse, 'accuracy': accuracy}
    
    return results


def get_model_efficiency_metrics(model, sample_input, device='cpu'):
    """Calculate model efficiency metrics."""
    import time
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    model.eval()
    model.to(device)
    sample_input = sample_input.to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    
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
    
    return {'param_count': param_count, 'model_size_mb': total_size_mb, 'inference_time_ms': avg_inference_time_ms,
            'inference_speed_samples_per_sec': 1000 / avg_inference_time_ms if avg_inference_time_ms > 0 else 0}
