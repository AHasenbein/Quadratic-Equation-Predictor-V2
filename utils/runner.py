"""
Experiment runner for orchestrating optimization experiments with callbacks.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils.optimizer import OptunaOptimizer, RandomSearchOptimizer, MultiObjectiveOptimizer
from utils.data import generate_data, generate_stratified_data
from typing import Optional, Callable


class ExperimentRunner:
    """Runs optimization experiments with callback support."""
    
    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_data_loaders(self, data, batch_size=64, val_split=0.2):
        """Create train and validation data loaders."""
        inputs, targets = data['inputs'], data['targets']
        split_idx = int(len(inputs) * (1 - val_split))
        
        train_dataset = TensorDataset(torch.FloatTensor(inputs[:split_idx]), torch.FloatTensor(targets[:split_idx]))
        val_dataset = TensorDataset(torch.FloatTensor(inputs[split_idx:]), torch.FloatTensor(targets[split_idx:]))
        
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    def run_optimization(self, train_data, search_space, optimization_method='optuna', n_trials=100,
                        epochs_per_trial=50, batch_size=64, device='cpu', callback=None):
        """
        Run hyperparameter optimization with optional callback.
        
        Args:
            callback: Function(trial_num, loss, params) called after each trial
        """
        train_loader, val_loader = self._create_data_loaders(train_data, batch_size)
        
        if optimization_method == 'optuna':
            optimizer = OptunaOptimizer(train_loader, val_loader, device=device, 
                                       epochs=epochs_per_trial, callback=callback)
        elif optimization_method == 'random':
            optimizer = RandomSearchOptimizer(train_loader, val_loader, device=device,
                                           epochs=epochs_per_trial, callback=callback)
        elif optimization_method == 'multi_objective':
            optimizer = MultiObjectiveOptimizer(train_loader, val_loader, device=device,
                                              epochs=epochs_per_trial, callback=callback)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        results = optimizer.optimize(n_trials=n_trials, search_space=search_space)
        results['optimization_method'] = optimization_method
        results['n_trials'] = n_trials
        results['epochs_per_trial'] = epochs_per_trial
        results['seed'] = self.seed
        return results
    
    def save_results(self, results, filepath):
        """Save results to JSON file."""
        import json
        
        serializable_results = {}
        for key, value in results.items():
            if key == 'study':
                continue
            elif isinstance(value, (int, float, str, bool, type(None))):
                serializable_results[key] = value
            elif isinstance(value, dict):
                serializable_results[key] = self._make_serializable(value)
            elif isinstance(value, list):
                serializable_results[key] = [self._make_serializable(item) if isinstance(item, (dict, list)) else item for item in value]
            else:
                serializable_results[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_serializable(self, obj):
        """Recursively make object JSON-serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
