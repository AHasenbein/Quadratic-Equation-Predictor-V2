"""
Experiment runner for systematic hyperparameter optimization experiments.
Handles experiment execution, metrics tracking, reproducibility, and results saving.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import json
import os
from datetime import datetime
import time
from utils.hyperparameter_optimization import (
    OptunaOptimizer, GridSearchOptimizer, RandomSearchOptimizer, MultiObjectiveOptimizer
)
from utils.evaluation import evaluate_model, get_model_efficiency_metrics
from models.quadratic_model import QuadraticPredictor, ModelTrainer


class ExperimentRunner:
    """Runs systematic experiments with different hyperparameters."""
    
    def __init__(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        test_data: Optional[Dict[str, np.ndarray]] = None,
        device: str = 'cpu',
        seed: int = 42
    ):
        """
        Initialize experiment runner.
        
        Args:
            train_data: Dictionary with 'inputs' and 'targets' arrays
            val_data: Validation data dictionary
            test_data: Optional test data dictionary
            device: Device to run on
            seed: Random seed for reproducibility
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.device = device
        self.seed = seed
        
        # Set seeds for reproducibility
        self._set_seeds(seed)
        
        # Results storage
        self.experiment_results = []
        self.best_model = None
        self.best_metrics = None
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_data_loaders(
        self,
        batch_size: int,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training and validation."""
        train_dataset = TensorDataset(
            torch.FloatTensor(self.train_data['inputs']),
            torch.FloatTensor(self.train_data['targets'])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.val_data['inputs']),
            torch.FloatTensor(self.val_data['targets'])
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def _create_model(
        self,
        hidden_layers: List[int],
        activation: str = 'relu',
        dropout: float = 0.1
    ) -> QuadraticPredictor:
        """Create model with specified architecture."""
        # Note: QuadraticPredictor will need to be enhanced to support dropout
        # For now, using basic version
        model = QuadraticPredictor(hidden_layers=hidden_layers, activation=activation)
        return model
    
    def _train_model(
        self,
        model: QuadraticPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        epochs: int,
        optimizer_type: str = 'adam',
        early_stopping: bool = True,
        patience: int = 10
    ) -> Dict[str, Any]:
        """Train a model and return training history."""
        trainer = ModelTrainer(model, self.device)
        trainer.setup_optimizer(learning_rate=learning_rate, optimizer_type=optimizer_type)
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch': []
        }
        
        for epoch in range(epochs):
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)
            
            training_history['epoch'].append(epoch + 1)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['train_acc'].append(train_metrics['accuracy'])
            training_history['val_acc'].append(val_metrics['accuracy'])
            
            # Early stopping
            if early_stopping:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        return training_history
    
    def run_single_experiment(
        self,
        hyperparameters: Dict[str, Any],
        epochs: int = 100
    ) -> Dict[str, Any]:
        """
        Run a single experiment with given hyperparameters.
        
        Args:
            hyperparameters: Dictionary with hyperparameters
            epochs: Number of training epochs
        
        Returns:
            Dictionary with experiment results
        """
        start_time = time.time()
        
        # Extract hyperparameters
        hidden_layers = hyperparameters.get('hidden_layers', [16, 8])
        activation = hyperparameters.get('activation', 'relu')
        learning_rate = hyperparameters.get('learning_rate', 0.001)
        batch_size = hyperparameters.get('batch_size', 64)
        optimizer_type = hyperparameters.get('optimizer', 'adam')
        dropout = hyperparameters.get('dropout', 0.1)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(batch_size)
        
        # Create model
        model = self._create_model(hidden_layers, activation, dropout)
        model.to(self.device)
        
        # Train model
        training_history = self._train_model(
            model, train_loader, val_loader,
            learning_rate, epochs, optimizer_type
        )
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader, self.device)
        
        # Evaluate on test set if available
        test_metrics = None
        if self.test_data is not None:
            test_dataset = TensorDataset(
                torch.FloatTensor(self.test_data['inputs']),
                torch.FloatTensor(self.test_data['targets'])
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_metrics = evaluate_model(model, test_loader, self.device)
        
        # Get efficiency metrics
        sample_input = torch.FloatTensor([[1.0, 2.0, 1.0]])  # Sample quadratic
        efficiency = get_model_efficiency_metrics(model, sample_input, self.device)
        
        training_time = time.time() - start_time
        
        # Compile results
        results = {
            'hyperparameters': hyperparameters,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'efficiency': efficiency,
            'training_history': training_history,
            'training_time': training_time,
            'epochs_trained': len(training_history['epoch']),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store model if it's the best so far
        if self.best_metrics is None or val_metrics['loss'] < self.best_metrics['loss']:
            self.best_model = model
            self.best_metrics = val_metrics
        
        self.experiment_results.append(results)
        
        return results
    
    def run_optimization(
        self,
        search_space: Dict,
        optimization_method: str = 'optuna',
        n_trials: int = 100,
        epochs_per_trial: int = 100,
        objective_metric: str = 'loss'
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            search_space: Dictionary defining search space
            optimization_method: 'optuna', 'grid', or 'random'
            n_trials: Number of optimization trials
            epochs_per_trial: Epochs to train per trial
            objective_metric: Metric to optimize ('loss', 'accuracy', etc.)
        
        Returns:
            Dictionary with optimization results
        """
        def objective(params: Dict) -> float:
            """Objective function for optimization."""
            try:
                results = self.run_single_experiment(params, epochs=epochs_per_trial)
                
                # Return the metric to optimize
                if objective_metric == 'loss':
                    return results['validation_metrics']['loss']
                elif objective_metric == 'accuracy':
                    return -results['validation_metrics']['accuracy']  # Negative for minimization
                elif objective_metric == 'rmse':
                    return results['validation_metrics']['rmse']
                else:
                    return results['validation_metrics'].get(objective_metric, float('inf'))
            except Exception as e:
                print(f"Error in experiment: {e}")
                return float('inf')
        
        # Choose optimization method
        if optimization_method == 'optuna':
            optimizer = OptunaOptimizer(objective, search_space)
        elif optimization_method == 'grid':
            optimizer = GridSearchOptimizer(objective, search_space)
        elif optimization_method == 'random':
            optimizer = RandomSearchOptimizer(objective, search_space, seed=self.seed)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Run optimization
        optimization_results = optimizer.optimize(n_trials=n_trials)
        
        # Get additional information
        if hasattr(optimizer, 'get_trial_history'):
            optimization_results['trial_history'] = optimizer.get_trial_history()
        
        if hasattr(optimizer, 'get_hyperparameter_importance'):
            optimization_results['hyperparameter_importance'] = optimizer.get_hyperparameter_importance()
        
        optimization_results['experiment_results'] = self.experiment_results
        optimization_results['best_model_metrics'] = self.best_metrics
        
        return optimization_results
    
    def run_multi_objective_optimization(
        self,
        search_space: Dict,
        objectives: List[str],
        n_trials: int = 100,
        epochs_per_trial: int = 100
    ) -> Dict[str, Any]:
        """
        Run multi-objective optimization.
        
        Args:
            search_space: Dictionary defining search space
            objectives: List of objective names (e.g., ['loss', 'model_size_mb'])
            n_trials: Number of optimization trials
            epochs_per_trial: Epochs to train per trial
        
        Returns:
            Dictionary with Pareto frontier results
        """
        def objective(params: Dict) -> Dict[str, float]:
            """Multi-objective function."""
            try:
                results = self.run_single_experiment(params, epochs=epochs_per_trial)
                
                # Extract objective values
                obj_values = {}
                for obj_name in objectives:
                    if obj_name in results['validation_metrics']:
                        obj_values[obj_name] = results['validation_metrics'][obj_name]
                    elif obj_name in results['efficiency']:
                        obj_values[obj_name] = results['efficiency'][obj_name]
                    else:
                        obj_values[obj_name] = float('inf')
                
                return obj_values
            except Exception as e:
                print(f"Error in experiment: {e}")
                return {obj: float('inf') for obj in objectives}
        
        optimizer = MultiObjectiveOptimizer(objective, search_space, objectives)
        results = optimizer.optimize(n_trials=n_trials)
        
        results['experiment_results'] = self.experiment_results
        
        return results
    
    def save_results(self, filepath: str):
        """Save experiment results to JSON file."""
        # Convert numpy types and models to serializable format
        serializable_results = []
        
        for result in self.experiment_results:
            serializable_result = {
                'hyperparameters': result['hyperparameters'],
                'validation_metrics': self._make_serializable(result['validation_metrics']),
                'efficiency': self._make_serializable(result['efficiency']),
                'training_time': result['training_time'],
                'epochs_trained': result['epochs_trained'],
                'timestamp': result['timestamp']
            }
            
            if result.get('test_metrics'):
                serializable_result['test_metrics'] = self._make_serializable(result['test_metrics'])
            
            # Don't save full training history (too large)
            # Just save final values
            if 'training_history' in result:
                history = result['training_history']
                serializable_result['final_train_loss'] = history['train_loss'][-1] if history['train_loss'] else None
                serializable_result['final_val_loss'] = history['val_loss'][-1] if history['val_loss'] else None
            
            serializable_results.append(serializable_result)
        
        output = {
            'experiment_results': serializable_results,
            'best_metrics': self._make_serializable(self.best_metrics) if self.best_metrics else None,
            'seed': self.seed,
            'device': self.device
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj
    
    def load_results(self, filepath: str):
        """Load experiment results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.experiment_results = data.get('experiment_results', [])
        self.best_metrics = data.get('best_metrics')
        self.seed = data.get('seed', 42)
        self.device = data.get('device', 'cpu')
    
    def get_best_experiment(self, metric: str = 'loss') -> Optional[Dict]:
        """Get the best experiment based on a metric."""
        if not self.experiment_results:
            return None
        
        if metric == 'loss':
            best = min(self.experiment_results, key=lambda x: x['validation_metrics']['loss'])
        elif metric == 'accuracy':
            best = max(self.experiment_results, key=lambda x: x['validation_metrics']['accuracy'])
        else:
            best = min(self.experiment_results, key=lambda x: x['validation_metrics'].get(metric, float('inf')))
        
        return best
    
    def compare_experiments(self, metric: str = 'loss', top_n: int = 5) -> List[Dict]:
        """Get top N experiments sorted by metric."""
        if not self.experiment_results:
            return []
        
        sorted_results = sorted(
            self.experiment_results,
            key=lambda x: x['validation_metrics'].get(metric, float('inf'))
        )
        
        return sorted_results[:top_n]
