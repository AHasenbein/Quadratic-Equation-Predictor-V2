"""
Hyperparameter optimization using multiple strategies with callback support.
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np
from models.model import QuadraticPredictor, ModelTrainer
from utils.evaluation import evaluate
import torch
from typing import Optional, Callable


class OptunaOptimizer:
    """Optuna-based Bayesian optimization with callback support."""
    
    def __init__(self, train_loader, val_loader, device='cpu', epochs=50, callback=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.callback = callback  # Callback function(trial_num, loss, params)
    
    def objective(self, trial, search_space):
        """Optuna objective function."""
        params = {}
        
        if 'hidden_layers_count' in search_space:
            params['hidden_layers_count'] = trial.suggest_int('hidden_layers_count',
                search_space['hidden_layers_count']['low'],
                search_space['hidden_layers_count']['high'],
                step=search_space['hidden_layers_count'].get('step', 1))
        
        if 'hidden_layer_size' in search_space:
            params['hidden_layer_size'] = trial.suggest_int('hidden_layer_size',
                search_space['hidden_layer_size']['low'],
                search_space['hidden_layer_size']['high'],
                step=search_space['hidden_layer_size'].get('step', 4))
        
        if 'activation' in search_space:
            params['activation'] = trial.suggest_categorical('activation',
                search_space['activation']['choices'])
        
        if 'learning_rate' in search_space:
            params['learning_rate'] = trial.suggest_float('learning_rate',
                search_space['learning_rate']['low'],
                search_space['learning_rate']['high'],
                log=search_space['learning_rate'].get('log', False))
        
        if 'optimizer' in search_space:
            params['optimizer'] = trial.suggest_categorical('optimizer',
                search_space['optimizer']['choices'])
        
        if 'dropout' in search_space:
            params['dropout'] = trial.suggest_float('dropout',
                search_space['dropout']['low'],
                search_space['dropout']['high'],
                step=search_space['dropout'].get('step', 0.05))
        
        if 'weight_decay' in search_space:
            # Ensure log scale only when low > 0
            use_log = search_space['weight_decay'].get('log', False)
            low = search_space['weight_decay']['low']
            if use_log and low <= 0:
                use_log = False  # Can't use log scale with low <= 0
            params['weight_decay'] = trial.suggest_float('weight_decay',
                max(low, 1e-6) if use_log else low,  # Ensure positive for log
                search_space['weight_decay']['high'],
                log=use_log)
        
        hidden_layers = [params['hidden_layer_size']] * params['hidden_layers_count']
        model = QuadraticPredictor(hidden_layers=hidden_layers, activation=params['activation'],
            dropout=params.get('dropout', 0.0)).to(self.device)
        
        trainer = ModelTrainer(model, learning_rate=params['learning_rate'],
            optimizer=params.get('optimizer', 'adam'), weight_decay=params.get('weight_decay', 0.0),
            device=self.device)
        
        for epoch in range(self.epochs):
            trainer.train_epoch(self.train_loader)
        
        metrics = evaluate(model, self.val_loader, self.device)
        loss = metrics['loss']
        
        # Store full metrics in trial user_attrs for later retrieval
        trial.set_user_attr('metrics', metrics)
        
        # Call callback if provided
        if self.callback:
            self.callback(trial.number, loss, params)
        
        return loss
    
    def optimize(self, n_trials=100, search_space=None):
        """Run Optuna optimization."""
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10))
        
        # Wrap objective to pass search_space
        def wrapped_objective(trial):
            return self.objective(trial, search_space)
        
        study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best trial metrics
        best_trial = study.best_trial
        best_metrics = best_trial.user_attrs.get('metrics', {})
        
        return {
            'best_params': study.best_params, 
            'best_value': study.best_value,
            'best_metrics': best_metrics,  # Include full metrics
            'study': study, 
            'n_trials': len(study.trials)
        }


class RandomSearchOptimizer:
    """Random search optimization with callback support."""
    
    def __init__(self, train_loader, val_loader, device='cpu', epochs=50, callback=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.callback = callback
    
    def optimize(self, n_trials=100, search_space=None):
        """Run random search."""
        best_loss, best_params, best_metrics = float('inf'), None, {}
        all_trials = []
        
        for trial_num in range(n_trials):
            params = {}
            
            if 'hidden_layers_count' in search_space:
                params['hidden_layers_count'] = np.random.randint(
                    search_space['hidden_layers_count']['low'],
                    search_space['hidden_layers_count']['high'] + 1)
            
            if 'hidden_layer_size' in search_space:
                low, high, step = search_space['hidden_layer_size']['low'], search_space['hidden_layer_size']['high'], search_space['hidden_layer_size'].get('step', 4)
                params['hidden_layer_size'] = np.random.choice(range(low, high + 1, step))
            
            if 'activation' in search_space:
                params['activation'] = np.random.choice(search_space['activation']['choices'])
            
            if 'learning_rate' in search_space:
                low, high = search_space['learning_rate']['low'], search_space['learning_rate']['high']
                if search_space['learning_rate'].get('log', False):
                    params['learning_rate'] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                else:
                    params['learning_rate'] = np.random.uniform(low, high)
            
            if 'optimizer' in search_space:
                params['optimizer'] = np.random.choice(search_space['optimizer']['choices'])
            
            if 'dropout' in search_space:
                low, high, step = search_space['dropout']['low'], search_space['dropout']['high'], search_space['dropout'].get('step', 0.05)
                params['dropout'] = np.random.choice(np.arange(low, high + step, step))
            
            if 'weight_decay' in search_space:
                low, high = search_space['weight_decay']['low'], search_space['weight_decay']['high']
                use_log = search_space['weight_decay'].get('log', False)
                if use_log and low > 0:
                    params['weight_decay'] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                else:
                    params['weight_decay'] = np.random.uniform(max(low, 0), high)
            
            hidden_layers = [params['hidden_layer_size']] * params['hidden_layers_count']
            model = QuadraticPredictor(hidden_layers=hidden_layers, activation=params['activation'],
                dropout=params.get('dropout', 0.0)).to(self.device)
            
            trainer = ModelTrainer(model, learning_rate=params['learning_rate'],
                optimizer=params.get('optimizer', 'adam'), weight_decay=params.get('weight_decay', 0.0),
                device=self.device)
            
            for epoch in range(self.epochs):
                trainer.train_epoch(self.train_loader)
            
            metrics = evaluate(model, self.val_loader, self.device)
            loss = metrics['loss']
            all_trials.append({'params': params, 'loss': loss, 'trial': trial_num, 'metrics': metrics})
            
            # Call callback
            if self.callback:
                self.callback(trial_num, loss, params)
            
            if loss < best_loss:
                best_loss, best_params = loss, params
                best_metrics = metrics  # Store best metrics
        
        return {
            'best_params': best_params, 
            'best_value': best_loss,
            'best_metrics': best_metrics,  # Include full metrics
            'n_trials': n_trials, 
            'all_trials': all_trials
        }


class MultiObjectiveOptimizer:
    """Multi-objective optimization using Optuna."""
    
    def __init__(self, train_loader, val_loader, device='cpu', epochs=50, callback=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.callback = callback
    
    def objective(self, trial, search_space):
        """Multi-objective function."""
        params = {}
        
        if 'hidden_layers_count' in search_space:
            params['hidden_layers_count'] = trial.suggest_int('hidden_layers_count',
                search_space['hidden_layers_count']['low'],
                search_space['hidden_layers_count']['high'],
                step=search_space['hidden_layers_count'].get('step', 1))
        
        if 'hidden_layer_size' in search_space:
            params['hidden_layer_size'] = trial.suggest_int('hidden_layer_size',
                search_space['hidden_layer_size']['low'],
                search_space['hidden_layer_size']['high'],
                step=search_space['hidden_layer_size'].get('step', 4))
        
        if 'activation' in search_space:
            params['activation'] = trial.suggest_categorical('activation',
                search_space['activation']['choices'])
        
        if 'dropout' in search_space:
            params['dropout'] = trial.suggest_float('dropout',
                search_space['dropout']['low'],
                search_space['dropout']['high'],
                step=search_space['dropout'].get('step', 0.05))
        
        hidden_layers = [params['hidden_layer_size']] * params['hidden_layers_count']
        model = QuadraticPredictor(hidden_layers=hidden_layers, activation=params['activation'],
            dropout=params.get('dropout', 0.0)).to(self.device)
        
        model_size_mb = model.get_model_size_mb()
        
        trainer = ModelTrainer(model, learning_rate=params.get('learning_rate', 0.001),
            optimizer=params.get('optimizer', 'adam'), device=self.device)
        
        for epoch in range(self.epochs):
            trainer.train_epoch(self.train_loader)
        
        metrics = evaluate(model, self.val_loader, self.device)
        loss = metrics['loss']
        
        # Store full metrics in trial user_attrs
        trial.set_user_attr('metrics', metrics)
        
        # Call callback
        if self.callback:
            self.callback(trial.number, loss, params)
        
        return loss, model_size_mb
    
    def optimize(self, n_trials=100, search_space=None):
        """Run multi-objective optimization."""
        study = optuna.create_study(directions=['minimize', 'minimize'], sampler=TPESampler(seed=42))
        
        def wrapped_objective(trial):
            return self.objective(trial, search_space)
        
        study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=False)
        
        pareto_trials = study.best_trials
        # Get metrics from best trial (first in Pareto front)
        best_metrics = {}
        if pareto_trials:
            best_metrics = pareto_trials[0].user_attrs.get('metrics', {})
        
        return {
            'pareto_front': [{'params': trial.params, 'values': trial.values, 'loss': trial.values[0],
                            'model_size_mb': trial.values[1]} for trial in pareto_trials],
            'best_metrics': best_metrics,  # Include metrics
            'study': study, 
            'n_trials': len(study.trials)
        }
