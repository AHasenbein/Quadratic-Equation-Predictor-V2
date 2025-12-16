"""
Hyperparameter optimization framework using Optuna, grid search, and random search.
Supports single-objective and multi-objective optimization.
"""

import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from optuna.pruners import MedianPruner
from typing import Dict, List, Callable, Optional, Tuple, Any
import numpy as np
import json
import os
from datetime import datetime


class HyperparameterOptimizer:
    """Base class for hyperparameter optimization."""
    
    def __init__(self, objective_function: Callable, search_space: Dict):
        """
        Initialize optimizer.
        
        Args:
            objective_function: Function that takes hyperparameters and returns score
            search_space: Dictionary defining search space
        """
        self.objective_function = objective_function
        self.search_space = search_space
        self.trials = []
        self.best_params = None
        self.best_score = None
    
    def optimize(self, n_trials: int = 100) -> Dict:
        """Run optimization."""
        raise NotImplementedError


class OptunaOptimizer(HyperparameterOptimizer):
    """Bayesian optimization using Optuna."""
    
    def __init__(
        self,
        objective_function: Callable,
        search_space: Dict,
        direction: str = 'minimize',
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ):
        """
        Initialize Optuna optimizer.
        
        Args:
            objective_function: Function that takes trial and returns score
            search_space: Dictionary defining search space
            direction: 'minimize' or 'maximize'
            study_name: Name for the study
            storage: Storage URL for study persistence
        """
        super().__init__(objective_function, search_space)
        self.direction = direction
        self.study_name = study_name or f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        if storage:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=True
            )
        else:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=direction,
                sampler=sampler,
                pruner=pruner
            )
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters from search space."""
        params = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1),
                    log=param_config.get('log', False)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', None),
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params
    
    def optimize(self, n_trials: int = 100) -> Dict:
        """
        Run optimization.
        
        Args:
            n_trials: Number of trials to run
        
        Returns:
            Dictionary with best parameters and score
        """
        def objective(trial):
            params = self._suggest_hyperparameters(trial)
            score = self.objective_function(params)
            return score
        
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials)
        }
    
    def get_trial_history(self) -> List[Dict]:
        """Get history of all trials."""
        history = []
        for trial in self.study.trials:
            history.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })
        return history
    
    def get_hyperparameter_importance(self) -> Dict[str, float]:
        """Get hyperparameter importance scores."""
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except:
            return {}


class GridSearchOptimizer(HyperparameterOptimizer):
    """Exhaustive grid search optimization."""
    
    def __init__(self, objective_function: Callable, search_space: Dict):
        super().__init__(objective_function, search_space)
    
    def _generate_grid(self) -> List[Dict]:
        """Generate all combinations from search space."""
        from itertools import product
        
        param_names = []
        param_values = []
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'int':
                values = list(range(
                    param_config['low'],
                    param_config['high'] + 1,
                    param_config.get('step', 1)
                ))
            elif param_type == 'float':
                low = param_config['low']
                high = param_config['high']
                step = param_config.get('step', (high - low) / 10)
                values = np.arange(low, high + step, step).tolist()
            elif param_type == 'categorical':
                values = param_config['choices']
            
            param_names.append(param_name)
            param_values.append(values)
        
        # Generate all combinations
        combinations = []
        for combo in product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def optimize(self, n_trials: Optional[int] = None) -> Dict:
        """
        Run grid search.
        
        Args:
            n_trials: Ignored for grid search (uses all combinations)
        
        Returns:
            Dictionary with best parameters and score
        """
        grid = self._generate_grid()
        
        best_score = float('inf') if self.objective_function.__name__ != 'maximize' else float('-inf')
        best_params = None
        
        for params in grid:
            score = self.objective_function(params)
            
            if (self.objective_function.__name__ == 'maximize' and score > best_score) or \
               (self.objective_function.__name__ != 'maximize' and score < best_score):
                best_score = score
                best_params = params
            
            self.trials.append({
                'params': params,
                'score': score
            })
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.trials)
        }


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search optimization."""
    
    def __init__(self, objective_function: Callable, search_space: Dict, seed: int = 42):
        super().__init__(objective_function, search_space)
        self.rng = np.random.RandomState(seed)
    
    def _sample_hyperparameters(self) -> Dict:
        """Randomly sample hyperparameters from search space."""
        params = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'int':
                params[param_name] = self.rng.randint(
                    param_config['low'],
                    param_config['high'] + 1
                )
            elif param_type == 'float':
                if param_config.get('log', False):
                    log_low = np.log(param_config['low'])
                    log_high = np.log(param_config['high'])
                    params[param_name] = np.exp(self.rng.uniform(log_low, log_high))
                else:
                    params[param_name] = self.rng.uniform(
                        param_config['low'],
                        param_config['high']
                    )
            elif param_type == 'categorical':
                params[param_name] = self.rng.choice(param_config['choices'])
        
        return params
    
    def optimize(self, n_trials: int = 100) -> Dict:
        """
        Run random search.
        
        Args:
            n_trials: Number of random trials
        
        Returns:
            Dictionary with best parameters and score
        """
        best_score = float('inf')
        best_params = None
        
        for _ in range(n_trials):
            params = self._sample_hyperparameters()
            score = self.objective_function(params)
            
            if score < best_score:
                best_score = score
                best_params = params
            
            self.trials.append({
                'params': params,
                'score': score
            })
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.trials)
        }


class MultiObjectiveOptimizer:
    """Multi-objective optimization using Optuna."""
    
    def __init__(
        self,
        objective_function: Callable,
        search_space: Dict,
        objectives: List[str],
        study_name: Optional[str] = None
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objective_function: Function that returns dict of objective values
            search_space: Dictionary defining search space
            objectives: List of objective names to optimize
            study_name: Name for the study
        """
        self.objective_function = objective_function
        self.search_space = search_space
        self.objectives = objectives
        self.study_name = study_name or f"multiobj_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create multi-objective study
        sampler = TPESampler(seed=42, multivariate=True)
        self.study = optuna.create_study(
            study_name=self.study_name,
            directions=['minimize'] * len(objectives),  # All minimize for now
            sampler=sampler
        )
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters from search space."""
        params = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1),
                    log=param_config.get('log', False)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', None),
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params
    
    def optimize(self, n_trials: int = 100) -> Dict:
        """
        Run multi-objective optimization.
        
        Args:
            n_trials: Number of trials
        
        Returns:
            Dictionary with Pareto frontier solutions
        """
        def objective(trial):
            params = self._suggest_hyperparameters(trial)
            results = self.objective_function(params)
            
            # Return tuple of objective values
            return tuple([results[obj] for obj in self.objectives])
        
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get Pareto frontier
        pareto_trials = self.study.best_trials
        
        pareto_solutions = []
        for trial in pareto_trials:
            pareto_solutions.append({
                'params': trial.params,
                'values': trial.values,
                'objectives': {obj: val for obj, val in zip(self.objectives, trial.values)}
            })
        
        return {
            'pareto_solutions': pareto_solutions,
            'n_trials': len(self.study.trials),
            'n_pareto': len(pareto_trials)
        }
    
    def get_pareto_frontier(self) -> List[Dict]:
        """Get Pareto frontier solutions."""
        pareto_trials = self.study.best_trials
        return [
            {
                'params': trial.params,
                'values': trial.values,
                'objectives': {obj: val for obj, val in zip(self.objectives, trial.values)}
            }
            for trial in pareto_trials
        ]


def save_optimization_results(results: Dict, filepath: str):
    """Save optimization results to JSON file."""
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_optimization_results(filepath: str) -> Dict:
    """Load optimization results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
