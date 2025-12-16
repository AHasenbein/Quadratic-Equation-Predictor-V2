"""
Optimization configuration: search spaces, experiment presets, and evaluation priorities.
"""

from typing import Dict, List


# Hyperparameter search spaces
SEARCH_SPACES = {
    'full': {
        'hidden_layers_count': {
            'type': 'int',
            'low': 1,
            'high': 5,
            'step': 1
        },
        'hidden_layer_size': {
            'type': 'int',
            'low': 4,
            'high': 128,
            'step': 4
        },
        'activation': {
            'type': 'categorical',
            'choices': ['relu', 'tanh', 'gelu', 'swish', 'sigmoid', 'elu']
        },
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-1,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 16,
            'high': 512,
            'step': 16,
            'log': True
        },
        'optimizer': {
            'type': 'categorical',
            'choices': ['adam', 'adamw', 'sgd', 'rmsprop']
        },
        'dropout': {
            'type': 'float',
            'low': 0.0,
            'high': 0.5,
            'step': 0.05
        },
        'weight_decay': {
            'type': 'float',
            'low': 0.0,
            'high': 1e-3,
            'log': True
        }
    },
    'architecture_only': {
        'hidden_layers_count': {
            'type': 'int',
            'low': 1,
            'high': 5,
            'step': 1
        },
        'hidden_layer_size': {
            'type': 'int',
            'low': 4,
            'high': 128,
            'step': 4
        },
        'activation': {
            'type': 'categorical',
            'choices': ['relu', 'tanh', 'gelu', 'swish']
        },
        'dropout': {
            'type': 'float',
            'low': 0.0,
            'high': 0.5,
            'step': 0.1
        }
    },
    'training_only': {
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-1,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 16,
            'high': 512,
            'step': 16,
            'log': True
        },
        'optimizer': {
            'type': 'categorical',
            'choices': ['adam', 'adamw', 'sgd', 'rmsprop']
        },
        'weight_decay': {
            'type': 'float',
            'low': 0.0,
            'high': 1e-3,
            'log': True
        }
    },
    'quick': {
        'hidden_layers_count': {
            'type': 'int',
            'low': 1,
            'high': 3,
            'step': 1
        },
        'hidden_layer_size': {
            'type': 'int',
            'low': 8,
            'high': 64,
            'step': 8
        },
        'activation': {
            'type': 'categorical',
            'choices': ['relu', 'tanh', 'gelu']
        },
        'learning_rate': {
            'type': 'float',
            'low': 1e-4,
            'high': 1e-2,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 32,
            'high': 256,
            'step': 32
        }
    }
}


# Experiment presets
EXPERIMENT_PRESETS = {
    'quick_optimization': {
        'search_space': 'quick',
        'optimization_method': 'optuna',
        'n_trials': 20,
        'epochs_per_trial': 50,
        'objective_metric': 'loss'
    },
    'full_optimization': {
        'search_space': 'full',
        'optimization_method': 'optuna',
        'n_trials': 100,
        'epochs_per_trial': 100,
        'objective_metric': 'loss'
    },
    'architecture_study': {
        'search_space': 'architecture_only',
        'optimization_method': 'optuna',
        'n_trials': 50,
        'epochs_per_trial': 100,
        'objective_metric': 'loss'
    },
    'training_study': {
        'search_space': 'training_only',
        'optimization_method': 'optuna',
        'n_trials': 50,
        'epochs_per_trial': 100,
        'objective_metric': 'loss'
    },
    'grid_search_small': {
        'search_space': 'quick',
        'optimization_method': 'grid',
        'n_trials': None,  # Grid search uses all combinations
        'epochs_per_trial': 50,
        'objective_metric': 'loss'
    },
    'random_search': {
        'search_space': 'full',
        'optimization_method': 'random',
        'n_trials': 100,
        'epochs_per_trial': 100,
        'objective_metric': 'loss'
    },
    'multi_objective': {
        'search_space': 'full',
        'optimization_method': 'multi_objective',
        'n_trials': 100,
        'epochs_per_trial': 100,
        'objectives': ['loss', 'model_size_mb']
    }
}


# Evaluation metric priorities
EVALUATION_PRIORITIES = {
    'accuracy_focused': {
        'primary': 'accuracy',
        'secondary': ['loss', 'rmse'],
        'tertiary': ['mae', 'r2']
    },
    'efficiency_focused': {
        'primary': 'model_size_mb',
        'secondary': ['inference_time_ms', 'param_count'],
        'tertiary': ['accuracy', 'loss']
    },
    'balanced': {
        'primary': 'loss',
        'secondary': ['accuracy', 'rmse'],
        'tertiary': ['model_size_mb', 'inference_time_ms']
    },
    'research': {
        'primary': 'loss',
        'secondary': ['accuracy', 'rmse', 'mae', 'r2'],
        'tertiary': ['model_size_mb', 'inference_time_ms', 'param_count']
    }
}


# Default hyperparameters (for baseline)
DEFAULT_HYPERPARAMETERS = {
    'hidden_layers': [16, 8],
    'activation': 'relu',
    'learning_rate': 0.001,
    'batch_size': 64,
    'optimizer': 'adam',
    'dropout': 0.1,
    'weight_decay': 0.0,
    'use_batch_norm': False,
    'epochs': 100
}


# Hyperparameter importance thresholds
IMPORTANCE_THRESHOLDS = {
    'high': 0.1,
    'medium': 0.05,
    'low': 0.01
}


def get_search_space(name: str = 'full') -> Dict:
    """Get search space by name."""
    return SEARCH_SPACES.get(name, SEARCH_SPACES['full'])


def get_experiment_preset(name: str) -> Dict:
    """Get experiment preset by name."""
    return EXPERIMENT_PRESETS.get(name)


def get_evaluation_priority(name: str = 'balanced') -> Dict:
    """Get evaluation priority configuration by name."""
    return EVALUATION_PRIORITIES.get(name, EVALUATION_PRIORITIES['balanced'])


def convert_search_space_to_hyperparameters(search_space: Dict, trial_params: Dict) -> Dict:
    """
    Convert search space trial parameters to actual hyperparameters.
    Handles special cases like hidden_layers construction.
    """
    hyperparameters = {}
    
    # Handle hidden layers
    if 'hidden_layers_count' in trial_params and 'hidden_layer_size' in trial_params:
        count = trial_params['hidden_layers_count']
        size = trial_params['hidden_layer_size']
        hyperparameters['hidden_layers'] = [size] * count
    elif 'hidden_layers' in trial_params:
        hyperparameters['hidden_layers'] = trial_params['hidden_layers']
    
    # Copy other parameters
    for key in ['activation', 'learning_rate', 'batch_size', 'optimizer', 
                'dropout', 'weight_decay', 'use_batch_norm']:
        if key in trial_params:
            hyperparameters[key] = trial_params[key]
    
    return hyperparameters
