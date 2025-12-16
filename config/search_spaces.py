"""
Hyperparameter search spaces and experiment presets.
"""

SEARCH_SPACES = {
    'tiny': {
        'hidden_layers_count': {'type': 'int', 'low': 1, 'high': 2, 'step': 1},
        'hidden_layer_size': {'type': 'int', 'low': 4, 'high': 16, 'step': 4},
        'activation': {'type': 'categorical', 'choices': ['relu', 'tanh']},
        'learning_rate': {'type': 'float', 'low': 1e-3, 'high': 1e-2, 'log': False},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.2, 'step': 0.1}
    },
    'small': {
        'hidden_layers_count': {'type': 'int', 'low': 1, 'high': 3, 'step': 1},
        'hidden_layer_size': {'type': 'int', 'low': 8, 'high': 32, 'step': 8},
        'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'gelu']},
        'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.3, 'step': 0.1}
    },
    'medium': {
        'hidden_layers_count': {'type': 'int', 'low': 2, 'high': 4, 'step': 1},
        'hidden_layer_size': {'type': 'int', 'low': 16, 'high': 64, 'step': 8},
        'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'gelu', 'swish']},
        'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.4, 'step': 0.1}
    },
    'large': {
        'hidden_layers_count': {'type': 'int', 'low': 3, 'high': 5, 'step': 1},
        'hidden_layer_size': {'type': 'int', 'low': 32, 'high': 128, 'step': 16},
        'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'gelu', 'swish', 'elu']},
        'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.1},
        'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw', 'sgd']},
        'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-3, 'log': True}
    }
}

EXPERIMENT_PRESETS = {
    'tiny': {'search_space': 'tiny', 'optimization_method': 'optuna', 'n_trials': 10, 'epochs_per_trial': 30, 'data_samples': 3000},
    'small': {'search_space': 'small', 'optimization_method': 'optuna', 'n_trials': 20, 'epochs_per_trial': 50, 'data_samples': 5000},
    'medium': {'search_space': 'medium', 'optimization_method': 'optuna', 'n_trials': 50, 'epochs_per_trial': 100, 'data_samples': 10000},
    'large': {'search_space': 'large', 'optimization_method': 'optuna', 'n_trials': 100, 'epochs_per_trial': 150, 'data_samples': 20000}
}

def get_search_space(name='small'):
    return SEARCH_SPACES.get(name, SEARCH_SPACES['small'])

def get_experiment_preset(name):
    return EXPERIMENT_PRESETS.get(name)
