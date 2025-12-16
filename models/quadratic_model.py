"""
Neural network model for predicting quadratic equation roots.
Enhanced with support for multiple architectures, activations, and training features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Callable
import numpy as np


class QuadraticPredictor(nn.Module):
    """
    Neural network for predicting quadratic equation roots.
    Supports multiple architectures, activation functions, and regularization.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [16, 8],
        activation: str = 'relu',
        dropout: float = 0.1,
        use_batch_norm: bool = False
    ):
        """
        Initialize model.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'gelu', 'swish', 'sigmoid')
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(QuadraticPredictor, self).__init__()
        
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        layers = []
        
        # Input layer (3 coefficients: a, b, c)
        input_size = 3
        prev_size = input_size
        
        # Get activation function
        activation_fn = self._get_activation(activation)
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(activation_fn)
            
            # Dropout (not on last layer)
            if dropout > 0 and i < len(hidden_layers) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer (4 values: r1_real, r2_real, r1_imag, r2_imag)
        layers.append(nn.Linear(prev_size, 4))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU(),  # SiLU is Swish
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def get_architecture_info(self) -> Dict:
        """Get information about the model architecture."""
        return {
            'hidden_layers': self.hidden_layers,
            'activation': self.activation_name,
            'dropout': self.dropout,
            'batch_norm': self.use_batch_norm,
            'total_params': self.count_parameters(),
            'model_size_mb': self.get_model_size_mb()
        }


class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Called at the end of each epoch."""
        pass
    
    def on_training_start(self):
        """Called at the start of training."""
        pass
    
    def on_training_end(self):
        """Called at the end of training."""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, monitor: str = 'val_loss'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor ('val_loss', 'val_accuracy', etc.)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf') if 'loss' in monitor else float('-inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Check if training should stop."""
        current_value = metrics.get(self.monitor, float('inf'))
        
        if 'loss' in self.monitor:
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True


class LearningRateSchedulerCallback(TrainingCallback):
    """Learning rate scheduler callback."""
    
    def __init__(self, scheduler_type: str = 'reduce_on_plateau', **kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler ('reduce_on_plateau', 'step', 'cosine')
            **kwargs: Additional arguments for scheduler
        """
        self.scheduler_type = scheduler_type
        self.scheduler = None
        self.scheduler_kwargs = kwargs
    
    def setup(self, optimizer: optim.Optimizer):
        """Setup the scheduler with optimizer."""
        if self.scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_kwargs.get('factor', 0.5),
                patience=self.scheduler_kwargs.get('patience', 5),
                verbose=True
            )
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_kwargs.get('step_size', 30),
                gamma=self.scheduler_kwargs.get('gamma', 0.1)
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_kwargs.get('T_max', 100)
            )
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Update learning rate."""
        if self.scheduler:
            if self.scheduler_type == 'reduce_on_plateau':
                self.scheduler.step(metrics.get('val_loss', metrics.get('loss', 0)))
            else:
                self.scheduler.step()


class ModelCheckpointCallback(TrainingCallback):
    """Model checkpointing callback."""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = True):
        """
        Initialize checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            save_best_only: Only save best model
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = float('inf') if 'loss' in monitor else float('-inf')
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Save checkpoint if needed."""
        current_value = metrics.get(self.monitor, float('inf'))
        
        should_save = False
        if self.save_best_only:
            if 'loss' in self.monitor:
                if current_value < self.best_value:
                    self.best_value = current_value
                    should_save = True
            else:
                if current_value > self.best_value:
                    self.best_value = current_value
                    should_save = True
        else:
            should_save = True
        
        if should_save:
            # Save would be handled by trainer
            pass


class ModelTrainer:
    """
    Handles training of the quadratic predictor model with callbacks and advanced features.
    """
    
    def __init__(self, model: QuadraticPredictor, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.MSELoss()
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'epoch': []
        }
        self.best_model_state = None
        self.best_loss = float('inf')
        self.callbacks: List[TrainingCallback] = []
    
    def add_callback(self, callback: TrainingCallback):
        """Add a training callback."""
        self.callbacks.append(callback)
        if isinstance(callback, LearningRateSchedulerCallback) and self.optimizer:
            callback.setup(self.optimizer)
    
    def setup_optimizer(
        self,
        learning_rate: float = 0.001,
        optimizer_type: str = 'adam',
        weight_decay: float = 0.0
    ):
        """Setup optimizer."""
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup schedulers if any
        for callback in self.callbacks:
            if isinstance(callback, LearningRateSchedulerCallback):
                callback.setup(self.optimizer)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (within tolerance)
            with torch.no_grad():
                pred_roots = outputs.cpu().numpy()
                true_roots = targets.cpu().numpy()
                # Simple accuracy: check if predictions are close
                tolerance = 0.1
                diff = np.abs(pred_roots - true_roots)
                correct += np.sum(diff < tolerance)
                total += pred_roots.size
        
        avg_loss = total_loss / len(train_loader)
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy
                pred_roots = outputs.cpu().numpy()
                true_roots = targets.cpu().numpy()
                tolerance = 0.1
                diff = np.abs(pred_roots - true_roots)
                correct += np.sum(diff < tolerance)
                total += pred_roots.size
        
        avg_loss = total_loss / len(val_loader)
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train_with_callbacks(
        self,
        train_loader,
        val_loader,
        epochs: int,
        save_checkpoint_path: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Train model with callbacks support.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_checkpoint_path: Optional path to save best model
        
        Returns:
            Training history dictionary
        """
        # Initialize callbacks
        for callback in self.callbacks:
            callback.on_training_start()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch': []
        }
        
        for epoch in range(epochs):
            # Callback: epoch start
            for callback in self.callbacks:
                callback.on_epoch_start(epoch)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                if save_checkpoint_path:
                    torch.save({
                        'model_state_dict': self.best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': self.best_loss
                    }, save_checkpoint_path)
            
            # Prepare metrics for callbacks
            epoch_metrics = {
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_acc': val_metrics['accuracy'],
                'loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy']
            }
            
            # Callback: epoch end
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_metrics)
            
            # Check early stopping
            should_stop = False
            for callback in self.callbacks:
                if isinstance(callback, EarlyStoppingCallback) and callback.should_stop:
                    should_stop = True
                    break
            
            if should_stop:
                break
        
        # Callback: training end
        for callback in self.callbacks:
            callback.on_training_end()
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return history



