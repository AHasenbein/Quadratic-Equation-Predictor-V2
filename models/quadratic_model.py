"""
Neural network model for predicting quadratic equation roots.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
import numpy as np


class QuadraticPredictor(nn.Module):
    """
    Small, efficient neural network for predicting quadratic equation roots.
    """
    
    def __init__(self, hidden_layers: List[int] = [16, 8], activation: str = 'relu'):
        super(QuadraticPredictor, self).__init__()
        
        self.hidden_layers = hidden_layers
        layers = []
        
        # Input layer (3 coefficients: a, b, c)
        input_size = 3
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        # Output layer (4 values: r1_real, r2_real, r1_imag, r2_imag)
        layers.append(nn.Linear(prev_size, 4))
        
        self.network = nn.Sequential(*layers)
    
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


class ModelTrainer:
    """
    Handles training of the quadratic predictor model.
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
    
    def setup_optimizer(self, learning_rate: float = 0.001, optimizer_type: str = 'adam'):
        """Setup optimizer."""
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
    
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

