"""
Neural network model for predicting quadratic equation roots.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List


class QuadraticPredictor(nn.Module):
    """Neural network for predicting quadratic equation roots."""
    
    def __init__(self, hidden_layers: List[int] = [16, 8], activation: str = 'relu', dropout: float = 0.0, use_batch_norm: bool = False):
        super().__init__()
        layers = []
        prev_size = 3  # a, b, c
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            if dropout > 0 and i < len(hidden_layers) - 1:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 4))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


class ModelTrainer:
    """Trainer for the model."""
    
    def __init__(self, model, learning_rate=0.001, optimizer='adam', weight_decay=0.0, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.MSELoss()
        
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)
