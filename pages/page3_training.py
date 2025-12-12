"""
Page 3: Training interface with metrics and model visualization.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                             QPushButton, QProgressBar, QScrollArea, QFrame,
                             QGroupBox, QTextEdit, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
# Lazy import torch to speed up startup
# torch, torch.nn, DataLoader, TensorDataset will be imported when needed
from pages.base_page import BasePage
from config import COLORS, GRAPH_LINE_WIDTH
from models.quadratic_model import QuadraticPredictor, ModelTrainer
from utils.graph_styling import apply_standard_graph_style, style_legend
from utils.widget_styling import get_groupbox_style, get_button_style


class TrainingThread(QThread):
    """Thread for training model without blocking UI."""
    epoch_complete = pyqtSignal(dict)
    training_complete = pyqtSignal(object, dict)
    
    def __init__(self, model, train_data, val_data, preset, device='cpu'):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.preset = preset
        self.device = device
        self.should_stop = False
    
    def run(self):
        """Run training."""
        # Lazy import torch here to avoid slow startup
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        trainer = ModelTrainer(self.model, self.device)
        trainer.setup_optimizer(
            learning_rate=self.preset['learning_rate'],
            optimizer_type='adam'
        )
        
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(self.train_data['inputs']),
                torch.FloatTensor(self.train_data['targets'])
            ),
            batch_size=self.preset['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(self.val_data['inputs']),
                torch.FloatTensor(self.val_data['targets'])
            ),
            batch_size=self.preset['batch_size'],
            shuffle=False
        )
        
        best_model_state = None
        best_loss = float('inf')
        
        for epoch in range(self.preset['epochs']):
            if self.should_stop:
                break
            
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)
            
            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_model_state = self.model.state_dict().copy()
            
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'best_loss': best_loss
            }
            
            self.epoch_complete.emit(metrics)
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        final_metrics = {
            'best_loss': best_loss,
            'final_val_acc': val_metrics['accuracy']
        }
        
        self.training_complete.emit(self.model, final_metrics)
    
    def stop(self):
        """Stop training."""
        self.should_stop = True


class TrainingPage(BasePage):
    """Page for model training."""
    
    def __init__(self, parent=None):
        # Initialize instance variables before calling super().__init__()
        # because setup_ui() will be called during initialization
        self.current_model = None
        self.best_model = None
        self.training_thread = None
        self.training_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        super().__init__(parent)
    
    def setup_ui(self):
        """Setup the training UI."""
        super().setup_ui()  # Initialize base layout
        # Title
        title = QLabel("Model Training")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {COLORS['accent_primary']}; padding: 10px;")
        self.layout.addWidget(title)
        
        # Content layout (no nested scroll - using global scroll from base_page)
        content_layout = self.layout
        
        # Training Controls
        controls_frame = QFrame()
        controls_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        controls_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        controls_layout = QHBoxLayout(controls_frame)
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.start_btn.setStyleSheet(self.get_button_style())
        self.start_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.start_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.stop_btn.setStyleSheet(self.get_button_style())
        self.stop_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        controls_layout.addWidget(self.stop_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%p%")  # Show percentage
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_tertiary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 5px;
                text-align: center;
                color: {COLORS['text_primary']};
                font-weight: bold;
                font-size: 11px;
                height: 25px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['accent_primary']}, stop:1 {COLORS['accent_secondary']});
                border-radius: 3px;
            }}
        """)
        controls_layout.addWidget(self.progress_bar)
        
        content_layout.addWidget(controls_frame)
        
        # Metrics Display with better layout
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet(controls_frame.styleSheet())
        metrics_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setSpacing(15)
        
        # Current Model Metrics
        current_group = QGroupBox("Current Model")
        current_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        current_group.setStyleSheet(self.get_groupbox_style())
        current_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        current_layout = QVBoxLayout(current_group)
        
        self.current_loss_label = QLabel("Loss: --")
        self.current_loss_label.setFont(QFont("Arial", 11))
        self.current_loss_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.current_loss_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        current_layout.addWidget(self.current_loss_label)
        
        self.current_acc_label = QLabel("Accuracy: --")
        self.current_acc_label.setFont(QFont("Arial", 11))
        self.current_acc_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.current_acc_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        current_layout.addWidget(self.current_acc_label)
        
        self.current_success_label = QLabel("Success Rate: --")
        self.current_success_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.current_success_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_success_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        current_layout.addWidget(self.current_success_label)
        
        metrics_layout.addWidget(current_group, 0, 0)
        
        # Best Model Metrics
        best_group = QGroupBox("Best Model")
        best_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        best_group.setStyleSheet(self.get_groupbox_style())
        best_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        best_layout = QVBoxLayout(best_group)
        
        self.best_loss_label = QLabel("Best Loss: --")
        self.best_loss_label.setFont(QFont("Arial", 11))
        self.best_loss_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.best_loss_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        best_layout.addWidget(self.best_loss_label)
        
        self.best_acc_label = QLabel("Best Accuracy: --")
        self.best_acc_label.setFont(QFont("Arial", 11))
        self.best_acc_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.best_acc_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        best_layout.addWidget(self.best_acc_label)
        
        self.best_success_label = QLabel("Success Rate: --")
        self.best_success_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.best_success_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.best_success_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        best_layout.addWidget(self.best_success_label)
        
        metrics_layout.addWidget(best_group, 0, 1)
        
        # Set equal column stretch
        metrics_layout.setColumnStretch(0, 1)
        metrics_layout.setColumnStretch(1, 1)
        
        content_layout.addWidget(metrics_frame)
        
        # Training Log
        log_group = QGroupBox("Training Log")
        log_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        log_group.setStyleSheet(self.get_groupbox_style())
        log_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Use a monospace font - try common ones
        monospace_font = QFont()
        monospace_font.setStyleHint(QFont.StyleHint.Monospace)
        monospace_font.setPointSize(9)
        self.log_text.setFont(monospace_font)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_tertiary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 5px;
                color: {COLORS['text_primary']};
                padding: 10px;
            }}
        """)
        log_layout.addWidget(self.log_text)
        
        content_layout.addWidget(log_group)
        
        # Visualization Section
        viz_group = QGroupBox("Training Visualization")
        viz_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        viz_group.setStyleSheet(self.get_groupbox_style())
        viz_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viz_layout = QVBoxLayout(viz_group)
        
        # Matplotlib figure with minimum size and proper size policy
        self.figure = Figure(figsize=(14, 10), facecolor=COLORS['bg_tertiary'])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(600)
        self.canvas.setMinimumWidth(1000)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setStyleSheet(f"background-color: {COLORS['bg_tertiary']};")
        viz_layout.addWidget(self.canvas)
        
        content_layout.addWidget(viz_group)
        
        # Initialize plots
        self.update_plots()
    
    def get_button_style(self):
        return get_button_style(primary=True)
    
    def get_groupbox_style(self):
        return get_groupbox_style()
    
    def start_training(self):
        """Start model training."""
        # Get data from page 2 using helper method
        main_window = self.get_main_window()
        if not main_window or not hasattr(main_window, 'page2'):
            self.log_text.append("Error: Cannot access data generation page.")
            return
        
        data = main_window.page2.get_generated_data()
        preset = main_window.page2.get_selected_preset()
        
        if data is None:
            self.log_text.append("Error: No data generated. Please generate data first.")
            return
        
        # Split data
        inputs = data['inputs']
        targets = data['targets']
        split_idx = int(len(inputs) * 0.8)
        
        train_data = {
            'inputs': inputs[:split_idx],
            'targets': targets[:split_idx]
        }
        val_data = {
            'inputs': inputs[split_idx:],
            'targets': targets[split_idx:]
        }
        
        # Create model
        self.current_model = QuadraticPredictor(hidden_layers=preset['hidden_layers'])
        # Lazy import torch to check for CUDA
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Start training thread
        self.training_thread = TrainingThread(self.current_model, train_data, val_data, preset, device)
        self.training_thread.epoch_complete.connect(self.on_epoch_complete)
        self.training_thread.training_complete.connect(self.on_training_complete)
        self.training_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setMaximum(preset['epochs'])
        self.progress_bar.setValue(0)
        self.log_text.append(f"Starting training with {preset['epochs']} epochs...")
    
    def stop_training(self):
        """Stop model training."""
        if self.training_thread:
            self.training_thread.stop()
            self.log_text.append("Training stopped by user.")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def on_epoch_complete(self, metrics):
        """Handle epoch completion."""
        epoch = metrics['epoch']
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        train_acc = metrics['train_acc']
        val_acc = metrics['val_acc']
        
        # Update history
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['val_acc'].append(val_acc)
        
        # Update UI
        self.current_loss_label.setText(f"Loss: {val_loss:.6f}")
        self.current_acc_label.setText(f"Accuracy: {train_acc:.2f}%")
        self.update_success_label(self.current_success_label, val_acc)
        
        self.best_loss_label.setText(f"Best Loss: {metrics['best_loss']:.6f}")
        
        self.progress_bar.setValue(epoch)
        
        log_msg = f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Acc={val_acc:.2f}%"
        self.log_text.append(log_msg)
        
        # Update plots
        self.update_plots()
    
    def on_training_complete(self, model, metrics):
        """Handle training completion."""
        self.best_model = model
        self.current_model = model
        
        self.log_text.append(f"\nTraining Complete!")
        self.log_text.append(f"Best Loss: {metrics['best_loss']:.6f}")
        self.log_text.append(f"Final Accuracy: {metrics['final_val_acc']:.2f}%")
        
        self.best_acc_label.setText(f"Best Accuracy: {metrics['final_val_acc']:.2f}%")
        self.update_success_label(self.best_success_label, metrics['final_val_acc'])
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
    
    def update_success_label(self, label, accuracy):
        """Update success rate label with color coding."""
        label.setText(f"Success Rate: {accuracy:.1f}%")
        
        if accuracy >= 90:
            color = COLORS['accent_success']
        elif accuracy >= 70:
            color = COLORS['accent_warning']
        else:
            color = COLORS['accent_error']
        
        label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background-color: {COLORS['bg_tertiary']};
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }}
        """)
    
    def update_plots(self):
        """Update training plots."""
        self.figure.clear()
        
        if not self.training_history['epoch']:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor(COLORS['bg_tertiary'])
            ax.text(0.5, 0.5, 'Training data will appear here', 
                   ha='center', va='center', 
                   color=COLORS['text_secondary'], fontsize=14,
                   transform=ax.transAxes)
            self.canvas.draw()
            return
        
        # Use gridspec for better control and spacing
        from matplotlib import gridspec
        gs = gridspec.GridSpec(2, 2, figure=self.figure, hspace=0.4, wspace=0.35, 
                               left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        # Loss plot
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax1.plot(self.training_history['epoch'], self.training_history['train_loss'], 
                label='Train Loss', color=COLORS['graph_line'], linewidth=GRAPH_LINE_WIDTH)
        ax1.plot(self.training_history['epoch'], self.training_history['val_loss'], 
                label='Val Loss', color=COLORS['accent_success'], linewidth=GRAPH_LINE_WIDTH)
        apply_standard_graph_style(ax1, title='Training Loss', xlabel='Epoch', ylabel='Loss')
        style_legend(ax1)
        
        # Accuracy plot
        ax2 = self.figure.add_subplot(gs[0, 1])
        ax2.plot(self.training_history['epoch'], self.training_history['train_acc'], 
                label='Train Acc', color=COLORS['graph_line'], linewidth=GRAPH_LINE_WIDTH)
        ax2.plot(self.training_history['epoch'], self.training_history['val_acc'], 
                label='Val Acc', color=COLORS['accent_success'], linewidth=GRAPH_LINE_WIDTH)
        apply_standard_graph_style(ax2, title='Training Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
        style_legend(ax2)
        
        # Model architecture visualization
        ax3 = self.figure.add_subplot(gs[1, 0])
        ax3.set_facecolor(COLORS['bg_tertiary'])
        if self.current_model:
            self.draw_neural_network(ax3, self.current_model, 'Current Model', COLORS['accent_primary'])
        else:
            ax3.text(0.5, 0.5, 'No model loaded', ha='center', va='center',
                    color=COLORS['text_secondary'], fontsize=12, transform=ax3.transAxes)
            ax3.axis('off')
        ax3.set_title('Current Model Architecture', color=COLORS['accent_primary'], fontweight='bold', pad=15)
        
        ax4 = self.figure.add_subplot(gs[1, 1])
        ax4.set_facecolor(COLORS['bg_tertiary'])
        if self.current_model:
            self.draw_layer_statistics(ax4, self.current_model)
        else:
            ax4.text(0.5, 0.5, 'No model loaded', ha='center', va='center',
                    color=COLORS['text_secondary'], fontsize=12, transform=ax4.transAxes)
            ax4.axis('off')
        ax4.set_title('Model Layer Analysis', color=COLORS['accent_primary'], fontweight='bold', pad=15)
        
        # Use constrained_layout instead of tight_layout to avoid warnings
        self.figure.set_constrained_layout(True)
        self.canvas.draw()
    
    def draw_neural_network(self, ax, model, title, color):
        """Draw a neural network architecture diagram with enhanced visuals."""
        ax.clear()
        ax.set_facecolor(COLORS['bg_tertiary'])
        ax.axis('off')
        
        # Get model structure
        layers = [3]  # Input layer (a, b, c)
        for layer in model.hidden_layers:
            layers.append(layer)
        layers.append(4)  # Output layer (4 values)
        
        # Color coding for layers
        layer_colors = {
            'input': COLORS['accent_success'],  # Green
            'hidden': COLORS['accent_primary'],  # Cyan
            'output': COLORS['accent_secondary']  # Purple
        }
        
        # Add padding for labels and info
        x_start = 0.15
        x_end = 0.85
        y_bottom = 0.2  # Space for layer labels
        y_top = 0.85    # Space for info box
        
        # Calculate actual drawing area
        num_layers = len(layers)
        draw_width = x_end - x_start
        draw_height = y_top - y_bottom
        layer_x_spacing = draw_width / (num_layers - 1) if num_layers > 1 else draw_width
        max_neurons = max(layers)
        
        # Draw connections first (so they appear behind neurons)
        layer_positions = []
        for i, num_neurons in enumerate(layers):
            x = x_start + (i * layer_x_spacing) if num_layers > 1 else (x_start + x_end) / 2
            neuron_spacing = draw_height / max(num_neurons, 1) if num_neurons > 1 else draw_height
            start_y = y_bottom + (max_neurons - num_neurons) * neuron_spacing / 2
            
            neuron_positions = []
            for j in range(num_neurons):
                y = start_y + j * neuron_spacing
                neuron_positions.append((x, y))
            layer_positions.append(neuron_positions)
        
        # Draw connections with varying thickness based on layer depth
        for i in range(len(layer_positions) - 1):
            # Thicker lines for connections closer to input/output
            depth_factor = 1.0 - abs(i - (num_layers - 1) / 2) / (num_layers / 2)
            line_width = 0.2 + depth_factor * 0.3
            connection_color = layer_colors['hidden'] if i < len(layer_positions) - 2 else layer_colors['output']
            
            for neuron1 in layer_positions[i]:
                for neuron2 in layer_positions[i + 1]:
                    ax.plot([neuron1[0], neuron2[0]], [neuron1[1], neuron2[1]], 
                           color=connection_color, alpha=0.2, linewidth=line_width, zorder=1)
        
        # Draw neurons with gradients and glow
        for i, (num_neurons, neuron_positions) in enumerate(zip(layers, layer_positions)):
            # Determine layer color
            if i == 0:
                layer_color = layer_colors['input']
            elif i == len(layers) - 1:
                layer_color = layer_colors['output']
            else:
                layer_color = layer_colors['hidden']
            
            for x, y in neuron_positions:
                # Create gradient effect using multiple circles
                neuron_radius = 0.018
                
                # Outer glow
                glow = Circle((x, y), neuron_radius * 1.5, 
                             color=layer_color, alpha=0.15, fill=True, zorder=2)
                ax.add_patch(glow)
                
                # Main neuron with gradient effect (darker center, lighter edges)
                # Draw multiple circles for gradient effect
                for alpha_val, radius_mult in [(0.9, 0.3), (0.7, 0.6), (0.5, 0.9), (0.3, 1.0)]:
                    circle = Circle((x, y), neuron_radius * radius_mult, 
                                   color=layer_color, alpha=alpha_val, fill=True, zorder=3)
                    ax.add_patch(circle)
                
                # Center highlight
                center = Circle((x, y), neuron_radius * 0.3, 
                               color='white', alpha=0.4, fill=True, zorder=4)
                ax.add_patch(center)
            
            # Draw layer label with better typography
            layer_name = 'Input' if i == 0 else ('Output' if i == len(layers) - 1 else f'H{i}')
            x_pos = neuron_positions[0][0] if neuron_positions else (x_start + x_end) / 2
            
            # Layer label with background box
            label_text = f'{layer_name}\n({num_neurons})'
            ax.text(x_pos, y_bottom - 0.08, label_text, 
                   ha='center', va='top', color=COLORS['text_primary'], 
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor=COLORS['bg_secondary'], 
                            edgecolor=layer_color, 
                            alpha=0.7, linewidth=1.5))
            
            # Activation function indicator (ReLU for hidden layers)
            if i > 0 and i < len(layers) - 1:
                ax.text(x_pos, y_top + 0.02, 'ReLU', 
                       ha='center', va='bottom', 
                       color=layer_color, fontsize=7, 
                       style='italic', alpha=0.7)
            
            # Add layer statistics (parameter count per layer)
            if i < len(layers) - 1:
                # Calculate parameters for this layer connection
                if i == 0:
                    layer_params = layers[i] * layers[i + 1] + layers[i + 1]  # weights + biases
                else:
                    layer_params = layers[i] * layers[i + 1] + layers[i + 1]
                
                # Show parameter count above layer (if space allows)
                if num_neurons <= 8:  # Only show for smaller layers to avoid clutter
                    stats_text = f'{layer_params:,} params'
                    ax.text(x_pos, y_top - 0.02, stats_text, 
                           ha='center', va='top', 
                           color=COLORS['text_muted'], fontsize=6, 
                           alpha=0.6)
        
        # Add model info at the top center with better styling
        params = model.count_parameters()
        size_mb = model.get_model_size_mb()
        info_text = f'Total Parameters: {params:,}\nModel Size: {size_mb:.2f} MB'
        ax.text(0.5, y_top + 0.05, info_text, ha='center', va='bottom',
               color=COLORS['text_secondary'], fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor=COLORS['bg_secondary'], 
                        edgecolor=color, alpha=0.9, linewidth=2))
        
        # Set axis limits with proper padding
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def draw_layer_statistics(self, ax, model):
        """Draw a cool layer statistics breakdown visualization."""
        ax.clear()
        ax.set_facecolor(COLORS['bg_tertiary'])
        
        # Get model structure
        layers = [3]  # Input layer
        for layer in model.hidden_layers:
            layers.append(layer)
        layers.append(4)  # Output layer
        
        # Calculate parameters per layer
        layer_params = []
        layer_names = []
        total_params = 0
        
        for i in range(len(layers) - 1):
            if i == 0:
                name = 'Input→H1'
            elif i == len(layers) - 2:
                name = f'H{len(layers)-2}→Output'
            else:
                name = f'H{i}→H{i+1}'
            
            # Calculate parameters: weights + biases
            params = layers[i] * layers[i + 1] + layers[i + 1]
            layer_params.append(params)
            layer_names.append(name)
            total_params += params
        
        # Create a cool 3D-style bar chart
        y_pos = np.arange(len(layer_names))
        colors_list = [COLORS['accent_success'], COLORS['accent_primary'], 
                      COLORS['accent_secondary'], COLORS['accent_warning']]
        
        # Normalize parameters for visualization
        max_params = max(layer_params) if layer_params else 1
        normalized_params = [p / max_params for p in layer_params]
        
        # Draw bars with gradient effect
        bars = ax.barh(y_pos, normalized_params, height=0.6, 
                      color=[colors_list[i % len(colors_list)] for i in range(len(layer_names))],
                      alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for i, (bar, params) in enumerate(zip(bars, layer_params)):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{params:,}', ha='left', va='center',
                   color=COLORS['text_primary'], fontsize=8, fontweight='bold')
            
            # Add percentage
            percentage = (params / total_params * 100) if total_params > 0 else 0
            ax.text(-0.05, bar.get_y() + bar.get_height()/2,
                   f'{percentage:.1f}%', ha='right', va='center',
                   color=COLORS['text_secondary'], fontsize=7)
        
        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layer_names, color=COLORS['text_primary'], fontsize=9, fontweight='bold')
        ax.set_xlabel('Normalized Parameter Count', color=COLORS['text_primary'], fontsize=10)
        
        # Add title info
        info_text = f'Total: {total_params:,} params'
        ax.text(0.5, -0.15, info_text, ha='center', va='top',
               color=COLORS['accent_primary'], fontsize=9, fontweight='bold',
               transform=ax.transAxes)
        
        # Style the plot
        ax.set_xlim(-0.15, 1.1)
        ax.set_ylim(-0.5, len(layer_names) - 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['text_primary'])
        ax.spines['left'].set_color(COLORS['text_primary'])
        ax.tick_params(colors=COLORS['text_primary'], labelsize=8)
        ax.grid(True, axis='x', color=COLORS['graph_grid'], alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    def get_trained_model(self):
        """Get the trained model."""
        return self.best_model if self.best_model else self.current_model

