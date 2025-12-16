"""
Page 2: Training & Optimization (Revamped)
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QSpinBox, QComboBox, QGroupBox, QTextEdit, QProgressBar,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import torch
from config import COLORS
from utils.runner import ExperimentRunner
from config.search_spaces import get_search_space, get_experiment_preset, EXPERIMENT_PRESETS
from models.model import QuadraticPredictor
from utils.model_visualization import visualize_model_architecture
from utils.evaluation import evaluate
from torch.utils.data import DataLoader, TensorDataset


class OptimizationThread(QThread):
    """Thread for running optimization."""
    progress = pyqtSignal(str)
    trial_update = pyqtSignal(int, float, dict)  # trial_num, loss, params
    finished = pyqtSignal(dict)
    
    def __init__(self, train_data, search_space, method, n_trials, epochs):
        super().__init__()
        self.train_data = train_data
        self.search_space = search_space
        self.method = method
        self.n_trials = n_trials
        self.epochs = epochs
    
    def run(self):
        runner = ExperimentRunner(seed=42)
        try:
            self.progress.emit("Starting optimization...")
            
            # Define callback to emit trial updates
            def trial_callback(trial_num, loss, params):
                self.trial_update.emit(trial_num, loss, params)
                self.progress.emit(f"Trial {trial_num + 1}/{self.n_trials}: Loss = {loss:.6f}")
            
            results = runner.run_optimization(
                train_data=self.train_data,
                search_space=self.search_space,
                optimization_method=self.method,
                n_trials=self.n_trials,
                epochs_per_trial=self.epochs,
                callback=trial_callback
            )
            self.finished.emit(results)
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit({})


class TrainingPage(QWidget):
    """Training and optimization page."""
    
    optimization_complete = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.training_data = None
        self.optimization_thread = None
        self.results = None
        self.best_models = []  # Track best models
        self.trial_losses = []
        self.trial_params = []
        self.architecture_history = []  # Track architecture history for slideshow
        self.current_arch_index = -1  # Current architecture in history
        self.best_loss_so_far = float('inf')  # Track best loss for real-time updates
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Training & Hyperparameter Optimization")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['accent_primary']};")
        layout.addWidget(title)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 5px;
                background-color: {COLORS['bg_secondary']};
            }}
            QTabBar::tab {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                padding: 10px 20px;
                border: 1px solid {COLORS['accent_primary']};
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['accent_primary']};
                color: {COLORS['bg_primary']};
            }}
        """)
        
        # Tab 1: Optimization Control
        control_tab = self._create_control_tab()
        tabs.addTab(control_tab, "Optimization")
        
        # Tab 2: Model Visualization
        viz_tab = self._create_viz_tab()
        tabs.addTab(viz_tab, "Model Architecture")
        
        # Tab 3: Best Models Tracker
        tracker_tab = self._create_tracker_tab()
        tabs.addTab(tracker_tab, "Performance Analysis")
        
        # Tab 4: Interactive Testing
        test_tab = self._create_test_tab()
        tabs.addTab(test_tab, "Test Model")
        
        layout.addWidget(tabs)
    
    def _create_control_tab(self):
        """Create optimization control tab."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: Controls
        controls_group = QGroupBox("Optimization Settings")
        controls_group.setStyleSheet(f"""
            QGroupBox {{
                background-color: {COLORS['bg_secondary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 10px;
                padding: 15px;
                color: {COLORS['accent_primary']};
                font-weight: bold;
            }}
        """)
        controls_layout = QVBoxLayout(controls_group)
        
        # Preset selection
        preset_label = QLabel("Model Size Preset:")
        preset_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        controls_layout.addWidget(preset_label)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['tiny', 'small', 'medium', 'large'])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        controls_layout.addWidget(self.preset_combo)
        
        # Method selection
        method_label = QLabel("Optimization Method:")
        method_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        controls_layout.addWidget(method_label)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(['optuna', 'random', 'multi_objective'])
        controls_layout.addWidget(self.method_combo)
        
        # Trials
        trials_label = QLabel("Number of Trials:")
        trials_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        controls_layout.addWidget(trials_label)
        
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 1000)
        self.trials_spin.setValue(20)
        controls_layout.addWidget(self.trials_spin)
        
        # Epochs
        epochs_label = QLabel("Epochs per Trial:")
        epochs_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        controls_layout.addWidget(epochs_label)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        controls_layout.addWidget(self.epochs_spin)
        
        # Start button
        self.start_btn = QPushButton("Start Optimization")
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_success']};
                color: {COLORS['bg_primary']};
                border: 2px solid {COLORS['accent_success']};
                border-radius: 5px;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_primary']};
                border-color: {COLORS['accent_primary']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.start_btn.clicked.connect(self._start_optimization)
        controls_layout.addWidget(self.start_btn)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        controls_layout.addWidget(self.progress)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['accent_primary']};
                border-radius: 5px;
            }}
        """)
        controls_layout.addWidget(self.status_text)
        
        controls_layout.addStretch()
        
        # Right: Progress visualization
        self.progress_figure = Figure(figsize=(8, 6), facecolor=COLORS['bg_tertiary'])
        self.progress_canvas = FigureCanvas(self.progress_figure)
        self.progress_ax = self.progress_figure.add_subplot(111)
        self.progress_ax.set_facecolor(COLORS['bg_tertiary'])
        self.progress_ax.tick_params(colors=COLORS['text_primary'])
        for spine in self.progress_ax.spines.values():
            spine.set_color(COLORS['text_primary'])
        self.progress_ax.set_title("Optimization Progress", color=COLORS['accent_primary'], fontsize=14, fontweight='bold')
        self.progress_ax.set_xlabel("Trial", color=COLORS['text_primary'])
        self.progress_ax.set_ylabel("Loss", color=COLORS['text_primary'])
        self.progress_ax.grid(True, alpha=0.3, color=COLORS['accent_primary'])
        
        layout.addWidget(controls_group, 1)
        layout.addWidget(self.progress_canvas, 2)
        
        # Update preset
        self._on_preset_changed(self.preset_combo.currentText())
        
        return widget
    
    def _create_viz_tab(self):
        """Create model visualization tab with architecture history slideshow."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Navigation controls
        nav_group = QGroupBox("Architecture History")
        nav_group.setStyleSheet(f"""
            QGroupBox {{
                background-color: {COLORS['bg_secondary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 10px;
                padding: 15px;
                color: {COLORS['accent_primary']};
                font-weight: bold;
            }}
        """)
        nav_layout = QHBoxLayout(nav_group)
        
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self._show_previous_architecture)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.arch_info_label = QLabel("No architecture history yet")
        self.arch_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.arch_info_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")
        nav_layout.addWidget(self.arch_info_label)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._show_next_architecture)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        
        self.current_best_btn = QPushButton("Current Best")
        self.current_best_btn.clicked.connect(self._show_current_best)
        nav_layout.addWidget(self.current_best_btn)
        
        layout.addWidget(nav_group)
        
        # Create default model for visualization
        default_model = QuadraticPredictor([16, 8], 'relu', 0.0)
        
        self.viz_figure = Figure(figsize=(12, 8), facecolor=COLORS['bg_tertiary'])
        self.viz_canvas = FigureCanvas(self.viz_figure)
        self.viz_ax = self.viz_figure.add_subplot(111)
        
        visualize_model_architecture(default_model, self.viz_ax)
        self.viz_canvas.draw()
        
        layout.addWidget(self.viz_canvas)
        
        return widget
    
    def _create_tracker_tab(self):
        """Create interactive model performance analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Control panel
        control_group = QGroupBox("Visualization Options")
        control_group.setStyleSheet(f"""
            QGroupBox {{
                background-color: {COLORS['bg_secondary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 10px;
                padding: 15px;
                color: {COLORS['accent_primary']};
                font-weight: bold;
            }}
        """)
        control_layout = QHBoxLayout(control_group)
        
        control_layout.addWidget(QLabel("View:"))
        self.tracker_view_combo = QComboBox()
        self.tracker_view_combo.addItems(["Loss vs Complexity", "Loss vs Layer Count", "Loss vs Layer Size", "3D: Loss/Size/Layers"])
        self.tracker_view_combo.currentTextChanged.connect(self._update_tracker_graph)
        control_layout.addWidget(self.tracker_view_combo)
        
        control_layout.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._update_tracker_graph)
        control_layout.addWidget(refresh_btn)
        
        layout.addWidget(control_group)
        
        # Interactive graph
        self.tracker_figure = Figure(figsize=(12, 8), facecolor=COLORS['bg_tertiary'])
        self.tracker_canvas = FigureCanvas(self.tracker_figure)
        self.tracker_ax = self.tracker_figure.add_subplot(111)
        layout.addWidget(self.tracker_canvas)
        
        # Info label
        self.tracker_info = QLabel("No model data available yet. Run optimization to see performance analysis.")
        self.tracker_info.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 10px;")
        self.tracker_info.setWordWrap(True)
        layout.addWidget(self.tracker_info)
        
        return widget
    
    def _create_test_tab(self):
        """Create interactive testing tab with visual graph."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Input section
        input_group = QGroupBox("Test Quadratic Equation")
        input_group.setStyleSheet(f"""
            QGroupBox {{
                background-color: {COLORS['bg_secondary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 10px;
                padding: 15px;
                color: {COLORS['accent_primary']};
                font-weight: bold;
            }}
        """)
        input_layout = QHBoxLayout(input_group)
        
        from PyQt6.QtWidgets import QDoubleSpinBox
        input_layout.addWidget(QLabel("a:"))
        self.test_a = QDoubleSpinBox()
        self.test_a.setRange(-100, 100)
        self.test_a.setValue(1.0)
        self.test_a.valueChanged.connect(self._test_model)  # Auto-update on change
        input_layout.addWidget(self.test_a)
        
        input_layout.addWidget(QLabel("b:"))
        self.test_b = QDoubleSpinBox()
        self.test_b.setRange(-100, 100)
        self.test_b.setValue(0.0)
        self.test_b.valueChanged.connect(self._test_model)
        input_layout.addWidget(self.test_b)
        
        input_layout.addWidget(QLabel("c:"))
        self.test_c = QDoubleSpinBox()
        self.test_c.setRange(-100, 100)
        self.test_c.setValue(-1.0)
        self.test_c.valueChanged.connect(self._test_model)
        input_layout.addWidget(self.test_c)
        
        test_btn = QPushButton("Test Model")
        test_btn.clicked.connect(self._test_model)
        input_layout.addWidget(test_btn)
        
        layout.addWidget(input_group)
        
        # Visual graph
        self.test_figure = Figure(figsize=(10, 6), facecolor=COLORS['bg_tertiary'])
        self.test_canvas = FigureCanvas(self.test_figure)
        self.test_ax = self.test_figure.add_subplot(111)
        layout.addWidget(self.test_canvas)
        
        # Results display
        self.test_results = QTextEdit()
        self.test_results.setReadOnly(True)
        self.test_results.setMaximumHeight(150)
        self.test_results.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['accent_primary']};
                border-radius: 5px;
            }}
        """)
        layout.addWidget(self.test_results)
        
        return widget
    
    def _on_preset_changed(self, preset_name):
        preset = get_experiment_preset(preset_name)
        if preset:
            self.trials_spin.setValue(preset['n_trials'])
            self.epochs_spin.setValue(preset['epochs_per_trial'])
    
    def set_training_data(self, data):
        """Set training data from page 1."""
        self.training_data = data
        if data:
            self.status_text.append(f"✓ Training data loaded: {len(data['inputs'])} samples")
    
    def _start_optimization(self):
        """Start optimization."""
        if not self.training_data:
            self.status_text.append("✗ No training data! Generate data first.")
            return
        
        self.start_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        
        self.trial_losses = []
        self.architecture_history = []  # Reset history for new optimization
        self.current_arch_index = -1
        self.best_loss_so_far = float('inf')  # Reset best loss
        self.trial_params = []
        
        preset_name = self.preset_combo.currentText()
        search_space = get_search_space(get_experiment_preset(preset_name)['search_space'])
        method = self.method_combo.currentText()
        n_trials = self.trials_spin.value()
        self.n_trials = n_trials  # Store for progress bar
        epochs = self.epochs_spin.value()
        
        self.status_text.clear()
        self.status_text.append(f"Starting {method} optimization...")
        self.status_text.append(f"Trials: {n_trials}, Epochs: {epochs}")
        
        # Clear plot
        self.progress_ax.clear()
        self.progress_ax.set_title("Optimization Progress", color=COLORS['accent_primary'], fontsize=14, fontweight='bold')
        self.progress_ax.set_xlabel("Trial", color=COLORS['text_primary'])
        self.progress_ax.set_ylabel("Loss", color=COLORS['text_primary'])
        self.progress_ax.grid(True, alpha=0.3, color=COLORS['accent_primary'])
        self.progress_canvas.draw()
        
        # Start thread
        self.optimization_thread = OptimizationThread(
            self.training_data, search_space, method, n_trials, epochs
        )
        self.optimization_thread.progress.connect(self._on_progress)
        self.optimization_thread.trial_update.connect(self._on_trial_update)
        self.optimization_thread.finished.connect(self._on_optimization_finished)
        self.optimization_thread.start()
    
    def _on_progress(self, message):
        """Handle progress updates."""
        self.status_text.append(message)
    

    def _on_trial_update(self, trial_num, loss, params):
        """Handle real-time trial updates."""
        self.trial_losses.append({'trial': trial_num, 'loss': loss, 'params': params})
        
        # Check if this is a new best model
        if loss < self.best_loss_so_far:
            self.best_loss_so_far = loss
            # Add to architecture history
            arch_info = {
                'trial': trial_num,
                'loss': loss,
                'params': params.copy(),
                'timestamp': trial_num
            }
            self.architecture_history.append(arch_info)
            # Add to best models tracker
            model_info = {
                'loss': loss,
                'params': params.copy(),
                'layers': params.get('hidden_layers_count', 0),
                'size': params.get('hidden_layer_size', 0),
                'activation': params.get('activation', 'relu'),
                'trial': trial_num
            }
            # Calculate parameter count for complexity
            hidden_layers = [model_info['size']] * model_info['layers']
            temp_model = QuadraticPredictor(hidden_layers=hidden_layers,
                                           activation=model_info['activation'],
                                           dropout=params.get('dropout', 0.0))
            model_info['param_count'] = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
            
            self.best_models.append(model_info)
            self.best_models.sort(key=lambda x: x['loss'])
            self.best_models = self.best_models[:10]  # Keep top 10
            
            # Update performance graph
            self._update_tracker_graph()
            
            self.current_arch_index = len(self.architecture_history) - 1
            
            # Update model visualization in real-time
            self._update_model_viz(params)
        
        # Update progress plot in real-time
        if len(self.trial_losses) > 0:
            self.progress_ax.clear()
            from utils.visualization import plot_optimization_convergence
            plot_optimization_convergence(self.trial_losses, self.progress_ax)
            self.progress_canvas.draw()
        
        # Update progress bar
        self.progress.setMaximum(self.n_trials)
        self.progress.setValue(trial_num + 1)
    def _on_optimization_finished(self, results):
        """Handle optimization completion."""
        self.progress.setVisible(False)
        self.start_btn.setEnabled(True)
        
        if results:
            self.results = results
            self.status_text.append("✓ Optimization complete!")
            
            if 'best_params' in results:
                self.status_text.append("Best Parameters:")
                for key, value in results['best_params'].items():
                    self.status_text.append(f"  {key}: {value}")
                
                # Update model visualization
                self._update_model_viz(results['best_params'])
                
                # Add to best models tracker
                self._add_to_tracker(results)
                
                # Also populate best_models from all trial history to show all relevant models
                if self.trial_losses:
                    # Get top models from all trials (sorted by loss)
                    all_trial_models = []
                    for trial_data in self.trial_losses:
                        if 'params' in trial_data and 'loss' in trial_data:
                            params = trial_data['params']
                            model_info = {
                                'loss': trial_data['loss'],
                                'params': params.copy(),
                                'layers': params.get('hidden_layers_count', 0),
                                'size': params.get('hidden_layer_size', 0),
                                'activation': params.get('activation', 'relu'),
                                'trial': trial_data.get('trial', 0)
                            }
                            # Calculate parameter count
                            hidden_layers = [model_info['size']] * model_info['layers']
                            temp_model = QuadraticPredictor(hidden_layers=hidden_layers,
                                                           activation=model_info['activation'],
                                                           dropout=params.get('dropout', 0.0))
                            model_info['param_count'] = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
                            all_trial_models.append(model_info)
                    
                    # Sort by loss and keep top 10 (by performance, not just unique architectures)
                    all_trial_models.sort(key=lambda x: x['loss'])
                    # Merge with existing best_models and keep top 10 overall
                    combined = self.best_models + all_trial_models
                    # Remove duplicates based on (layers, size, activation, loss) - keep best loss for each architecture
                    seen = {}
                    for model in combined:
                        arch_key = (model['layers'], model['size'], model['activation'])
                        if arch_key not in seen or model['loss'] < seen[arch_key]['loss']:
                            seen[arch_key] = model
                    
                    # Get top 10 by loss
                    unique_models = list(seen.values())
                    unique_models.sort(key=lambda x: x['loss'])
                    self.best_models = unique_models[:10]
                    self._update_tracker_graph()
            
            if 'best_value' in results:
                self.status_text.append(f"Best Loss: {results['best_value']:.6f}")
            
            # Add trial history to results
            if self.trial_losses:
                results['trial_history'] = self.trial_losses
            
            # Emit signal
            self.optimization_complete.emit(results)
        else:
            self.status_text.append("✗ Optimization failed!")
    
    def _update_model_viz(self, params):
        """Update model visualization with best params."""
        hidden_layers = [params.get('hidden_layer_size', 16)] * params.get('hidden_layers_count', 2)
        model = QuadraticPredictor(
            hidden_layers=hidden_layers,
            activation=params.get('activation', 'relu'),
            dropout=params.get('dropout', 0.0)
        )
        
        self.viz_ax.clear()
        visualize_model_architecture(model, self.viz_ax)
        self.viz_canvas.draw()
        
        # Update info label and navigation buttons if available
        if hasattr(self, 'arch_info_label'):
            self._update_arch_info_label()
            self._update_nav_buttons()
    def _show_previous_architecture(self):
        """Show previous architecture in history."""
        if self.architecture_history and self.current_arch_index > 0:
            self.current_arch_index -= 1
            arch = self.architecture_history[self.current_arch_index]
            self._update_model_viz(arch['params'])
            self._update_arch_info_label()
            self._update_nav_buttons()
    
    def _show_next_architecture(self):
        """Show next architecture in history."""
        if self.architecture_history and self.current_arch_index < len(self.architecture_history) - 1:
            self.current_arch_index += 1
            arch = self.architecture_history[self.current_arch_index]
            self._update_model_viz(arch['params'])
            self._update_arch_info_label()
            self._update_nav_buttons()
    
    def _show_current_best(self):
        """Show current best architecture."""
        if self.architecture_history:
            self.current_arch_index = len(self.architecture_history) - 1
            arch = self.architecture_history[self.current_arch_index]
            self._update_model_viz(arch['params'])
            self._update_arch_info_label()
            self._update_nav_buttons()
    
    def _update_arch_info_label(self):
        """Update architecture info label."""
        if self.architecture_history and 0 <= self.current_arch_index < len(self.architecture_history):
            arch = self.architecture_history[self.current_arch_index]
            info = f"Trial {arch['trial']} | Loss: {arch['loss']:.6f} | "
            info += f"Layers: {arch['params'].get('hidden_layers_count', 0)} × {arch['params'].get('hidden_layer_size', 0)}"
            info += f" | {arch['params'].get('activation', 'relu').upper()}"
            self.arch_info_label.setText(f"{self.current_arch_index + 1} / {len(self.architecture_history)} - {info}")
        else:
            self.arch_info_label.setText("No architecture history yet")
    
    def _update_nav_buttons(self):
        """Update navigation button states."""
        if hasattr(self, 'prev_btn') and hasattr(self, 'next_btn'):
            self.prev_btn.setEnabled(self.architecture_history and self.current_arch_index > 0)
            self.next_btn.setEnabled(self.architecture_history and self.current_arch_index < len(self.architecture_history) - 1)
    
    
    
    def _add_to_tracker(self, results):
        """Add model to best models tracker."""
        if 'best_params' in results and 'best_value' in results:
            model_info = {
                'loss': results['best_value'],
                'params': results['best_params'],
                'layers': results['best_params'].get('hidden_layers_count', 0),
                'size': results['best_params'].get('hidden_layer_size', 0),
                'activation': results['best_params'].get('activation', 'relu')
            }
            
            # Calculate parameter count for complexity
            hidden_layers = [model_info['size']] * model_info['layers']
            temp_model = QuadraticPredictor(hidden_layers=hidden_layers,
                                           activation=model_info['activation'],
                                           dropout=results['best_params'].get('dropout', 0.0))
            model_info['param_count'] = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
            
            self.best_models.append(model_info)
            self.best_models.sort(key=lambda x: x['loss'])
            self.best_models = self.best_models[:10]  # Keep top 10
            
            # Update graph
            self._update_tracker_graph()
    
    def _update_tracker_graph(self):
        """Update the interactive performance analysis graph."""
        if not hasattr(self, 'tracker_ax') or not self.best_models:
            if hasattr(self, 'tracker_ax'):
                self.tracker_ax.clear()
                self.tracker_ax.text(0.5, 0.5, 'No model data available yet.\nRun optimization to see performance analysis.',
                                  ha='center', va='center', transform=self.tracker_ax.transAxes,
                                  color=COLORS['text_secondary'], fontsize=12)
                self.tracker_ax.axis('off')
                if hasattr(self, 'tracker_canvas'):
                    self.tracker_canvas.draw()
            return
        
        view_type = self.tracker_view_combo.currentText() if hasattr(self, 'tracker_view_combo') else "Loss vs Complexity"

        # Check if we need to switch between 2D and 3D
        is_3d_view = view_type == "3D: Loss/Size/Layers"
        is_currently_3d = hasattr(self.tracker_ax, 'zaxis')  # zaxis is unique to 3D axes
        
        # If switching between 2D and 3D, recreate axes properly
        if is_3d_view != is_currently_3d:
            # Remove old axes
            self.tracker_ax.remove()
            # Create new axes with appropriate projection
            if is_3d_view:
                from mpl_toolkits.mplot3d import Axes3D
                self.tracker_ax = self.tracker_figure.add_subplot(111, projection='3d')
            else:
                self.tracker_ax = self.tracker_figure.add_subplot(111)
        else:
            # Just clear if same type
            self.tracker_ax.clear()
        
        
        if view_type == "Loss vs Complexity":
            # Scatter plot: Loss vs Parameter Count
            param_counts = [m['param_count'] for m in self.best_models]
            losses = [m['loss'] for m in self.best_models]
            
            scatter = self.tracker_ax.scatter(param_counts, losses, s=200, alpha=0.7,
                                            c=range(len(self.best_models)), cmap='viridis',
                                            edgecolors=COLORS['accent_primary'], linewidths=2)
            
            # Add labels for top 3
            for i, model in enumerate(self.best_models[:3]):
                self.tracker_ax.annotate(f"#{i+1}\n{model['layers']}×{model['size']}",
                                       (model['param_count'], model['loss']),
                                       xytext=(10, 10), textcoords='offset points',
                                       bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_secondary'],
                                                edgecolor=COLORS['accent_primary'], alpha=0.8),
                                       color=COLORS['text_primary'], fontsize=9)
            
            self.tracker_ax.set_xlabel('Model Complexity (Parameters)', color=COLORS['text_primary'], fontsize=11)
            self.tracker_ax.set_ylabel('Validation Loss', color=COLORS['text_primary'], fontsize=11)
            self.tracker_ax.set_title('Loss vs Model Complexity', color=COLORS['accent_primary'],
                                    fontsize=14, fontweight='bold')
            
        elif view_type == "Loss vs Layer Count":
            layer_counts = [m['layers'] for m in self.best_models]
            losses = [m['loss'] for m in self.best_models]
            
            self.tracker_ax.scatter(layer_counts, losses, s=200, alpha=0.7,
                                  color=COLORS['accent_primary'], edgecolors='white', linewidths=2)
            
            self.tracker_ax.set_xlabel('Number of Hidden Layers', color=COLORS['text_primary'], fontsize=11)
            self.tracker_ax.set_ylabel('Validation Loss', color=COLORS['text_primary'], fontsize=11)
            self.tracker_ax.set_title('Loss vs Layer Count', color=COLORS['accent_primary'],
                                    fontsize=14, fontweight='bold')
            
        elif view_type == "Loss vs Layer Size":
            layer_sizes = [m['size'] for m in self.best_models]
            losses = [m['loss'] for m in self.best_models]
            
            self.tracker_ax.scatter(layer_sizes, losses, s=200, alpha=0.7,
                                  color=COLORS['accent_success'], edgecolors='white', linewidths=2)
            
            self.tracker_ax.set_xlabel('Hidden Layer Size', color=COLORS['text_primary'], fontsize=11)
            self.tracker_ax.set_ylabel('Validation Loss', color=COLORS['text_primary'], fontsize=11)
            self.tracker_ax.set_title('Loss vs Layer Size', color=COLORS['accent_primary'],
                                    fontsize=14, fontweight='bold')
            
        elif view_type == "3D: Loss/Size/Layers":
            
            layer_counts = [m['layers'] for m in self.best_models]
            layer_sizes = [m['size'] for m in self.best_models]
            losses = [m['loss'] for m in self.best_models]
            
            scatter = self.tracker_ax.scatter(layer_counts, layer_sizes, losses, s=200, alpha=0.7,
                                            c=losses, cmap='viridis', edgecolors='white', linewidths=2)
            
            self.tracker_ax.set_xlabel('Layer Count', color=COLORS['text_primary'], fontsize=10)
            self.tracker_ax.set_ylabel('Layer Size', color=COLORS['text_primary'], fontsize=10)
            self.tracker_ax.set_zlabel('Loss', color=COLORS['text_primary'], fontsize=10)
            self.tracker_ax.set_title('3D: Loss vs Architecture', color=COLORS['accent_primary'],
                                     fontsize=14, fontweight='bold')
            
            # Remove any existing colorbars before adding new one
            for cbar in self.tracker_figure.axes:
                if cbar is not self.tracker_ax:
                    cbar.remove()
            
            self.tracker_figure.colorbar(scatter, ax=self.tracker_ax, label='Loss')
        
        # Apply styling (only for 2D axes - 3D handles this differently)
        if not is_3d_view:
            self.tracker_ax.tick_params(colors=COLORS['text_primary'])
            self.tracker_ax.grid(True, alpha=0.3, color=COLORS['accent_primary'])
            self.tracker_ax.set_facecolor(COLORS['bg_tertiary'])
            for spine in self.tracker_ax.spines.values():
                spine.set_color(COLORS['text_primary'])
        else:
            # 3D axes styling
            self.tracker_ax.tick_params(colors=COLORS['text_primary'])
            self.tracker_ax.xaxis.pane.fill = False
            self.tracker_ax.yaxis.pane.fill = False
            self.tracker_ax.zaxis.pane.fill = False
            self.tracker_ax.xaxis.pane.set_edgecolor(COLORS['bg_tertiary'])
            self.tracker_ax.yaxis.pane.set_edgecolor(COLORS['bg_tertiary'])
            self.tracker_ax.zaxis.pane.set_edgecolor(COLORS['bg_tertiary'])
            self.tracker_ax.set_facecolor(COLORS['bg_tertiary'])
        
        # Update info
        best = self.best_models[0]
        info_text = f"Best Model: {best['layers']} layers × {best['size']} units, "
        info_text += f"Loss: {best['loss']:.6f}, Parameters: {best['param_count']:,}"
        if hasattr(self, 'tracker_info'):
            self.tracker_info.setText(info_text)
        
        self.tracker_canvas.draw()
    
    def _test_model(self):
        """Test model with user input and visual graph."""
        if not self.results or 'best_params' not in self.results:
            self.test_results.setText("No trained model available. Run optimization first.")
            self.test_ax.clear()
            self.test_ax.text(0.5, 0.5, "No model available", ha='center', va='center',
                            transform=self.test_ax.transAxes, color=COLORS['text_secondary'])
            self.test_canvas.draw()
            return
        
        a = self.test_a.value()
        b = self.test_b.value()
        c = self.test_c.value()
        
        # Create model with best params
        params = self.results['best_params']
        hidden_layers = [params.get('hidden_layer_size', 16)] * params.get('hidden_layers_count', 2)
        model = QuadraticPredictor(
            hidden_layers=hidden_layers,
            activation=params.get('activation', 'relu'),
            dropout=params.get('dropout', 0.0)
        )
        
        model.eval()
        
        # Predict
        input_tensor = torch.FloatTensor([[a, b, c]])
        with torch.no_grad():
            output = model(input_tensor)
        
        r1_real, r2_real, r1_imag, r2_imag = output[0].numpy()
        pred_roots = (r1_real, r2_real, r1_imag, r2_imag)
        
        # Calculate true roots
        from utils.data import solve_quadratic
        true_root1, true_root2, disc_type = solve_quadratic(a, b, c)
        true_roots = (true_root1, true_root2)
        
        # Plot with roots
        self.test_ax.clear()
        from utils.visualization import plot_quadratic_with_roots
        plot_quadratic_with_roots(a, b, c, true_roots, pred_roots, self.test_ax)
        self.test_canvas.draw()
        
        # Format results
        result_text = f"Equation: {a}x² + {b}x + {c} = 0\n\n"
        result_text += f"True Roots: {disc_type}\n"
        if disc_type == "Two real roots":
            result_text += f"  Root 1: {true_root1.real:.6f}\n"
            result_text += f"  Root 2: {true_root2.real:.6f}\n"
        elif disc_type == "One real root":
            result_text += f"  Root: {true_root1.real:.6f}\n"
        else:
            result_text += f"  Root 1: {true_root1:.6f}\n"
            result_text += f"  Root 2: {true_root2:.6f}\n"
        
        result_text += f"\nPredicted Roots:\n"
        if abs(r1_imag) < 1e-6 and abs(r2_imag) < 1e-6:
            result_text += f"  Root 1: {r1_real:.6f}\n"
            result_text += f"  Root 2: {r2_real:.6f}\n"
        else:
            result_text += f"  Root 1: {r1_real:.6f} + {r1_imag:.6f}i\n"
            result_text += f"  Root 2: {r2_real:.6f} + {r2_imag:.6f}i\n"
        
        # Calculate error
        if disc_type == "Two real roots":
            error1 = abs(r1_real - true_root1.real)
            error2 = abs(r2_real - true_root2.real)
            result_text += f"\nErrors:\n"
            result_text += f"  Root 1 error: {error1:.6f}\n"
            result_text += f"  Root 2 error: {error2:.6f}\n"
            avg_error = (error1 + error2) / 2
            result_text += f"  Average error: {avg_error:.6f}\n"
        
        self.test_results.setText(result_text)
