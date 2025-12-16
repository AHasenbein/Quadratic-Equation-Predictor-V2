"""
Page 1: Data Generation
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QSpinBox, QDoubleSpinBox, QGroupBox, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from config import COLORS
from utils.data import generate_data, generate_stratified_data


class DataGenerationPage(QWidget):
    """Data generation page."""
    
    data_generated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.training_data = None
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Data Generation")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['accent_primary']};")
        layout.addWidget(title)
        
        # Main content
        content_layout = QHBoxLayout()
        
        # Left: Controls
        controls_group = QGroupBox("Generation Settings")
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
        
        # Strategy selection
        strategy_label = QLabel("Data Strategy:")
        strategy_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        controls_layout.addWidget(strategy_label)
        
        self.strategy_btn1 = QPushButton("Uniform")
        self.strategy_btn2 = QPushButton("Stratified")
        self.strategy_btn2.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                color: {COLORS['bg_primary']};
            }}
        """)
        self.current_strategy = 'stratified'
        
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(self.strategy_btn1)
        strategy_layout.addWidget(self.strategy_btn2)
        controls_layout.addLayout(strategy_layout)
        
        self.strategy_btn1.clicked.connect(lambda: self._set_strategy('uniform'))
        self.strategy_btn2.clicked.connect(lambda: self._set_strategy('stratified'))
        
        # Sample count
        samples_label = QLabel("Number of Samples:")
        samples_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        controls_layout.addWidget(samples_label)
        
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1000, 100000)
        self.samples_spin.setValue(10000)
        self.samples_spin.setSingleStep(1000)
        controls_layout.addWidget(self.samples_spin)
        
        # Coefficient ranges
        range_label = QLabel("Coefficient Ranges:")
        range_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        controls_layout.addWidget(range_label)
        
        for name, default in [('a', -10), ('b', -10), ('c', -10)]:
            range_layout = QHBoxLayout()
            range_layout.addWidget(QLabel(f"{name}:"))
            spin_min = QDoubleSpinBox()
            spin_min.setRange(-100, 100)
            spin_min.setValue(default)
            spin_max = QDoubleSpinBox()
            spin_max.setRange(-100, 100)
            spin_max.setValue(10)
            range_layout.addWidget(spin_min)
            range_layout.addWidget(QLabel("to"))
            range_layout.addWidget(spin_max)
            controls_layout.addLayout(range_layout)
            setattr(self, f'{name}_min', spin_min)
            setattr(self, f'{name}_max', spin_max)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Data")
        self.generate_btn.setStyleSheet(f"""
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
        """)
        self.generate_btn.clicked.connect(self._generate_data)
        controls_layout.addWidget(self.generate_btn)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        controls_layout.addWidget(self.progress)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        controls_layout.addWidget(self.status_label)
        
        controls_layout.addStretch()
        
        # Right: Visualization
        self.figure = Figure(figsize=(8, 6), facecolor=COLORS['bg_tertiary'])
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(COLORS['bg_tertiary'])
        self.ax.tick_params(colors=COLORS['text_primary'])
        for spine in self.ax.spines.values():
            spine.set_color(COLORS['text_primary'])
        self.ax.set_title("Data Distribution", color=COLORS['accent_primary'], fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Coefficient Value", color=COLORS['text_primary'])
        self.ax.set_ylabel("Frequency", color=COLORS['text_primary'])
        
        content_layout.addWidget(controls_group, 1)
        content_layout.addWidget(self.canvas, 2)
        
        layout.addLayout(content_layout)
    
    def _set_strategy(self, strategy):
        self.current_strategy = strategy
        if strategy == 'uniform':
            self.strategy_btn1.setStyleSheet(f"background-color: {COLORS['accent_primary']}; color: {COLORS['bg_primary']};")
            self.strategy_btn2.setStyleSheet("")
        else:
            self.strategy_btn2.setStyleSheet(f"background-color: {COLORS['accent_primary']}; color: {COLORS['bg_primary']};")
            self.strategy_btn1.setStyleSheet("")
    
    def _generate_data(self):
        """Generate training data."""
        self.generate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        
        num_samples = self.samples_spin.value()
        a_range = (self.a_min.value(), self.a_max.value())
        b_range = (self.b_min.value(), self.b_max.value())
        c_range = (self.c_min.value(), self.c_max.value())
        
        try:
            if self.current_strategy == 'stratified':
                data = generate_stratified_data(num_samples, a_range, b_range, c_range)
            else:
                data = generate_data(num_samples, a_range, b_range, c_range)
            
            self.training_data = data
            
            # Update visualization
            self.ax.clear()
            self.ax.hist(data['inputs'][:, 0], bins=50, alpha=0.7, label='a', color=COLORS['accent_primary'])
            self.ax.hist(data['inputs'][:, 1], bins=50, alpha=0.7, label='b', color=COLORS['accent_success'])
            self.ax.hist(data['inputs'][:, 2], bins=50, alpha=0.7, label='c', color=COLORS['accent_warning'])
            self.ax.legend()
            self.ax.set_title(f"Data Distribution ({len(data['inputs'])} samples)", 
                            color=COLORS['accent_primary'], fontsize=14, fontweight='bold')
            self.canvas.draw()
            
            self.status_label.setText(f"✓ Generated {len(data['inputs'])} samples")
            self.status_label.setStyleSheet(f"color: {COLORS['accent_success']};")
            
            # Emit signal
            self.data_generated.emit(data)
            
        except Exception as e:
            self.status_label.setText(f"✗ Error: {str(e)}")
            self.status_label.setStyleSheet(f"color: {COLORS['accent_error']};")
        finally:
            self.progress.setVisible(False)
            self.generate_btn.setEnabled(True)
