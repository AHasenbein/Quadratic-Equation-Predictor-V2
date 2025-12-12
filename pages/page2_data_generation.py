"""
Page 2: Data Generation, Model Presets, and Visualization.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                             QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
                             QScrollArea, QFrame, QGroupBox, QFormLayout, QSizePolicy,
                             QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from pages.base_page import BasePage
from config import COLORS, MODEL_PRESETS
from utils.data_generator import generate_training_data, generate_training_data_around_equation
from utils.graph_styling import apply_standard_graph_style
from utils.widget_styling import get_groupbox_style, get_button_style, get_input_style


class DataGenerationThread(QThread):
    """Thread for generating data without blocking UI."""
    data_generated = pyqtSignal(dict)
    
    def __init__(self, num_samples, a_range, b_range, c_range, center_around=None):
        super().__init__()
        self.num_samples = num_samples
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.center_around = center_around  # (a, b, c) tuple to center around
    
    def run(self):
        if self.center_around:
            # Generate data centered around the equation
            data = generate_training_data_around_equation(
                self.num_samples,
                self.center_around,
                self.a_range,
                self.b_range,
                self.c_range
            )
        else:
            # Normal generation
            data = generate_training_data(
                self.num_samples,
                self.a_range,
                self.b_range,
                self.c_range
            )
        self.data_generated.emit(data)


class DataGenerationPage(BasePage):
    """Page for data generation and model presets."""
    
    def __init__(self, parent=None):
        # Initialize instance variables before calling super().__init__()
        self.generated_data = None
        super().__init__(parent)
    
    def setup_ui(self):
        """Setup the data generation UI."""
        super().setup_ui()  # Initialize base layout
        # Title
        title = QLabel("Data Generation & Model Configuration")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {COLORS['accent_primary']}; padding: 10px;")
        self.layout.addWidget(title)
        
        # Content layout (no nested scroll - using global scroll from base_page)
        content_layout = self.layout
        
        # Data Generation Section
        data_group = QGroupBox("Data Generation")
        data_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        data_group.setStyleSheet(get_groupbox_style())
        data_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        data_layout = QFormLayout(data_group)
        
        # Number of samples
        self.num_samples = QSpinBox()
        self.num_samples.setRange(100, 1000000)
        self.num_samples.setValue(10000)
        self.num_samples.setStyleSheet(self.get_spinbox_style())
        data_layout.addRow("Number of Samples:", self.num_samples)
        
        # Coefficient ranges
        self.a_min = QDoubleSpinBox()
        self.a_min.setRange(-100, 100)
        self.a_min.setValue(-10)
        self.a_min.setStyleSheet(self.get_spinbox_style())
        
        self.a_max = QDoubleSpinBox()
        self.a_max.setRange(-100, 100)
        self.a_max.setValue(10)
        self.a_max.setStyleSheet(self.get_spinbox_style())
        
        a_range_layout = QHBoxLayout()
        a_range_layout.addWidget(self.a_min)
        a_range_layout.addWidget(QLabel("to"))
        a_range_layout.addWidget(self.a_max)
        data_layout.addRow("Coefficient 'a' Range:", a_range_layout)
        
        self.b_min = QDoubleSpinBox()
        self.b_min.setRange(-100, 100)
        self.b_min.setValue(-10)
        self.b_min.setStyleSheet(self.get_spinbox_style())
        
        self.b_max = QDoubleSpinBox()
        self.b_max.setRange(-100, 100)
        self.b_max.setValue(10)
        self.b_max.setStyleSheet(self.get_spinbox_style())
        
        b_range_layout = QHBoxLayout()
        b_range_layout.addWidget(self.b_min)
        b_range_layout.addWidget(QLabel("to"))
        b_range_layout.addWidget(self.b_max)
        data_layout.addRow("Coefficient 'b' Range:", b_range_layout)
        
        self.c_min = QDoubleSpinBox()
        self.c_min.setRange(-100, 100)
        self.c_min.setValue(-10)
        self.c_min.setStyleSheet(self.get_spinbox_style())
        
        self.c_max = QDoubleSpinBox()
        self.c_max.setRange(-100, 100)
        self.c_max.setValue(10)
        self.c_max.setStyleSheet(self.get_spinbox_style())
        
        c_range_layout = QHBoxLayout()
        c_range_layout.addWidget(self.c_min)
        c_range_layout.addWidget(QLabel("to"))
        c_range_layout.addWidget(self.c_max)
        data_layout.addRow("Coefficient 'c' Range:", c_range_layout)
        
        # Load equation from page 1 button
        load_eq_btn = QPushButton("Load Equation from Page 1")
        load_eq_btn.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        load_eq_btn.setStyleSheet(self.get_button_style(primary=False))
        load_eq_btn.clicked.connect(self.load_equation_from_page1)
        data_layout.addRow("", load_eq_btn)
        
        # Checkbox to use equation as center
        self.use_equation_center = QCheckBox("Generate data around equation from Page 1")
        self.use_equation_center.setFont(QFont("Arial", 10))
        self.use_equation_center.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_primary']};
                padding: 5px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 3px;
                background-color: {COLORS['bg_tertiary']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['accent_primary']};
            }}
        """)
        self.use_equation_center.setToolTip("If checked, generates data centered around the equation from Page 1 with variation")
        data_layout.addRow("", self.use_equation_center)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Data")
        self.generate_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.generate_btn.setStyleSheet(self.get_button_style())
        self.generate_btn.clicked.connect(self.generate_data)
        data_layout.addRow("", self.generate_btn)
        
        content_layout.addWidget(data_group)
        
        # Model Presets Section
        preset_group = QGroupBox("Model Presets")
        preset_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        preset_group.setStyleSheet(data_group.styleSheet())
        preset_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        preset_layout = QVBoxLayout(preset_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(MODEL_PRESETS.keys()))
        self.preset_combo.setStyleSheet(self.get_combobox_style())
        self.preset_combo.currentTextChanged.connect(self.update_preset_display)
        preset_layout.addWidget(self.preset_combo)
        
        self.preset_display = QLabel()
        self.preset_display.setFont(QFont("Arial", 10))
        self.preset_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.preset_display.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_tertiary']};
                border-radius: 5px;
                padding: 10px;
                color: {COLORS['text_primary']};
            }}
        """)
        self.preset_display.setWordWrap(True)
        preset_layout.addWidget(self.preset_display)
        self.update_preset_display()
        
        content_layout.addWidget(preset_group)
        
        # Visualization Section
        viz_group = QGroupBox("Data Visualization")
        viz_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        viz_group.setStyleSheet(data_group.styleSheet())
        viz_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viz_layout = QVBoxLayout(viz_group)
        
        # Matplotlib figure for data distribution with minimum size and resizable
        self.figure = Figure(figsize=(12, 8), facecolor=COLORS['bg_tertiary'])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(500)
        self.canvas.setMinimumWidth(800)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setStyleSheet(f"background-color: {COLORS['bg_tertiary']};")
        viz_layout.addWidget(self.canvas)
        
        content_layout.addWidget(viz_group)
    
    def get_spinbox_style(self):
        return get_input_style()
    
    def get_combobox_style(self):
        return get_input_style()
    
    def get_button_style(self, primary=True):
        return get_button_style(primary=primary)
    
    def update_preset_display(self):
        """Update the preset display when selection changes."""
        preset_name = self.preset_combo.currentText()
        preset = MODEL_PRESETS[preset_name]
        text = f"Hidden Layers: {preset['hidden_layers']}\n"
        text += f"Learning Rate: {preset['learning_rate']}\n"
        text += f"Batch Size: {preset['batch_size']}\n"
        text += f"Epochs: {preset['epochs']}"
        self.preset_display.setText(text)
    
    def load_equation_from_page1(self):
        """Load equation from page 1 and set ranges centered around it."""
        # Get equation from page 1 using helper method
        main_window = self.get_main_window()
        if not main_window or not hasattr(main_window, 'page1'):
            return
        
        coeffs = main_window.page1.get_current_coefficients()
        if coeffs is None:
            return
        
        a, b, c = coeffs
        
        # Set ranges centered around the equation with some variation
        variation = 5.0  # Default variation range
        
        self.a_min.setValue(a - variation)
        self.a_max.setValue(a + variation)
        self.b_min.setValue(b - variation)
        self.b_max.setValue(b + variation)
        self.c_min.setValue(c - variation)
        self.c_max.setValue(c + variation)
        
        # Check the checkbox to use equation as center
        self.use_equation_center.setChecked(True)
    
    def generate_data(self):
        """Generate training data."""
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")
        
        num_samples = self.num_samples.value()
        
        # Check if we should use equation from page 1 as center
        if self.use_equation_center.isChecked():
            # Get equation from page 1 using helper method
            main_window = self.get_main_window()
            if main_window and hasattr(main_window, 'page1'):
                coeffs = main_window.page1.get_current_coefficients()
                if coeffs is not None:
                    a_center, b_center, c_center = coeffs
                    # Use ranges but generate data centered around the equation
                    a_range = (self.a_min.value(), self.a_max.value())
                    b_range = (self.b_min.value(), self.b_max.value())
                    c_range = (self.c_min.value(), self.c_max.value())
                    
                    # Generate in thread with equation center
                    self.gen_thread = DataGenerationThread(
                        num_samples, a_range, b_range, c_range,
                        center_around=(a_center, b_center, c_center)
                    )
                    self.gen_thread.data_generated.connect(self.on_data_generated)
                    self.gen_thread.start()
                    return
        
        # Normal generation without equation center
        a_range = (self.a_min.value(), self.a_max.value())
        b_range = (self.b_min.value(), self.b_max.value())
        c_range = (self.c_min.value(), self.c_max.value())
        
        # Generate in thread
        self.gen_thread = DataGenerationThread(num_samples, a_range, b_range, c_range)
        self.gen_thread.data_generated.connect(self.on_data_generated)
        self.gen_thread.start()
    
    def on_data_generated(self, data):
        """Handle generated data."""
        self.generated_data = data
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate Data")
        self.visualize_data(data)
    
    def visualize_data(self, data):
        """Visualize the generated data."""
        self.figure.clear()
        
        inputs = data['inputs']
        targets = data['targets']
        
        # Create subplots
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)
        
        axes = [ax1, ax2, ax3, ax4]
        titles = ['Coefficient a', 'Coefficient b', 'Coefficient c', 'Root Distribution']
        
        for ax, title in zip(axes[:3], titles[:3]):
            ax.hist(inputs[:, axes.index(ax)], bins=50, color=COLORS['graph_line'], 
                   alpha=0.7, edgecolor=COLORS['accent_primary'], linewidth=0.5)
            apply_standard_graph_style(ax, title=title, xlabel='Value', ylabel='Frequency')
        
        # Root distribution
        real_roots = targets[:, 0]  # First real root
        ax4.hist(real_roots, bins=50, color=COLORS['accent_success'], 
               alpha=0.7, edgecolor=COLORS['accent_primary'], linewidth=0.5)
        apply_standard_graph_style(ax4, title='Root Distribution (Real Part)', 
                                  xlabel='Root Value', ylabel='Frequency')
        
        self.figure.set_constrained_layout(True)
        self.canvas.draw()
    
    def get_generated_data(self):
        """Get the generated data."""
        return self.generated_data
    
    def get_selected_preset(self):
        """Get the selected model preset."""
        preset_name = self.preset_combo.currentText()
        return MODEL_PRESETS[preset_name]

