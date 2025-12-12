"""
Page 1: Equation Entry with LaTeX display and graph visualization.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QScrollArea, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# matplotlib.pyplot not needed - using Figure directly
from sympy import symbols, latex, sympify, simplify
import numpy as np
from pages.base_page import BasePage
from config import COLORS, GRAPH_LINE_WIDTH
from utils.quadratic_utils import parse_equation, generate_equation_data, solve_quadratic
from utils.graph_styling import apply_standard_graph_style, style_legend, add_zero_lines
from utils.widget_styling import get_button_style, get_input_style


class EquationPage(BasePage):
    """Page for entering and visualizing quadratic equations."""
    
    def __init__(self, parent=None):
        # Initialize instance variables before calling super().__init__()
        self.current_coeffs = None
        super().__init__(parent)
    
    def setup_ui(self):
        """Setup the equation entry UI."""
        super().setup_ui()  # Initialize base layout
        # Title
        title = QLabel("Quadratic Equation Predictor")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent_primary']};
                padding: 10px;
            }}
        """)
        self.layout.addWidget(title)
        
        # Equation input section
        input_frame = QFrame()
        input_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        input_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        input_layout = QVBoxLayout(input_frame)
        
        input_label = QLabel("Enter Quadratic Equation:")
        input_label.setFont(QFont("Arial", 12))
        input_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        input_layout.addWidget(input_label)
        
        # Input field
        self.equation_input = QLineEdit()
        self.equation_input.setPlaceholderText("e.g., x^2 + 2x + 1 or 1 2 1 (a b c)")
        self.equation_input.setFont(QFont("Arial", 11))
        self.equation_input.setStyleSheet(get_input_style())
        self.equation_input.returnPressed.connect(self.process_equation)
        input_layout.addWidget(self.equation_input)
        
        # Process button
        process_btn = QPushButton("Process Equation")
        process_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        process_btn.setStyleSheet(get_button_style(primary=True))
        process_btn.clicked.connect(self.process_equation)
        input_layout.addWidget(process_btn)
        
        self.layout.addWidget(input_frame)
        
        # LaTeX display section
        latex_frame = QFrame()
        latex_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        latex_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        latex_layout = QVBoxLayout(latex_frame)
        
        latex_label = QLabel("Equation Display:")
        latex_label.setFont(QFont("Arial", 12))
        latex_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        latex_layout.addWidget(latex_label)
        
        self.latex_display = QLabel("Enter an equation to see it displayed here")
        self.latex_display.setFont(QFont("Arial", 16))
        self.latex_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.latex_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.latex_display.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_tertiary']};
                border-radius: 5px;
                padding: 20px;
                color: {COLORS['accent_primary']};
                min-height: 60px;
            }}
        """)
        latex_layout.addWidget(self.latex_display)
        
        self.layout.addWidget(latex_frame)
        
        # Graph section with scroll area
        graph_scroll = QScrollArea()
        graph_scroll.setWidgetResizable(False)  # Important: set to False for proper scrolling
        graph_scroll.setMinimumHeight(550)  # Set minimum height for scroll area
        graph_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        graph_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        graph_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        graph_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 10px;
                border: none;
            }}
        """)
        
        graph_widget = QWidget()
        graph_widget.setMinimumSize(900, 600)  # Set minimum size for the widget
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(10, 10, 10, 10)
        
        graph_label = QLabel("Graph Visualization:")
        graph_label.setFont(QFont("Arial", 12))
        graph_label.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 10px;")
        graph_layout.addWidget(graph_label)
        
        # Matplotlib figure - make it larger with minimum height and resizable
        self.figure = Figure(figsize=(12, 8), facecolor=COLORS['bg_tertiary'])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(500)  # Set minimum height
        self.canvas.setMinimumWidth(800)  # Set minimum width
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setStyleSheet(f"background-color: {COLORS['bg_tertiary']};")
        graph_layout.addWidget(self.canvas)
        
        graph_scroll.setWidget(graph_widget)
        self.layout.addWidget(graph_scroll)
        
        # Results display
        self.results_label = QLabel("")
        self.results_label.setFont(QFont("Arial", 11))
        self.results_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.results_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 10px;
                padding: 15px;
                color: {COLORS['text_primary']};
            }}
        """)
        self.results_label.setWordWrap(True)
        self.layout.addWidget(self.results_label)
    
    def process_equation(self):
        """Process the entered equation."""
        equation_str = self.equation_input.text().strip()
        if not equation_str:
            return
        
        # Parse equation
        coeffs = parse_equation(equation_str)
        if coeffs is None:
            self.latex_display.setText("Invalid equation format")
            self.latex_display.setStyleSheet(f"""
                QLabel {{
                    background-color: {COLORS['bg_tertiary']};
                    border-radius: 5px;
                    padding: 20px;
                    color: {COLORS['accent_error']};
                    min-height: 60px;
                }}
            """)
            return
        
        a, b, c = coeffs
        self.current_coeffs = coeffs
        
        # Display equation with nice formatting
        try:
            # Format the equation nicely
            parts = []
            if a != 0:
                if abs(a) == 1:
                    parts.append("x²" if a > 0 else "-x²")
                else:
                    parts.append(f"{a}x²" if a > 0 else f"{a}x²")
            
            if b != 0:
                sign = "+" if b > 0 and parts else ""
                if abs(b) == 1:
                    parts.append(f"{sign}x" if b > 0 else "-x")
                else:
                    parts.append(f"{sign}{b}x" if b > 0 else f"{b}x")
            
            if c != 0:
                sign = "+" if c > 0 and parts else ""
                parts.append(f"{sign}{c}" if c > 0 else str(c))
            
            if not parts:
                eq_text = "0 = 0"
            else:
                eq_text = "".join(parts) + " = 0"
            
            self.latex_display.setText(eq_text)
            self.latex_display.setStyleSheet(f"""
                QLabel {{
                    background-color: {COLORS['bg_tertiary']};
                    border-radius: 5px;
                    padding: 20px;
                    color: {COLORS['accent_primary']};
                    min-height: 60px;
                    font-size: 18px;
                    font-weight: bold;
                }}
            """)
        except Exception as e:
            # Fallback to simple display
            self.latex_display.setText(f"{a}x² + {b}x + {c} = 0")
            self.latex_display.setStyleSheet(f"""
                QLabel {{
                    background-color: {COLORS['bg_tertiary']};
                    border-radius: 5px;
                    padding: 20px;
                    color: {COLORS['accent_primary']};
                    min-height: 60px;
                    font-size: 18px;
                    font-weight: bold;
                }}
            """)
        
        # Update graph
        self.update_graph(a, b, c)
        
        # Display results
        root1, root2, disc_type = solve_quadratic(a, b, c)
        if disc_type == "Two real roots":
            results_text = f"Roots: x₁ = {root1:.4f}, x₂ = {root2:.4f}\nType: {disc_type}"
        elif disc_type == "One real root":
            results_text = f"Root: x = {root1:.4f}\nType: {disc_type}"
        else:
            results_text = f"Roots: x₁ = {root1:.4f}, x₂ = {root2:.4f}\nType: {disc_type}"
        
        self.results_label.setText(results_text)
    
    def update_graph(self, a: float, b: float, c: float):
        """Update the graph with the equation."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Generate data
        x, y = generate_equation_data(a, b, c, (-10, 10), 1000)
        
        # Plot with standardized styling
        ax.plot(x, y, color=COLORS['graph_line'], linewidth=GRAPH_LINE_WIDTH, 
               label=f'{a}x² + {b}x + {c}')
        ax.fill_between(x, y, alpha=0.3, color=COLORS['graph_fill'])
        
        # Apply standard styling
        apply_standard_graph_style(ax, title='Quadratic Equation Graph', 
                                  xlabel='x', ylabel='y')
        style_legend(ax)
        add_zero_lines(ax)
        
        self.figure.set_constrained_layout(True)
        self.canvas.draw()
    
    def get_current_coefficients(self):
        """Get the current equation coefficients."""
        return self.current_coeffs

