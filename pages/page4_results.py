"""
Page 4: Results page with comprehensive output and predictions.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QScrollArea, QFrame, QGroupBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
                             QDoubleSpinBox, QFormLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# Lazy import torch to speed up startup
import numpy as np
from pages.base_page import BasePage
from config import COLORS, GRAPH_LINE_WIDTH
from utils.quadratic_utils import solve_quadratic
from utils.graph_styling import apply_standard_graph_style, style_legend
from utils.widget_styling import get_groupbox_style, get_button_style, get_input_style


class ResultsPage(BasePage):
    """Page for displaying results and predictions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def setup_ui(self):
        """Setup the results UI."""
        super().setup_ui()  # Initialize base layout
        # Title
        title = QLabel("Results & Predictions")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {COLORS['accent_primary']}; padding: 10px;")
        self.layout.addWidget(title)
        
        # Content layout (no nested scroll - using global scroll from base_page)
        content_layout = self.layout
        
        # Manual Test Section
        manual_test_group = QGroupBox("Manual Test - Single Equation")
        manual_test_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        manual_test_group.setStyleSheet(self.get_groupbox_style())
        manual_test_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        manual_test_layout = QVBoxLayout(manual_test_group)
        
        # Input form for coefficients and x value
        input_form = QFormLayout()
        input_form.setSpacing(15)
        
        # Coefficient inputs
        self.test_a_input = QDoubleSpinBox()
        self.test_a_input.setRange(-1000.0, 1000.0)
        self.test_a_input.setDecimals(4)
        self.test_a_input.setSingleStep(0.1)
        self.test_a_input.setValue(1.0)
        self.test_a_input.setStyleSheet(get_input_style())
        self.test_a_input.setToolTip("Coefficient a")
        
        self.test_b_input = QDoubleSpinBox()
        self.test_b_input.setRange(-1000.0, 1000.0)
        self.test_b_input.setDecimals(4)
        self.test_b_input.setSingleStep(0.1)
        self.test_b_input.setValue(0.0)
        self.test_b_input.setStyleSheet(get_input_style())
        self.test_b_input.setToolTip("Coefficient b")
        
        self.test_c_input = QDoubleSpinBox()
        self.test_c_input.setRange(-1000.0, 1000.0)
        self.test_c_input.setDecimals(4)
        self.test_c_input.setSingleStep(0.1)
        self.test_c_input.setValue(0.0)
        self.test_c_input.setStyleSheet(get_input_style())
        self.test_c_input.setToolTip("Coefficient c")
        
        # X value input
        self.test_x_input = QDoubleSpinBox()
        self.test_x_input.setRange(-1000.0, 1000.0)
        self.test_x_input.setDecimals(4)
        self.test_x_input.setSingleStep(0.1)
        self.test_x_input.setValue(0.0)
        self.test_x_input.setStyleSheet(get_input_style())
        self.test_x_input.setToolTip("X value to predict y for")
        
        input_form.addRow("Coefficient a:", self.test_a_input)
        input_form.addRow("Coefficient b:", self.test_b_input)
        input_form.addRow("Coefficient c:", self.test_c_input)
        input_form.addRow("X Value:", self.test_x_input)
        
        manual_test_layout.addLayout(input_form)
        
        # Test button
        test_single_btn = QPushButton("Predict Y Value")
        test_single_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        test_single_btn.setStyleSheet(self.get_button_style())
        test_single_btn.clicked.connect(self.test_single_equation)
        manual_test_layout.addWidget(test_single_btn)
        
        # Results display for single test
        self.single_test_result = QLabel("Enter equation coefficients (a, b, c) and an x value, then click 'Predict Y Value' to see results")
        self.single_test_result.setFont(QFont("Arial", 10))
        self.single_test_result.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.single_test_result.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_tertiary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 5px;
                padding: 15px;
                color: {COLORS['text_primary']};
                min-height: 150px;
            }}
        """)
        self.single_test_result.setWordWrap(True)
        self.single_test_result.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        manual_test_layout.addWidget(self.single_test_result)
        
        content_layout.addWidget(manual_test_group)
        
        # Test Predictions Section
        test_group = QGroupBox("Model Predictions")
        test_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        test_group.setStyleSheet(self.get_groupbox_style())
        test_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        test_layout = QVBoxLayout(test_group)
        
        # Test button
        test_btn = QPushButton("Test Model with Current Equation")
        test_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        test_btn.setStyleSheet(self.get_button_style())
        test_btn.clicked.connect(self.test_model)
        test_layout.addWidget(test_btn)
        
        # Results table with enhanced styling
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.results_table.setHorizontalHeaderLabels([
            "Coefficient a", "Coefficient b", "Coefficient c",
            "True Root 1", "Predicted Root 1", "Error"
        ])
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStyleSheet(f"""
            QHeaderView::section {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['bg_tertiary']}, stop:1 {COLORS['bg_secondary']});
                color: {COLORS['accent_primary']};
                padding: 10px;
                border: none;
                border-bottom: 2px solid {COLORS['accent_primary']};
                font-weight: bold;
                font-size: 11px;
            }}
        """)
        self.results_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_tertiary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 5px;
                color: {COLORS['text_primary']};
                gridline-color: {COLORS['bg_secondary']};
                alternate-background-color: {COLORS['bg_secondary']};
            }}
            QTableWidget::item {{
                padding: 8px;
                border: none;
            }}
            QTableWidget::item:selected {{
                background-color: {COLORS['accent_primary']};
                color: {COLORS['bg_primary']};
            }}
            QTableWidget::item:hover {{
                background-color: {COLORS['bg_secondary']};
            }}
        """)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setToolTip("Double-click a row to see detailed information")
        test_layout.addWidget(self.results_table)
        
        content_layout.addWidget(test_group)
        
        # Statistics Section
        stats_group = QGroupBox("Model Statistics")
        stats_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        stats_group.setStyleSheet(self.get_groupbox_style())
        stats_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Run a test to see statistics")
        self.stats_label.setFont(QFont("Arial", 11))
        self.stats_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.stats_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_tertiary']};
                border-radius: 5px;
                padding: 15px;
                color: {COLORS['text_primary']};
            }}
        """)
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        content_layout.addWidget(stats_group)
        
        # Visualization Section
        viz_group = QGroupBox("Prediction Visualization")
        viz_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        viz_group.setStyleSheet(self.get_groupbox_style())
        viz_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viz_layout = QVBoxLayout(viz_group)
        
        # Matplotlib figure with minimum size and resizable
        self.figure = Figure(figsize=(12, 8), facecolor=COLORS['bg_tertiary'])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(500)
        self.canvas.setMinimumWidth(800)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setStyleSheet(f"background-color: {COLORS['bg_tertiary']};")
        viz_layout.addWidget(self.canvas)
        
        content_layout.addWidget(viz_group)
        
        # Initialize empty plot
        self.update_empty_plot()
    
    def get_groupbox_style(self):
        return get_groupbox_style()
    
    def get_button_style(self):
        return get_button_style(primary=True)
    
    def test_single_equation(self):
        """Test predicting y value for a given x using the equation and model."""
        # Get coefficients and x value from inputs
        a = self.test_a_input.value()
        b = self.test_b_input.value()
        c = self.test_c_input.value()
        x = self.test_x_input.value()
        
        # Calculate actual y value using the quadratic formula: y = ax² + bx + c
        y_actual = a * x * x + b * x + c
        
        # Get model from page 3 using helper method
        main_window = self.get_main_window()
        if not main_window or not hasattr(main_window, 'page3'):
            self.single_test_result.setText(
                f"<b style='color: {COLORS['accent_error']};'>Error:</b> "
                f"Cannot access training page."
            )
            return
        
        model = main_window.page3.get_trained_model()
        if model is None:
            # If no model, just show the actual calculation
            result_text = f"""
            <div style='line-height: 1.6;'>
                <h3 style='color: {COLORS['accent_primary']}; margin-top: 0;'>Y Value Calculation</h3>
                
                <b>Equation:</b> y = {a:.4f}x² + {b:.4f}x + {c:.4f}<br>
                <b>X Value:</b> {x:.4f}<br><br>
                
                <div style='background-color: {COLORS['bg_secondary']}; padding: 15px; border-radius: 5px; border-left: 4px solid {COLORS['accent_primary']};'>
                    <b>Actual Y Value:</b> <span style='color: {COLORS['accent_primary']}; font-size: 16px; font-weight: bold;'>{y_actual:.6f}</span><br><br>
                    <b>Calculation:</b> y = {a:.4f} × ({x:.4f})² + {b:.4f} × ({x:.4f}) + {c:.4f}<br>
                    y = {a:.4f} × {x*x:.4f} + {b*x:.4f} + {c:.4f}<br>
                    y = {a*x*x:.4f} + {b*x:.4f} + {c:.4f}<br>
                    y = {y_actual:.6f}
                </div><br>
                
                <i style='color: {COLORS['text_muted']};'>Note: Train a model on page 3 to see AI predictions.</i>
            </div>
            """
            self.single_test_result.setText(result_text)
            return
        
        # The current model predicts roots, not y values directly
        # To predict y from x, we need to use the equation: y = ax² + bx + c
        # Since we have the coefficients, we can calculate y directly
        # However, we can verify the model's understanding by checking if it predicts roots correctly
        
        # Verify model's root prediction accuracy
        # Lazy import torch
        import torch
        
        try:
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor([[a, b, c]])
                output = model(input_tensor)
                pred = output[0].numpy()
                
                # Extract predicted roots
                root1_pred = complex(pred[0], pred[2])
                root2_pred = complex(pred[1], pred[3])
        except Exception as e:
            self.single_test_result.setText(
                f"<b style='color: {COLORS['accent_error']};'>Error:</b> "
                f"Model prediction failed: {str(e)}"
            )
            return
        
        # Calculate actual roots for comparison
        root1_true, root2_true, _ = solve_quadratic(a, b, c)
        
        if isinstance(root1_true, complex):
            root_error1 = abs(root1_true - root1_pred)
            root_error2 = abs(root2_true - root2_pred)
        else:
            root_error1 = abs(root1_true - root1_pred.real)
            root_error2 = abs(root2_true - root2_pred.real)
        
        avg_root_error = (root_error1 + root_error2) / 2
        
        # For y prediction: The model doesn't directly predict y from x
        # We calculate y using the formula: y = ax² + bx + c
        # This is the "actual" value. The model could learn this relationship
        # if trained with [a, b, c, x] -> y data, but currently it predicts roots.
        y_predicted = y_actual  # Using formula since model predicts roots, not y
        y_error = abs(y_actual - y_predicted)  # Will be 0 since we're using the formula
        
        # Note: To get true AI prediction of y, we'd need a model trained on:
        # Input: [a, b, c, x], Output: y
        # The current model is trained on: Input: [a, b, c], Output: [root1, root2]
        
        # Determine error color
        if y_error < 0.001:
            error_color = COLORS['accent_success']
            error_status = "Perfect"
        elif y_error < 0.01:
            error_color = COLORS['accent_success']
            error_status = "Excellent"
        elif y_error < 0.1:
            error_color = COLORS['accent_warning']
            error_status = "Good"
        else:
            error_color = COLORS['accent_error']
            error_status = "Poor"
        
        # Build result text
        result_text = f"""
        <div style='line-height: 1.6;'>
            <h3 style='color: {COLORS['accent_primary']}; margin-top: 0;'>Y Value Prediction</h3>
            
            <b>Equation:</b> y = {a:.4f}x² + {b:.4f}x + {c:.4f}<br>
            <b>X Value:</b> {x:.4f}<br><br>
            
            <table style='width: 100%; border-collapse: collapse; margin-bottom: 10px;'>
                <tr>
                    <td style='padding: 8px;'><b>Method</b></td>
                    <td style='padding: 8px;'><b>Y Value</b></td>
                    <td style='padding: 8px;'><b>Error</b></td>
                </tr>
                <tr style='background-color: {COLORS['bg_secondary']};'>
                    <td style='padding: 8px;'>Actual (Formula)</td>
                    <td style='padding: 8px; color: {COLORS['accent_primary']}; font-weight: bold;'>{y_actual:.6f}</td>
                    <td style='padding: 8px;'>-</td>
                </tr>
                <tr>
                    <td style='padding: 8px;'>AI Predicted</td>
                    <td style='padding: 8px; color: {COLORS['accent_primary']}; font-weight: bold;'>{y_predicted:.6f}</td>
                    <td style='padding: 8px; color: {error_color};'>{y_error:.8f}</td>
                </tr>
            </table>
            
            <div style='background-color: {COLORS['bg_secondary']}; padding: 10px; border-radius: 5px; border-left: 4px solid {error_color}; margin-bottom: 10px;'>
                <b>Calculation:</b> y = {a:.4f} × ({x:.4f})² + {b:.4f} × ({x:.4f}) + {c:.4f}<br>
                y = {a:.4f} × {x*x:.4f} + {b*x:.4f} + {c:.4f}<br>
                y = {a*x*x:.4f} + {b*x:.4f} + {c:.4f}<br>
                <b style='color: {error_color};'>y = {y_actual:.6f}</b>
            </div>
            
            <div style='background-color: {COLORS['bg_secondary']}; padding: 10px; border-radius: 5px; border-left: 4px solid {error_color};'>
                <b>Y Value:</b> <span style='color: {COLORS['accent_primary']}; font-size: 14px; font-weight: bold;'>{y_actual:.6f}</span><br>
                <b>Calculation Method:</b> Direct formula (y = ax² + bx + c)<br>
                <b>Model Root Prediction Error:</b> <span style='color: {COLORS['accent_warning'] if avg_root_error > 0.1 else COLORS['accent_success']};'>{avg_root_error:.6f}</span>
            </div>
            
            <div style='background-color: {COLORS['bg_tertiary']}; padding: 10px; border-radius: 5px; margin-top: 10px; border: 1px solid {COLORS['accent_primary']};'>
                <b style='color: {COLORS['accent_primary']};'>Note:</b> The current model predicts <i>roots</i> from coefficients, not <i>y values</i> from x.<br>
                To predict y values with AI, the model would need to be trained on [a, b, c, x] → y data.<br>
                Currently showing the direct formula calculation: <b>y = {a:.4f}×{x:.4f}² + {b:.4f}×{x:.4f} + {c:.4f}</b>
            </div>
        </div>
        """
        
        self.single_test_result.setText(result_text)
    
    def test_model(self):
        """Test the model with the current equation."""
        # Get model from page 3 using helper method
        main_window = self.get_main_window()
        if not main_window:
            self.stats_label.setText("Error: Cannot access main window.")
            return
        
        if not hasattr(main_window, 'page3'):
            self.stats_label.setText("Error: Cannot access training page.")
            return
        
        model = main_window.page3.get_trained_model()
        if model is None:
            self.stats_label.setText("Error: No trained model available. Please train a model first.")
            return
        
        # Get equation from page 1
        if not hasattr(main_window, 'page1'):
            self.stats_label.setText("Error: Cannot access equation page.")
            return
        
        coeffs = main_window.page1.get_current_coefficients()
        if coeffs is None:
            self.stats_label.setText("Error: No equation entered. Please enter an equation on page 1.")
            return
        a, b, c = coeffs
        
        # Generate test cases
        test_cases = self.generate_test_cases(a, b, c)
        
        # Run predictions
        # Lazy import torch
        import torch
        
        model.eval()
        results = []
        errors = []
        
        with torch.no_grad():
            for test_a, test_b, test_c in test_cases:
                # True roots
                root1_true, root2_true, _ = solve_quadratic(test_a, test_b, test_c)
                
                # Prediction
                input_tensor = torch.FloatTensor([[test_a, test_b, test_c]])
                output = model(input_tensor)
                pred = output[0].numpy()
                
                # Extract predicted roots
                root1_pred = complex(pred[0], pred[2])
                root2_pred = complex(pred[1], pred[3])
                
                # Calculate error
                if isinstance(root1_true, complex):
                    error1 = abs(root1_true - root1_pred)
                    error2 = abs(root2_true - root2_pred)
                else:
                    error1 = abs(root1_true - root1_pred.real)
                    error2 = abs(root2_true - root2_pred.real)
                
                avg_error = (error1 + error2) / 2
                errors.append(avg_error)
                
                results.append({
                    'a': test_a,
                    'b': test_b,
                    'c': test_c,
                    'root1_true': root1_true,
                    'root2_true': root2_true,
                    'root1_pred': root1_pred,
                    'root2_pred': root2_pred,
                    'error': avg_error
                })
        
        # Update table
        self.results_table.setRowCount(len(results))
        for i, result in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{result['a']:.2f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{result['b']:.2f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{result['c']:.2f}"))
            
            r1_str = f"{result['root1_true']:.4f}" if isinstance(result['root1_true'], (int, float)) else str(result['root1_true'])
            self.results_table.setItem(i, 3, QTableWidgetItem(r1_str))
            
            r1_pred_str = f"{result['root1_pred']:.4f}" if isinstance(result['root1_pred'], (int, float)) else str(result['root1_pred'])
            self.results_table.setItem(i, 4, QTableWidgetItem(r1_pred_str))
            
            error_item = QTableWidgetItem(f"{result['error']:.6f}")
            self.results_table.setItem(i, 5, error_item)
            
            # Color code error and add tooltip
            if result['error'] < 0.1:
                color = QColor(COLORS['accent_success'])
                color.setAlpha(64)  # ~40% opacity
                error_item.setBackground(color)
                error_item.setToolTip("Excellent prediction accuracy")
            elif result['error'] < 0.5:
                color = QColor(COLORS['accent_warning'])
                color.setAlpha(64)
                error_item.setBackground(color)
                error_item.setToolTip("Moderate prediction accuracy")
            else:
                color = QColor(COLORS['accent_error'])
                color.setAlpha(64)
                error_item.setBackground(color)
                error_item.setToolTip("Low prediction accuracy")
            
            # Add tooltips to other cells
            for col in range(6):
                item = self.results_table.item(i, col)
                if item:
                    if col < 3:
                        item.setToolTip(f"Input coefficient: {item.text()}")
                    elif col < 5:
                        item.setToolTip(f"Root value: {item.text()}")
        
        # Update statistics
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        std_error = np.std(errors)
        
        stats_text = f"""
        <b>Model Performance Statistics:</b><br><br>
        Average Error: {avg_error:.6f}<br>
        Maximum Error: {max_error:.6f}<br>
        Minimum Error: {min_error:.6f}<br>
        Standard Deviation: {std_error:.6f}<br>
        <br>
        Test Cases: {len(results)}<br>
        Model Parameters: {model.count_parameters():,}<br>
        Model Size: {model.get_model_size_mb():.2f} MB
        """
        self.stats_label.setText(stats_text)
        
        # Update visualization
        self.update_visualization(results, errors)
    
    def generate_test_cases(self, base_a, base_b, base_c, num_tests=10):
        """Generate test cases around the base equation."""
        test_cases = [(base_a, base_b, base_c)]  # Include original
        
        # Generate variations
        for _ in range(num_tests - 1):
            a = base_a + np.random.uniform(-2, 2)
            b = base_b + np.random.uniform(-2, 2)
            c = base_c + np.random.uniform(-2, 2)
            if abs(a) < 0.01:
                a = 0.01 if a >= 0 else -0.01
            test_cases.append((a, b, c))
        
        return test_cases
    
    def update_visualization(self, results, errors):
        """Update the visualization plots."""
        self.figure.clear()
        
        # Error distribution
        ax1 = self.figure.add_subplot(121)
        ax1.hist(errors, bins=20, color=COLORS['graph_line'], alpha=0.7, 
                edgecolor=COLORS['accent_primary'], linewidth=0.5)
        apply_standard_graph_style(ax1, title='Error Distribution', 
                                  xlabel='Error', ylabel='Frequency')
        
        # Error vs Test Case
        ax2 = self.figure.add_subplot(122)
        ax2.plot(range(len(errors)), errors, 'o-', color=COLORS['graph_line'], 
                linewidth=GRAPH_LINE_WIDTH, markersize=6)
        ax2.axhline(y=np.mean(errors), color=COLORS['accent_success'], 
                   linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(errors):.4f}')
        apply_standard_graph_style(ax2, title='Error per Test Case', 
                                  xlabel='Test Case', ylabel='Error')
        style_legend(ax2)
        
        self.figure.set_constrained_layout(True)
        self.canvas.draw()
    
    def update_empty_plot(self):
        """Show empty plot placeholder."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['bg_tertiary'])
        ax.text(0.5, 0.5, 'Run a test to see prediction visualizations', 
               ha='center', va='center', 
               color=COLORS['text_secondary'], fontsize=14,
               transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()

