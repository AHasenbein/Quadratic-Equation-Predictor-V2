"""
Page 3: Results & Analysis (Comprehensive)
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, 
                             QTableWidget, QTableWidgetItem, QTextEdit, QTabWidget, QHeaderView,
                             QPushButton, QComboBox, QScrollArea, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from config import COLORS
from utils.visualization import (plot_optimization_convergence, plot_hyperparameter_importance,
                                 plot_hyperparameter_heatmap, plot_pareto_frontier,
                                 plot_error_analysis, plot_quadratic_with_roots)
import optuna


class ResultsPage(QWidget):
    """Comprehensive results and analysis page."""
    
    def __init__(self):
        super().__init__()
        self.results = None
        self.trial_history = []
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Results & Analysis")
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
        
        # Tab 1: Summary & Metrics
        summary_tab = self._create_summary_tab()
        tabs.addTab(summary_tab, "Summary")
        
        # Tab 2: Convergence Analysis
        convergence_tab = self._create_convergence_tab()
        tabs.addTab(convergence_tab, "Convergence")
        
        # Tab 3: Hyperparameter Importance
        importance_tab = self._create_importance_tab()
        tabs.addTab(importance_tab, "Importance")
        
        # Tab 4: Hyperparameter Space
        space_tab = self._create_space_tab()
        tabs.addTab(space_tab, "Parameter Space")
        
        # Tab 5: Error Analysis
        error_tab = self._create_error_tab()
        tabs.addTab(error_tab, "Error Analysis")
        
        # Tab 6: Trial History
        history_tab = self._create_history_tab()
        tabs.addTab(history_tab, "Trial History")
        
        # Tab 7: Insights
        insights_tab = self._create_insights_tab()
        tabs.addTab(insights_tab, "Research Insights")
        
        layout.addWidget(tabs)
    
    def _create_summary_tab(self):
        """Create summary tab with metric cards."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Metrics cards
        cards_layout = QHBoxLayout()
        
        self.metric_cards = {}
        metrics = ['loss', 'mae', 'rmse', 'r2', 'accuracy']
        metric_labels = ['Loss', 'MAE', 'RMSE', 'R²', 'Accuracy %']
        
        for metric, label in zip(metrics, metric_labels):
            card = QGroupBox(label)
            card.setStyleSheet(f"""
                QGroupBox {{
                    background-color: {COLORS['bg_secondary']};
                    border: 2px solid {COLORS['accent_primary']};
                    border-radius: 10px;
                    padding: 15px;
                    color: {COLORS['accent_primary']};
                    font-weight: bold;
                }}
            """)
            card_layout = QVBoxLayout(card)
            
            value_label = QLabel("--")
            value_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
            value_label.setStyleSheet(f"color: {COLORS['accent_primary']};")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(value_label)
            
            self.metric_cards[metric] = value_label
            cards_layout.addWidget(card)
        
        layout.addLayout(cards_layout)
        
        # Summary text
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        self.summary_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['accent_primary']};
                border-radius: 5px;
                font-size: 12px;
            }}
        """)
        layout.addWidget(self.summary_text)
        
        # Export buttons
        export_layout = QHBoxLayout()
        export_pdf_btn = QPushButton("Export PDF Report")
        export_pdf_btn.clicked.connect(self._export_pdf)
        export_csv_btn = QPushButton("Export CSV Data")
        export_csv_btn.clicked.connect(self._export_csv)
        export_layout.addWidget(export_pdf_btn)
        export_layout.addWidget(export_csv_btn)
        layout.addLayout(export_layout)
        
        return widget
    
    def _create_convergence_tab(self):
        """Create convergence analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.conv_figure = Figure(figsize=(12, 6), facecolor=COLORS['bg_tertiary'])
        self.conv_canvas = FigureCanvas(self.conv_figure)
        self.conv_ax = self.conv_figure.add_subplot(111)
        
        layout.addWidget(self.conv_canvas)
        
        return widget
    
    def _create_importance_tab(self):
        """Create hyperparameter importance tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.importance_figure = Figure(figsize=(10, 6), facecolor=COLORS['bg_tertiary'])
        self.importance_canvas = FigureCanvas(self.importance_figure)
        self.importance_ax = self.importance_figure.add_subplot(111)
        
        layout.addWidget(self.importance_canvas)
        
        return widget
    
    def _create_space_tab(self):
        """Create hyperparameter space exploration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Parameter selection
        param_group = QGroupBox("Select Parameters for Heatmap")
        param_group.setStyleSheet(f"""
            QGroupBox {{
                background-color: {COLORS['bg_secondary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 10px;
                padding: 15px;
                color: {COLORS['accent_primary']};
                font-weight: bold;
            }}
        """)
        param_layout = QHBoxLayout(param_group)
        
        param_layout.addWidget(QLabel("Parameter 1:"))
        self.param1_combo = QComboBox()
        param_layout.addWidget(self.param1_combo)
        
        param_layout.addWidget(QLabel("Parameter 2:"))
        self.param2_combo = QComboBox()
        param_layout.addWidget(self.param2_combo)
        
        update_btn = QPushButton("Update Heatmap")
        update_btn.clicked.connect(self._update_heatmap)
        param_layout.addWidget(update_btn)
        
        layout.addWidget(param_group)
        
        # Heatmap
        self.heatmap_figure = Figure(figsize=(10, 8), facecolor=COLORS['bg_tertiary'])
        self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
        self.heatmap_ax = self.heatmap_figure.add_subplot(111)
        self.heatmap_cbar = None  # Store colorbar reference
        # Set initial subplot parameters for resizing
        self.heatmap_figure.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95)
        self.heatmap_cbar = None  # Store colorbar reference
        
        layout.addWidget(self.heatmap_canvas)
        
        return widget
    
    def _create_error_tab(self):
        """Create error analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.error_figure = Figure(figsize=(10, 6), facecolor=COLORS['bg_tertiary'])
        self.error_canvas = FigureCanvas(self.error_figure)
        self.error_ax = self.error_figure.add_subplot(111)
        
        layout.addWidget(self.error_canvas)
        
        return widget
    
    def _create_history_tab(self):
        """Create trial history explorer tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["Trial", "Loss", "Layers", "Size", "Activation", "LR"])
        self.history_table.horizontalHeader().setStyleSheet(f"color: {COLORS['accent_primary']}; font-weight: bold;")
        self.history_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['accent_primary']};
                border-radius: 5px;
            }}
            QTableWidget::item {{
                padding: 5px;
            }}
        """)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setSortingEnabled(True)
        
        layout.addWidget(QLabel("All Trials (click headers to sort):"))
        layout.addWidget(self.history_table)
        
        # Export button
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(self._export_trials)
        layout.addWidget(export_btn)
        
        return widget
    
    def _create_insights_tab(self):
        """Create research insights tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['accent_primary']};
                border-radius: 5px;
                font-size: 12px;
            }}
        """)
        layout.addWidget(self.insights_text)
        
        return widget
    
    def _update_heatmap(self):
        """Update heatmap with selected parameters."""
        if not self.results or 'study' not in self.results:
            return
        
        param1 = self.param1_combo.currentText()
        param2 = self.param2_combo.currentText()
        
        if param1 and param2:
            # Store positions before clearing
            original_ax_pos = self.heatmap_ax.get_position()
            # Store colorbar position if it exists (for first update, use default)
            if hasattr(self, 'heatmap_cbar') and self.heatmap_cbar:
                try:
                    stored_cbar_pos = self.heatmap_cbar.ax.get_position()
                except:
                    stored_cbar_pos = None
            else:
                # First time - calculate default position
                stored_cbar_pos = None
            
            self.heatmap_ax.clear()
            # Plot and capture colorbar
            _, cbar = plot_hyperparameter_heatmap(self.results['study'], param1, param2, self.heatmap_ax)
            if cbar:
                self.heatmap_cbar = cbar
                # Restore axes position first
                self.heatmap_ax.set_position(original_ax_pos)
                # Then set colorbar position relative to axes (fixed position)
                from matplotlib.transforms import Bbox
                pos = self.heatmap_ax.get_position()
                cbar_left = pos.x1 + 0.02
                cbar_bottom = pos.y0
                cbar_width = 0.02
                cbar_height = pos.height
                cbar_bbox = Bbox.from_bounds(cbar_left, cbar_bottom, cbar_width, cbar_height)
                self.heatmap_cbar.ax.set_position(cbar_bbox)
            else:
                # Restore axes position even if no colorbar
                self.heatmap_ax.set_position(original_ax_pos)
            
            # Use subplots_adjust to maintain layout and allow resizing
            self.heatmap_figure.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95)
            
            self.heatmap_canvas.draw()
    
    def _export_trials(self):
        """Export trial history to CSV."""
        import csv
        from PyQt6.QtWidgets import QFileDialog
        
        if not self.trial_history:
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save Trials", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Trial', 'Loss'] + list(self.trial_history[0]['params'].keys()))
                for trial in self.trial_history:
                    row = [trial.get('trial', 0), trial.get('loss', 0)]
                    row.extend([trial['params'].get(k, '') for k in self.trial_history[0]['params'].keys()])
                    writer.writerow(row)
    

    def _export_pdf(self):
        """Export results to PDF."""
        from PyQt6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "", "PDF Files (*.pdf)")
        if filename:
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(filename) as pdf:
                    # Save convergence plot
                    if self.trial_history:
                        self.conv_figure.savefig(pdf, format='pdf', bbox_inches='tight')
                    
                    # Save importance plot
                    if 'study' in self.results:
                        self.importance_figure.savefig(pdf, format='pdf', bbox_inches='tight')
                self.summary_text.append(f"\n✓ PDF exported to {filename}")
            except Exception as e:
                self.summary_text.append(f"\n✗ PDF export failed: {str(e)}")
    
    def _export_csv(self):
        """Export results to CSV."""
        from PyQt6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Value'])
                    if 'best_value' in self.results:
                        writer.writerow(['Best Loss', self.results['best_value']])
                    if 'best_params' in self.results:
                        for key, value in self.results['best_params'].items():
                            writer.writerow([f'Param: {key}', value])
                self.summary_text.append(f"\n✓ CSV exported to {filename}")
            except Exception as e:
                self.summary_text.append(f"\n✗ CSV export failed: {str(e)}")
    
    def set_results(self, results):
        """Set and display results."""
        self.results = results
        
        if not results:
            return
        
        # Extract trial history
        if 'trial_history' in results:
            # Direct trial history from training page
            self.trial_history = results['trial_history']
        elif 'study' in results:
            study = results['study']
            self.trial_history = []
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    self.trial_history.append({
                        'trial': trial.number,
                        'loss': trial.value,
                        'params': trial.params
                    })
        elif 'all_trials' in results:
            self.trial_history = results['all_trials']
        else:
            self.trial_history = []
        
        # Update metrics cards
        if 'best_value' in results:
            self.metric_cards['loss'].setText(f"{results['best_value']:.6f}")
            # Color code based on performance
            if results['best_value'] < 0.01:
                self.metric_cards['loss'].setStyleSheet(f"color: {COLORS['accent_success']};")
            elif results['best_value'] < 0.1:
                self.metric_cards['loss'].setStyleSheet(f"color: {COLORS['accent_warning']};")
            else:
                self.metric_cards['loss'].setStyleSheet(f"color: {COLORS['accent_error']};")
        
        # Update other metrics from best_metrics
        if 'best_metrics' in results and results['best_metrics']:
            metrics = results['best_metrics']
            if 'mae' in metrics:
                self.metric_cards['mae'].setText(f"{metrics['mae']:.6f}")
            if 'rmse' in metrics:
                self.metric_cards['rmse'].setText(f"{metrics['rmse']:.6f}")
            if 'r2' in metrics:
                self.metric_cards['r2'].setText(f"{metrics['r2']:.4f}")
            if 'accuracy' in metrics:
                self.metric_cards['accuracy'].setText(f"{metrics['accuracy']:.2f}%")
        
        # Update summary
        summary = []
        summary.append("OPTIMIZATION RESULTS SUMMARY")
        summary.append("=" * 60)
        summary.append("")
        
        if 'best_value' in results:
            summary.append(f"Best Validation Loss: {results['best_value']:.6f}")
            summary.append("")
            summary.append("Performance Interpretation:")
            if results['best_value'] < 0.01:
                summary.append("  ✓ EXCELLENT - Model predicts roots very accurately")
            elif results['best_value'] < 0.1:
                summary.append("  ✓ GOOD - Model makes reasonably accurate predictions")
            elif results['best_value'] < 1.0:
                summary.append("  ⚠ MODERATE - Consider more training or better architecture")
            else:
                summary.append("  ✗ POOR - Adjust hyperparameters or train longer")
            summary.append("")
        
        if 'n_trials' in results:
            summary.append(f"Total Trials: {results['n_trials']}")
        if 'optimization_method' in results:
            summary.append(f"Method: {results['optimization_method'].upper()}")
        if 'epochs_per_trial' in results:
            summary.append(f"Epochs per Trial: {results['epochs_per_trial']}")
        
        self.summary_text.setText("\n".join(summary))
        
        # Update convergence plot
        if self.trial_history:
            self.conv_ax.clear()
            plot_optimization_convergence(self.trial_history, self.conv_ax)
            self.conv_canvas.draw()
        
        # Update importance plot
        if 'study' in results:
            self.importance_ax.clear()
            plot_hyperparameter_importance(results['study'], self.importance_ax)
            self.importance_canvas.draw()
            
            # Update parameter combos - only show numerical parameters
            if results['study'].trials:
                # Get all parameters
                all_params = list(results['study'].trials[0].params.keys())
                # Filter to only numerical parameters
                numerical_params = []
                for param_name in all_params:
                    # Check if parameter is numerical by looking at trial values
                    sample_val = results['study'].trials[0].params.get(param_name)
                    if isinstance(sample_val, (int, float)):
                        numerical_params.append(param_name)
                
                self.param1_combo.clear()
                self.param2_combo.clear()
                if numerical_params:
                    self.param1_combo.addItems(numerical_params)
                    self.param2_combo.addItems(numerical_params)
                    if len(numerical_params) >= 2:
                        self.param1_combo.setCurrentIndex(0)
                        self.param2_combo.setCurrentIndex(1)
                        self._update_heatmap()
                    elif len(numerical_params) == 1:
                        # Only one numerical param, show message
                        self.heatmap_ax.clear()
                        self.heatmap_ax.text(0.5, 0.5, 'Need at least 2 numerical parameters\nfor heatmap visualization',
                                           ha='center', va='center', transform=self.heatmap_ax.transAxes,
                                           color=COLORS['text_secondary'], fontsize=12)
                        self.heatmap_canvas.draw()
                else:
                    # No numerical parameters
                    self.heatmap_ax.clear()
                    self.heatmap_ax.text(0.5, 0.5, 'No numerical parameters available\nfor heatmap visualization',
                                       ha='center', va='center', transform=self.heatmap_ax.transAxes,
                                       color=COLORS['text_secondary'], fontsize=12)
                    self.heatmap_canvas.draw()
        
        # Update error analysis
        if 'best_metrics' in results and results['best_metrics']:
            error_analysis = results['best_metrics'].get('error_analysis', {})
            if error_analysis:
                self.error_ax.clear()
                plot_error_analysis(error_analysis, self.error_ax)
                self.error_canvas.draw()
        
        # Update trial history table
        if self.trial_history:
            self.history_table.setRowCount(len(self.trial_history))
            for i, trial in enumerate(self.trial_history):
                params = trial.get('params', {})
                self.history_table.setItem(i, 0, QTableWidgetItem(str(trial.get('trial', i))))
                self.history_table.setItem(i, 1, QTableWidgetItem(f"{trial.get('loss', 0):.6f}"))
                self.history_table.setItem(i, 2, QTableWidgetItem(str(params.get('hidden_layers_count', ''))))
                self.history_table.setItem(i, 3, QTableWidgetItem(str(params.get('hidden_layer_size', ''))))
                self.history_table.setItem(i, 4, QTableWidgetItem(str(params.get('activation', ''))))
                self.history_table.setItem(i, 5, QTableWidgetItem(f"{params.get('learning_rate', 0):.6f}"))
        
        # Generate insights
        self._generate_insights()
    
    def _generate_insights(self):
        """Generate automated research insights."""
        insights = []
        insights.append("RESEARCH INSIGHTS & RECOMMENDATIONS")
        insights.append("=" * 60)
        insights.append("")
        
        if not self.results:
            self.insights_text.setText("No results available.")
            return
        
        # Analyze best parameters
        if 'best_params' in self.results:
            params = self.results['best_params']
            insights.append("OPTIMAL HYPERPARAMETERS:")
            insights.append("")
            
            if 'activation' in params:
                insights.append(f"• Activation Function: {params['activation'].upper()}")
                insights.append(f"  → This activation performed best for this problem")
            
            if 'hidden_layers_count' in params and 'hidden_layer_size' in params:
                layers = params['hidden_layers_count']
                size = params['hidden_layer_size']
                total_params = layers * size * 3 + size * 4  # Approximate
                insights.append(f"• Architecture: {layers} layers × {size} units")
                insights.append(f"  → Total parameters: ~{total_params}")
            
            if 'learning_rate' in params:
                lr = params['learning_rate']
                insights.append(f"• Learning Rate: {lr:.6f}")
                if lr < 1e-4:
                    insights.append("  → Very small LR suggests stable but slow convergence")
                elif lr > 1e-2:
                    insights.append("  → Larger LR suggests faster learning was beneficial")
            
            if 'dropout' in params:
                dropout = params['dropout']
                if dropout > 0.2:
                    insights.append(f"• Dropout: {dropout:.2f} (high regularization needed)")
                elif dropout > 0:
                    insights.append(f"• Dropout: {dropout:.2f} (moderate regularization)")
                else:
                    insights.append("• Dropout: 0.0 (no overfitting issues)")
        
        insights.append("")
        insights.append("PRACTICAL RECOMMENDATIONS:")
        insights.append("")
        
        if 'best_value' in self.results:
            loss = self.results['best_value']
            if loss < 0.01:
                insights.append("✓ Model is highly accurate - ready for deployment")
                insights.append("✓ Can potentially reduce model size while maintaining accuracy")
            elif loss < 0.1:
                insights.append("✓ Good performance - suitable for most applications")
                insights.append("→ Consider ensemble methods for even better accuracy")
            else:
                insights.append("→ Increase training epochs or model capacity")
                insights.append("→ Try different optimization methods (Optuna vs Random)")
                insights.append("→ Consider data augmentation or more training samples")
        
        if self.trial_history and len(self.trial_history) > 10:
            losses = [t['loss'] for t in self.trial_history]
            improvement = (max(losses) - min(losses)) / max(losses) * 100
            insights.append("")
            insights.append(f"OPTIMIZATION EFFECTIVENESS:")
            insights.append(f"• Loss improved by {improvement:.1f}% over {len(self.trial_history)} trials")
            if improvement > 50:
                insights.append("  → Optimization was highly effective")
            elif improvement > 20:
                insights.append("  → Moderate improvement achieved")
            else:
                insights.append("  → Limited improvement - may need larger search space")
        
        self.insights_text.setText("\n".join(insights))
