"""
Main application file for Quadratic Equation Predictor.
Modern GUI with page navigation system.
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QStackedWidget, QLabel)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSettings
from PyQt6.QtGui import QFont, QIcon
from config import COLORS, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT
from pages.page1_equation import EquationPage
from pages.page2_data_generation import DataGenerationPage
from pages.page3_training import TrainingPage
from pages.page4_results import ResultsPage


class MainWindow(QMainWindow):
    """Main application window with page navigation."""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings('QEP', 'QuadraticEquationPredictor')
        self.setup_window()
        self.setup_ui()
        self.apply_styles()
    
    def setup_window(self):
        """Setup window properties with saved size."""
        self.setWindowTitle("Quadratic Equation Predictor - AI Hyper-Optimization")
        
        # Load saved window geometry or use defaults
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Load saved window state
        state = self.settings.value('windowState')
        if state:
            self.restoreState(state)
        
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
    
    def closeEvent(self, event):
        """Save window size and position when closing."""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
        super().closeEvent(event)
    
    def setup_ui(self):
        """Setup the main UI."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Navigation bar
        nav_bar = self.create_nav_bar()
        main_layout.addWidget(nav_bar)
        
        # Page stack with expanding size policy
        self.page_stack = QStackedWidget()
        self.page_stack.setStyleSheet(f"background-color: {COLORS['bg_primary']};")
        from PyQt6.QtWidgets import QSizePolicy
        self.page_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.page_stack)
        
        # Create pages
        self.page1 = EquationPage(self)
        self.page2 = DataGenerationPage(self)
        self.page3 = TrainingPage(self)
        self.page4 = ResultsPage(self)
        
        # Add pages to stack
        self.page_stack.addWidget(self.page1)
        self.page_stack.addWidget(self.page2)
        self.page_stack.addWidget(self.page3)
        self.page_stack.addWidget(self.page4)
        
        # Set initial page
        self.page_stack.setCurrentIndex(0)
        self.current_page_index = 0
    
    def create_nav_bar(self):
        """Create navigation bar with page buttons."""
        nav_frame = QWidget()
        nav_frame.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_secondary']};
                border-bottom: 2px solid {COLORS['accent_primary']};
            }}
        """)
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(10, 10, 10, 10)
        nav_layout.setSpacing(10)
        
        # Page buttons
        self.nav_buttons = []
        page_names = ["Equation", "Data & Presets", "Training", "Results"]
        
        for i, name in enumerate(page_names):
            btn = QPushButton(name)
            btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.clicked.connect(lambda checked, idx=i: self.navigate_to_page(idx))
            btn.setStyleSheet(self.get_nav_button_style(i == 0))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)
        
        nav_layout.addStretch()
        
        # Title label
        title_label = QLabel("QEP AI")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COLORS['accent_primary']}; padding: 5px;")
        nav_layout.addWidget(title_label)
        
        return nav_frame
    
    def get_nav_button_style(self, is_active=False):
        """Get style for navigation button."""
        if is_active:
            return f"""
                QPushButton {{
                    background-color: {COLORS['accent_primary']};
                    color: {COLORS['bg_primary']};
                    border: none;
                    border-radius: 5px;
                    padding: 10px 20px;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['accent_secondary']};
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: {COLORS['bg_tertiary']};
                    color: {COLORS['text_primary']};
                    border: 2px solid {COLORS['accent_primary']};
                    border-radius: 5px;
                    padding: 10px 20px;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['bg_secondary']};
                    border: 2px solid {COLORS['accent_secondary']};
                }}
            """
    
    def navigate_to_page(self, index):
        """Navigate to a specific page with animation."""
        if index == self.current_page_index:
            return
        
        # Update button states
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
            btn.setStyleSheet(self.get_nav_button_style(i == index))
        
        # Animate page transition
        self.animate_page_transition(index)
        self.current_page_index = index
    
    def animate_page_transition(self, new_index):
        """Animate transition between pages."""
        # Simple fade transition
        self.page_stack.setCurrentIndex(new_index)
    
    def apply_styles(self):
        """Apply global styles to the window."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['bg_primary']};
            }}
            QScrollBar:vertical {{
                background-color: {COLORS['bg_secondary']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {COLORS['accent_primary']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {COLORS['accent_secondary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {COLORS['bg_secondary']};
                height: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {COLORS['accent_primary']};
                border-radius: 6px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {COLORS['accent_secondary']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better appearance
    
    # Set application properties
    app.setApplicationName("Quadratic Equation Predictor")
    app.setOrganizationName("QEP AI")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
