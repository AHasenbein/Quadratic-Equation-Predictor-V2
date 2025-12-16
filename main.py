"""
Main GUI application for hyperparameter optimization research.
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from config import COLORS, WINDOW_WIDTH, WINDOW_HEIGHT
from pages.page1_data_generation import DataGenerationPage
from pages.page2_training import TrainingPage
from pages.page3_results import ResultsPage


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperparameter Optimization Research")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['bg_primary']};
            }}
        """)
        
        # Central widget with stacked pages
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Navigation bar
        nav_bar = self._create_nav_bar()
        layout.addWidget(nav_bar)
        
        # Stacked widget for pages
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        
        # Create pages
        self.page1 = DataGenerationPage()
        self.page2 = TrainingPage()
        self.page3 = ResultsPage()
        
        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)
        
        # Connect pages
        self.page1.data_generated.connect(self._on_data_generated)
        self.page2.optimization_complete.connect(self._on_optimization_complete)
        
        # Show first page
        self.stacked_widget.setCurrentIndex(0)
    
    def _create_nav_bar(self):
        """Create navigation bar."""
        nav_widget = QWidget()
        nav_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_secondary']};
                border-bottom: 2px solid {COLORS['accent_primary']};
            }}
        """)
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(20, 10, 20, 10)
        nav_layout.setSpacing(10)
        
        btn1 = QPushButton("1. Data Generation")
        btn2 = QPushButton("2. Training & Optimization")
        btn3 = QPushButton("3. Results & Analysis")
        
        for btn in [btn1, btn2, btn3]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['bg_tertiary']};
                    color: {COLORS['text_primary']};
                    border: 2px solid {COLORS['accent_primary']};
                    border-radius: 5px;
                    padding: 10px 20px;
                    font-size: 12px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['accent_primary']};
                    color: {COLORS['bg_primary']};
                }}
            """)
        
        btn1.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        btn2.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        btn3.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        
        nav_layout.addWidget(btn1)
        nav_layout.addWidget(btn2)
        nav_layout.addWidget(btn3)
        nav_layout.addStretch()
        
        return nav_widget
    
    def _on_data_generated(self, data):
        """Handle data generation completion."""
        self.page2.set_training_data(data)
        self.stacked_widget.setCurrentIndex(1)
    
    def _on_optimization_complete(self, results):
        """Handle optimization completion."""
        self.page3.set_results(results)
        self.stacked_widget.setCurrentIndex(2)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
