"""
Graph Manager utility for dynamic graph sizing and viewport management.
Ensures all graphs fit properly within their containers.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QSize, pyqtSignal
from matplotlib.figure import Figure
from config import GRAPH_MIN_WIDTH, GRAPH_MIN_HEIGHT, GRAPH_ASPECT_RATIO, SCROLL_AREA_PADDING


class GraphManager:
    """
    Manages graph sizing and ensures proper display within viewports.
    """
    
    def __init__(self, dpi=100):
        self.dpi = dpi
        self.graph_configs = {}
    
    def calculate_optimal_size(self, available_width: int, available_height: int, 
                             min_width: int = None, min_height: int = None,
                             aspect_ratio: float = None) -> tuple:
        """
        Calculate optimal figure size based on available space.
        
        Args:
            available_width: Available width in pixels
            available_height: Available height in pixels
            min_width: Minimum width (defaults to GRAPH_MIN_WIDTH)
            min_height: Minimum height (defaults to GRAPH_MIN_HEIGHT)
            aspect_ratio: Desired aspect ratio (defaults to GRAPH_ASPECT_RATIO)
        
        Returns:
            Tuple of (width, height) in inches for matplotlib figsize
        """
        min_w = min_width or GRAPH_MIN_WIDTH
        min_h = min_height or GRAPH_MIN_HEIGHT
        aspect = aspect_ratio or GRAPH_ASPECT_RATIO
        
        # Account for padding
        padding = SCROLL_AREA_PADDING * 2
        available_width = max(available_width - padding, min_w)
        available_height = max(available_height - padding, min_h)
        
        # Calculate size maintaining aspect ratio
        width_inches = available_width / self.dpi
        height_inches = available_height / self.dpi
        
        # Adjust to maintain aspect ratio
        if width_inches / height_inches > aspect:
            width_inches = height_inches * aspect
        else:
            height_inches = width_inches / aspect
        
        # Ensure minimum sizes
        width_inches = max(width_inches, min_w / self.dpi)
        height_inches = max(height_inches, min_h / self.dpi)
        
        return (width_inches, height_inches)
    
    def get_canvas_minimum_size(self, figsize: tuple) -> QSize:
        """
        Get minimum canvas size from figure size.
        
        Args:
            figsize: Tuple of (width, height) in inches
        
        Returns:
            QSize for canvas minimum size
        """
        width = int(figsize[0] * self.dpi)
        height = int(figsize[1] * self.dpi)
        return QSize(width, height)
    
    def register_graph_config(self, page_name: str, config: dict):
        """
        Register graph configuration for a specific page.
        
        Args:
            page_name: Name of the page
            config: Dictionary with graph configuration
        """
        self.graph_configs[page_name] = config
    
    def get_graph_config(self, page_name: str) -> dict:
        """
        Get graph configuration for a page.
        
        Args:
            page_name: Name of the page
        
        Returns:
            Dictionary with graph configuration
        """
        return self.graph_configs.get(page_name, {})


class ResizableGraphWidget(QWidget):
    """
    Widget that automatically resizes graphs based on available space.
    """
    size_changed = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_manager = GraphManager()
        self._last_size = None
    
    def resizeEvent(self, event):
        """Handle resize events to update graph sizes."""
        new_size = event.size()
        if self._last_size != new_size:
            self.size_changed.emit(new_size.width(), new_size.height())
            self._last_size = new_size
        super().resizeEvent(event)
    
    def calculate_figure_size(self, available_width: int, available_height: int,
                             min_width: int = None, min_height: int = None) -> tuple:
        """Calculate optimal figure size for this widget."""
        return self.graph_manager.calculate_optimal_size(
            available_width, available_height, min_width, min_height
        )



