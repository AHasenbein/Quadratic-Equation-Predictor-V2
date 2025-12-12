"""
Base page class for all pages in the application.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QScrollArea, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QWheelEvent
from config import COLORS, SCROLL_AREA_PADDING, SECTION_SPACING, ELEMENT_SPACING, PAGE_PADDING


class BasePage(QWidget):
    """
    Base class for all pages. Provides common functionality and styling.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.apply_styles()
    
    def setup_ui(self):
        """Setup the UI layout with global scroll area."""
        # Main layout for this widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create global scroll area for the entire page
        self.scroll_area = SmoothScrollArea()  # Use custom smooth scrolling
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Enable smooth scrolling for Mac trackpad
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_primary']};
                border: none;
            }}
        """)
        
        # Content widget that will be scrollable
        self.content_widget = QWidget()
        # Allow content to grow vertically beyond viewport for scrolling
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setContentsMargins(PAGE_PADDING, PAGE_PADDING, PAGE_PADDING, PAGE_PADDING)
        self.layout.setSpacing(SECTION_SPACING)
        
        # Set the content widget to the scroll area
        self.scroll_area.setWidget(self.content_widget)
        
        # Add scroll area to main layout with expanding policy
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.scroll_area)
    
    def add_stretch(self):
        """Add a stretch item to push content to top."""
        self.layout.addStretch()
    
    def add_spacer(self, height: int = None):
        """Add a fixed spacer item."""
        if height:
            self.layout.addSpacing(height)
        else:
            self.layout.addSpacing(ELEMENT_SPACING)
    
    def get_main_window(self):
        """
        Helper method to traverse up the parent hierarchy to find MainWindow.
        
        Returns:
            MainWindow instance or None if not found
        """
        parent = self.parent()
        while parent:
            # Check if this is the MainWindow (it should have page1, page2, etc. attributes)
            if hasattr(parent, 'page1') and hasattr(parent, 'page2'):
                return parent
            parent = parent.parent()
        return None
    
    def apply_styles(self):
        """Apply base styles to the page."""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
            }}
        """)
    
    def create_graph_scroll_area(self, min_width: int = 800, min_height: int = 500) -> QScrollArea:
        """
        Create a standardized scroll area for graphs.
        
        Args:
            min_width: Minimum width for the scroll area
            min_height: Minimum height for the scroll area
        
        Returns:
            Configured QScrollArea widget
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)  # Important for proper scrolling
        scroll.setMinimumHeight(min_height)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 10px;
                border: none;
            }}
        """)
        return scroll
    
    def add_graph_to_scroll(self, scroll_area: QScrollArea, canvas, 
                           min_width: int = 800, min_height: int = 500):
        """
        Add a graph canvas to a scroll area with proper sizing.
        
        Args:
            scroll_area: The QScrollArea to add the graph to
            canvas: The matplotlib FigureCanvas to add
            min_width: Minimum width for the graph widget
            min_height: Minimum height for the graph widget
        """
        from PyQt6.QtWidgets import QWidget, QVBoxLayout
        
        graph_widget = QWidget()
        graph_widget.setMinimumSize(min_width, min_height)
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(SCROLL_AREA_PADDING, SCROLL_AREA_PADDING, 
                                       SCROLL_AREA_PADDING, SCROLL_AREA_PADDING)
        graph_layout.addWidget(canvas)
        
        scroll_area.setWidget(graph_widget)


class SmoothScrollArea(QScrollArea):
    """
    Custom QScrollArea with smooth scrolling support for Mac trackpad.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable smooth scrolling
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Accept wheel events for smooth scrolling
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
    
    def wheelEvent(self, event: QWheelEvent):
        """
        Override wheel event to enable smooth scrolling on Mac trackpad.
        Mac trackpads send pixelDelta events for smooth scrolling.
        """
        # Check for pixel-based scrolling (Mac trackpad)
        pixel_delta = event.pixelDelta()
        if not pixel_delta.isNull():
            # Mac trackpad - use pixel deltas directly for smooth scrolling
            scroll_bar = self.verticalScrollBar() if pixel_delta.y() != 0 else self.horizontalScrollBar()
            if scroll_bar:
                if pixel_delta.y() != 0:
                    scroll_bar.setValue(scroll_bar.value() - pixel_delta.y())
                elif pixel_delta.x() != 0:
                    scroll_bar = self.horizontalScrollBar()
                    if scroll_bar:
                        scroll_bar.setValue(scroll_bar.value() - pixel_delta.x())
                event.accept()
                return
        
        # Fallback to angle-based scrolling (regular mouse wheel)
        angle_delta = event.angleDelta()
        if not angle_delta.isNull():
            scroll_bar = self.verticalScrollBar() if angle_delta.y() != 0 else self.horizontalScrollBar()
            if scroll_bar:
                # Standard mouse wheel - use angle delta
                delta = angle_delta.y() if angle_delta.y() != 0 else angle_delta.x()
                scroll_bar.setValue(scroll_bar.value() - delta // 8)
                event.accept()
                return
        
        # If we get here, pass to parent
        super().wheelEvent(event)

