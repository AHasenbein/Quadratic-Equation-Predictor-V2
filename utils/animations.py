"""
Utility functions for animations and visual feedback.
"""

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, Qt
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtGui import QFont
from config import COLORS, ANIMATION_DURATION, ANIMATION_EASING


def fade_in_widget(widget: QWidget, duration: int = None):
    """
    Create a fade-in animation for a widget.
    
    Args:
        widget: Widget to animate
        duration: Animation duration in ms (defaults to ANIMATION_DURATION)
    """
    widget.setGraphicsEffect(None)  # Remove any existing effects
    animation = QPropertyAnimation(widget, b"windowOpacity")
    animation.setDuration(duration or ANIMATION_DURATION)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setEasingCurve(getattr(QEasingCurve.Type, ANIMATION_EASING, QEasingCurve.Type.EaseInOut))
    animation.start()
    return animation


def create_loading_indicator(parent: QWidget, text: str = "Loading...") -> QLabel:
    """
    Create a loading indicator label.
    
    Args:
        parent: Parent widget
        text: Loading text
    
    Returns:
        QLabel with loading indicator styling
    """
    label = QLabel(text, parent)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
    label.setStyleSheet(f"""
        QLabel {{
            background-color: {COLORS['bg_secondary']};
            border: 2px solid {COLORS['accent_primary']};
            border-radius: 10px;
            padding: 20px;
            color: {COLORS['accent_primary']};
        }}
    """)
    return label


def create_notification_toast(parent: QWidget, message: str, 
                              notification_type: str = "info") -> QLabel:
    """
    Create a notification toast message.
    
    Args:
        parent: Parent widget
        message: Message text
        notification_type: 'info', 'success', 'warning', or 'error'
    
    Returns:
        QLabel styled as notification toast
    """
    color_map = {
        'info': COLORS['accent_primary'],
        'success': COLORS['accent_success'],
        'warning': COLORS['accent_warning'],
        'error': COLORS['accent_error']
    }
    
    color = color_map.get(notification_type, COLORS['accent_primary'])
    
    toast = QLabel(message, parent)
    toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
    toast.setFont(QFont("Arial", 11))
    toast.setStyleSheet(f"""
        QLabel {{
            background-color: {COLORS['bg_secondary']};
            border: 2px solid {color};
            border-radius: 5px;
            padding: 10px 20px;
            color: {color};
            font-weight: bold;
        }}
    """)
    toast.setWordWrap(True)
    
    # Fade in animation
    fade_in_widget(toast)
    
    return toast



