"""
Utility functions for standardizing widget appearance across all pages.
"""

from config import (COLORS, GROUPBOX_PADDING, BUTTON_PADDING, INPUT_PADDING,
                   SECTION_SPACING, ELEMENT_SPACING)


def get_groupbox_style(glow_color: str = None) -> str:
    """
    Get standardized GroupBox style with glow effects.
    
    Args:
        glow_color: Color for glow effect (defaults to accent_primary)
    
    Returns:
        CSS style string for GroupBox
    """
    glow = glow_color or COLORS['accent_primary']
    
    return f"""
        QGroupBox {{
            background-color: {COLORS['bg_secondary']};
            border: 2px solid {glow};
            border-radius: 10px;
            padding: {GROUPBOX_PADDING}px;
            margin-top: 10px;
            color: {glow};
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            background-color: {COLORS['bg_secondary']};
        }}
    """


def get_button_style(primary: bool = True, color: str = None) -> str:
    """
    Get standardized button style with gradients and transitions.
    
    Args:
        primary: Whether this is a primary button
        color: Accent color (defaults based on primary)
    
    Returns:
        CSS style string for QPushButton
    """
    if color is None:
        color = COLORS['accent_primary'] if primary else COLORS['bg_tertiary']
    
    bg_color = color if primary else COLORS['bg_tertiary']
    text_color = COLORS['bg_primary'] if primary else COLORS['text_primary']
    hover_color = COLORS['accent_secondary'] if primary else COLORS['bg_secondary']
    border_color = color
    
    return f"""
        QPushButton {{
            background-color: {bg_color};
            color: {text_color};
            border: 2px solid {border_color};
            border-radius: 5px;
            padding: {BUTTON_PADDING}px;
            font-size: 12px;
            font-weight: bold;
            min-height: 40px;
        }}
        QPushButton:hover {{
            background-color: {hover_color};
            border: 2px solid {hover_color};
        }}
        QPushButton:pressed {{
            background-color: {hover_color};
            border: 2px solid {hover_color};
            padding-top: {BUTTON_PADDING + 2}px;
            padding-bottom: {BUTTON_PADDING - 2}px;
        }}
        QPushButton:disabled {{
            background-color: {COLORS['bg_tertiary']};
            color: {COLORS['text_muted']};
            border: 2px solid {COLORS['bg_tertiary']};
        }}
    """


def get_input_style(valid: bool = None) -> str:
    """
    Get standardized input field style with validation indicators.
    
    Args:
        valid: Validation state (None = neutral, True = valid, False = invalid)
    
    Returns:
        CSS style string for QLineEdit/QSpinBox/QDoubleSpinBox
    """
    if valid is True:
        border_color = COLORS['accent_success']
    elif valid is False:
        border_color = COLORS['accent_error']
    else:
        border_color = COLORS['accent_primary']
    
    return f"""
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {COLORS['bg_tertiary']};
            border: 2px solid {border_color};
            border-radius: 5px;
            padding: {INPUT_PADDING}px;
            color: {COLORS['text_primary']};
            font-size: 11px;
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border: 2px solid {COLORS['accent_primary']};
            background-color: {COLORS['bg_secondary']};
        }}
        QLineEdit::placeholder {{
            color: {COLORS['text_muted']};
        }}
    """



