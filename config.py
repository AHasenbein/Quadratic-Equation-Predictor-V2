"""
Configuration file for theme, colors, and app settings.
Easy to modify for customization.
"""

# Color Scheme - Dark theme with glowing effects
COLORS = {
    # Background colors
    'bg_primary': '#0a0e1a',
    'bg_secondary': '#141b2d',
    'bg_tertiary': '#1e2742',
    
    # Accent colors with glow
    'accent_primary': '#00d4ff',
    'accent_secondary': '#7b2cbf',
    'accent_success': '#00ff88',
    'accent_warning': '#ffb800',
    'accent_error': '#ff3366',
    
    # Text colors
    'text_primary': '#ffffff',
    'text_secondary': '#b0b8c4',
    'text_muted': '#6b7280',
    
    # Graph colors
    'graph_line': '#00d4ff',
    'graph_fill': '#00d4ff33',
    'graph_grid': '#1e2742',
    
    # Glow effects (for use in stylesheets)
    'glow_primary': 'rgba(0, 212, 255, 0.5)',
    'glow_success': 'rgba(0, 255, 136, 0.5)',
    'glow_warning': 'rgba(255, 184, 0, 0.5)',
    'glow_error': 'rgba(255, 51, 102, 0.5)',
}

# Animation settings
ANIMATION_DURATION = 300  # milliseconds
ANIMATION_EASING = 'easeInOut'

# Window settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 700

# Page settings
PAGE_PADDING = 20
GRAPH_HEIGHT = 300
GRAPH_MARGIN = 10

# Responsive sizing constants
GRAPH_MIN_WIDTH = 800
GRAPH_MIN_HEIGHT = 500
GRAPH_ASPECT_RATIO = 16 / 9
SCROLL_AREA_PADDING = 15
CONTENT_MAX_WIDTH = 1400

# Spacing and padding constants
SECTION_SPACING = 20  # Between major sections
ELEMENT_SPACING = 10  # Between form elements
GROUPBOX_PADDING = 15
BUTTON_PADDING = 12
INPUT_PADDING = 10

# Graph styling constants
GRAPH_TITLE_FONT_SIZE = 14
GRAPH_LABEL_FONT_SIZE = 11
GRAPH_TICK_FONT_SIZE = 9
GRAPH_LINE_WIDTH = 2
GRAPH_GRID_LINE_WIDTH = 1

# Model presets
MODEL_PRESETS = {
    'Tiny': {
        'hidden_layers': [8, 4],
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 50,
    },
    'Small': {
        'hidden_layers': [16, 8],
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100,
    },
    'Medium': {
        'hidden_layers': [32, 16, 8],
        'learning_rate': 0.0005,
        'batch_size': 128,
        'epochs': 200,
    },
    'Large': {
        'hidden_layers': [64, 32, 16],
        'learning_rate': 0.0001,
        'batch_size': 256,
        'epochs': 300,
    },
}


def calculate_graph_size(available_width: int, available_height: int, 
                        min_width: int = None, min_height: int = None,
                        aspect_ratio: float = None) -> tuple:
    """
    Calculate optimal graph size based on available space.
    
    Args:
        available_width: Available width in pixels
        available_height: Available height in pixels
        min_width: Minimum width (defaults to GRAPH_MIN_WIDTH)
        min_height: Minimum height (defaults to GRAPH_MIN_HEIGHT)
        aspect_ratio: Desired aspect ratio (defaults to GRAPH_ASPECT_RATIO)
    
    Returns:
        Tuple of (width, height) in pixels
    """
    min_w = min_width or GRAPH_MIN_WIDTH
    min_h = min_height or GRAPH_MIN_HEIGHT
    aspect = aspect_ratio or GRAPH_ASPECT_RATIO
    
    # Account for padding
    padding = SCROLL_AREA_PADDING * 2
    available_width = max(available_width - padding, min_w)
    available_height = max(available_height - padding, min_h)
    
    # Calculate size maintaining aspect ratio
    width = available_width
    height = available_height
    
    # Adjust to maintain aspect ratio
    if width / height > aspect:
        width = int(height * aspect)
    else:
        height = int(width / aspect)
    
    # Ensure minimum sizes
    width = max(width, min_w)
    height = max(height, min_h)
    
    return (width, height)


def get_screen_size_category(width: int, height: int) -> str:
    """
    Determine screen size category for responsive design.
    
    Args:
        width: Screen width
        height: Screen height
    
    Returns:
        'small', 'medium', or 'large'
    """
    if width < 1400 or height < 800:
        return 'small'
    elif width < 1920 or height < 1080:
        return 'medium'
    else:
        return 'large'

