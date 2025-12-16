"""
Model architecture visualization utilities.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from config import COLORS


def visualize_model_architecture(model, ax=None, figsize=(10, 8)):
    """
    Visualize neural network architecture.
    
    Args:
        model: PyTorch model
        ax: Matplotlib axes (creates new if None)
        figsize: Figure size if creating new figure
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['bg_tertiary'])
        ax.set_facecolor(COLORS['bg_tertiary'])
    
    # Extract layer information
    layers = []
    layer_names = []
    
    # Input layer
    layers.append(3)  # a, b, c
    layer_names.append('Input\n(a, b, c)')
    
    # Hidden layers
    for name, module in model.named_modules():
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            if 'Linear' in str(type(module)):
                layers.append(module.out_features)
                layer_names.append(f'Hidden\n{module.out_features} units')
    
    # Output layer
    layers.append(4)  # r1_real, r2_real, r1_imag, r2_imag
    layer_names.append('Output\n(roots)')
    
    # Calculate positions
    num_layers = len(layers)
    layer_width = 0.15
    layer_spacing = 0.8 / (num_layers - 1) if num_layers > 1 else 0.8
    
    # Draw layers
    for i, (size, name) in enumerate(zip(layers, layer_names)):
        x = 0.1 + i * layer_spacing
        y_center = 0.5
        height = size * 0.02
        
        # Layer box
        box = FancyBboxPatch(
            (x - layer_width/2, y_center - height/2),
            layer_width, height,
            boxstyle="round,pad=0.01",
            edgecolor=COLORS['accent_primary'],
            facecolor=COLORS['bg_secondary'],
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)
        
        # Layer label
        ax.text(x, y_center, name, ha='center', va='center',
               color=COLORS['text_primary'], fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_tertiary'], alpha=0.8))
        
        # Draw connections to next layer
        if i < num_layers - 1:
            next_x = 0.1 + (i + 1) * layer_spacing
            next_size = layers[i + 1]
            next_height = next_size * 0.02
            
            # Draw connections
            for j in range(min(size, 20)):  # Limit connections for visibility
                y1 = y_center - height/2 + (j + 0.5) * (height / min(size, 20))
                for k in range(min(next_size, 20)):
                    y2 = y_center - next_height/2 + (k + 0.5) * (next_height / min(next_size, 20))
                    
                    arrow = FancyArrowPatch(
                        (x + layer_width/2, y1),
                        (next_x - layer_width/2, y2),
                        arrowstyle='->',
                        color=COLORS['accent_primary'],
                        alpha=0.3,
                        linewidth=0.5,
                        zorder=1
                    )
                    ax.add_patch(arrow)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Model Architecture', color=COLORS['accent_primary'], 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add info box
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info_text = f"Total Parameters: {param_count:,}"
    ax.text(0.5, 0.05, info_text, ha='center', va='bottom',
           color=COLORS['text_secondary'], fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_secondary'], 
                    edgecolor=COLORS['accent_primary'], linewidth=1))
    
    return ax
