"""
Utility functions for standardizing graph appearance across all pages.
"""

import matplotlib.pyplot as plt
from config import (COLORS, GRAPH_TITLE_FONT_SIZE, GRAPH_LABEL_FONT_SIZE, 
                   GRAPH_TICK_FONT_SIZE, GRAPH_LINE_WIDTH, GRAPH_GRID_LINE_WIDTH)


def apply_standard_graph_style(ax, title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Apply standard styling to a matplotlib axes object.
    
    Args:
        ax: Matplotlib axes object
        title: Graph title (optional)
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
    """
    # Set background color
    ax.set_facecolor(COLORS['bg_tertiary'])
    
    # Set title
    if title:
        ax.set_title(title, color=COLORS['accent_primary'], 
                    fontsize=GRAPH_TITLE_FONT_SIZE, fontweight='bold', pad=15)
    
    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS['text_primary'], 
                     fontsize=GRAPH_LABEL_FONT_SIZE, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS['text_primary'], 
                     fontsize=GRAPH_LABEL_FONT_SIZE, labelpad=10)
    
    # Style ticks
    ax.tick_params(colors=COLORS['text_primary'], labelsize=GRAPH_TICK_FONT_SIZE)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
        spine.set_linewidth(1)
    
    # Style grid with major and minor grid lines
    ax.grid(True, color=COLORS['graph_grid'], alpha=0.3, 
           linestyle='--', linewidth=GRAPH_GRID_LINE_WIDTH, which='major')
    ax.grid(True, color=COLORS['graph_grid'], alpha=0.15, 
           linestyle=':', linewidth=GRAPH_GRID_LINE_WIDTH * 0.5, which='minor')
    ax.set_axisbelow(True)  # Grid behind plot elements
    
    # Format axis labels better
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}' if abs(x) < 1000 else f'{x:.1e}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}' if abs(x) < 1000 else f'{x:.1e}'))


def style_legend(ax, loc='best', ncol=1):
    """
    Style the legend with consistent appearance and better positioning.
    
    Args:
        ax: Matplotlib axes object
        loc: Legend location ('best', 'upper right', etc.)
        ncol: Number of columns for legend items
    """
    # Try to position legend to avoid overlap
    if loc == 'best':
        # Use bbox_to_anchor for better control
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                          facecolor=COLORS['bg_secondary'], 
                          edgecolor=COLORS['accent_primary'], 
                          framealpha=0.9, fontsize=GRAPH_TICK_FONT_SIZE,
                          labelcolor=COLORS['text_primary'], ncol=ncol)
    else:
        legend = ax.legend(loc=loc, facecolor=COLORS['bg_secondary'], 
                          edgecolor=COLORS['accent_primary'], 
                          framealpha=0.9, fontsize=GRAPH_TICK_FONT_SIZE,
                          labelcolor=COLORS['text_primary'], ncol=ncol)
    
    if legend:
        legend.get_frame().set_linewidth(1.5)
        legend.get_frame().set_boxstyle('round', pad=0.5)


def add_zero_lines(ax):
    """
    Add zero reference lines to the graph.
    
    Args:
        ax: Matplotlib axes object
    """
    ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='--', 
              alpha=0.5, linewidth=1, zorder=0)
    ax.axvline(x=0, color=COLORS['text_secondary'], linestyle='--', 
              alpha=0.5, linewidth=1, zorder=0)

