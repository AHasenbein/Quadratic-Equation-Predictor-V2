"""
Advanced visualization utilities for optimization results, hyperparameter analysis,
and model comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from config import COLORS
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


def plot_hyperparameter_importance(
    importance_dict: Dict[str, float],
    ax=None,
    top_n: int = 10
) -> plt.Axes:
    """
    Plot hyperparameter importance scores.
    
    Args:
        importance_dict: Dictionary of {param_name: importance_score}
        ax: Matplotlib axes (creates new if None)
        top_n: Number of top parameters to show
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by importance
    sorted_params = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    param_names = [p[0] for p in sorted_params]
    importance_values = [p[1] for p in sorted_params]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(param_names))
    bars = ax.barh(y_pos, importance_values, color=COLORS['accent_primary'], alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Importance Score', color=COLORS['text_primary'], fontsize=11)
    ax.set_title('Hyperparameter Importance', color=COLORS['accent_primary'], fontweight='bold', fontsize=12)
    ax.set_facecolor(COLORS['bg_tertiary'])
    ax.tick_params(colors=COLORS['text_primary'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    
    ax.grid(True, axis='x', color=COLORS['graph_grid'], alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    return ax


def plot_optimization_convergence(
    trial_history: List[Dict],
    ax=None,
    metric: str = 'value'
) -> plt.Axes:
    """
    Plot optimization convergence over trials.
    
    Args:
        trial_history: List of trial dictionaries with 'number' and metric
        ax: Matplotlib axes (creates new if None)
        metric: Metric to plot ('value', 'loss', etc.)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    trial_numbers = [t['number'] for t in trial_history if metric in t]
    values = [t[metric] for t in trial_history if metric in t]
    
    # Plot line
    ax.plot(trial_numbers, values, color=COLORS['graph_line'], linewidth=2, alpha=0.7, label='Trial Value')
    
    # Plot best so far
    if values:
        best_so_far = []
        best_value = float('inf')
        for v in values:
            if v < best_value:
                best_value = v
            best_so_far.append(best_value)
        
        ax.plot(trial_numbers, best_so_far, color=COLORS['accent_success'], 
               linewidth=2, linestyle='--', label='Best So Far')
    
    ax.set_xlabel('Trial Number', color=COLORS['text_primary'], fontsize=11)
    ax.set_ylabel('Objective Value', color=COLORS['text_primary'], fontsize=11)
    ax.set_title('Optimization Convergence', color=COLORS['accent_primary'], fontweight='bold', fontsize=12)
    ax.set_facecolor(COLORS['bg_tertiary'])
    ax.tick_params(colors=COLORS['text_primary'])
    ax.legend(facecolor=COLORS['bg_secondary'], edgecolor=COLORS['accent_primary'], 
             labelcolor=COLORS['text_primary'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    
    ax.grid(True, color=COLORS['graph_grid'], alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    return ax


def plot_pareto_frontier(
    pareto_solutions: List[Dict],
    objective1: str,
    objective2: str,
    ax=None
) -> plt.Axes:
    """
    Plot Pareto frontier for multi-objective optimization.
    
    Args:
        pareto_solutions: List of Pareto solutions with objectives
        objective1: Name of first objective
        objective2: Name of second objective
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    obj1_values = [s['objectives'].get(objective1, 0) for s in pareto_solutions]
    obj2_values = [s['objectives'].get(objective2, 0) for s in pareto_solutions]
    
    # Sort by first objective for line plot
    sorted_indices = np.argsort(obj1_values)
    obj1_sorted = [obj1_values[i] for i in sorted_indices]
    obj2_sorted = [obj2_values[i] for i in sorted_indices]
    
    # Plot Pareto frontier
    ax.plot(obj1_sorted, obj2_sorted, 'o-', color=COLORS['accent_primary'], 
           linewidth=2, markersize=8, label='Pareto Frontier')
    ax.scatter(obj1_values, obj2_values, color=COLORS['accent_success'], 
              s=100, alpha=0.7, zorder=5)
    
    ax.set_xlabel(objective1, color=COLORS['text_primary'], fontsize=11)
    ax.set_ylabel(objective2, color=COLORS['text_primary'], fontsize=11)
    ax.set_title('Pareto Frontier', color=COLORS['accent_primary'], fontweight='bold', fontsize=12)
    ax.set_facecolor(COLORS['bg_tertiary'])
    ax.tick_params(colors=COLORS['text_primary'])
    ax.legend(facecolor=COLORS['bg_secondary'], edgecolor=COLORS['accent_primary'], 
             labelcolor=COLORS['text_primary'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    
    ax.grid(True, color=COLORS['graph_grid'], alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    return ax


def plot_tradeoff_analysis(
    experiments: List[Dict],
    metric1: str,
    metric2: str,
    ax=None,
    label_key: Optional[str] = None
) -> plt.Axes:
    """
    Plot trade-off between two metrics across experiments.
    
    Args:
        experiments: List of experiment results
        metric1: First metric name (e.g., 'accuracy')
        metric2: Second metric name (e.g., 'model_size_mb')
        ax: Matplotlib axes (creates new if None)
        label_key: Optional key in experiment dict for labeling points
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    values1 = []
    values2 = []
    labels = []
    
    for exp in experiments:
        # Try to get metrics from different locations
        val1 = None
        val2 = None
        
        if 'validation_metrics' in exp:
            val1 = exp['validation_metrics'].get(metric1)
            val2 = exp['validation_metrics'].get(metric2)
        
        if val1 is None and 'efficiency' in exp:
            val1 = exp['efficiency'].get(metric1)
        if val2 is None and 'efficiency' in exp:
            val2 = exp['efficiency'].get(metric2)
        
        if val1 is not None and val2 is not None:
            values1.append(val1)
            values2.append(val2)
            if label_key:
                labels.append(str(exp.get(label_key, '')))
    
    if values1 and values2:
        scatter = ax.scatter(values1, values2, c=COLORS['accent_primary'], 
                           s=100, alpha=0.7, edgecolors=COLORS['accent_secondary'], linewidths=2)
        
        # Add labels if provided
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (values1[i], values2[i]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, color=COLORS['text_primary'])
    
    ax.set_xlabel(metric1, color=COLORS['text_primary'], fontsize=11)
    ax.set_ylabel(metric2, color=COLORS['text_primary'], fontsize=11)
    ax.set_title(f'Trade-off: {metric1} vs {metric2}', 
                color=COLORS['accent_primary'], fontweight='bold', fontsize=12)
    ax.set_facecolor(COLORS['bg_tertiary'])
    ax.tick_params(colors=COLORS['text_primary'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    
    ax.grid(True, color=COLORS['graph_grid'], alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    return ax


def plot_hyperparameter_heatmap(
    experiments: List[Dict],
    param1: str,
    param2: str,
    metric: str = 'loss',
    ax=None
) -> plt.Axes:
    """
    Plot 2D heatmap of hyperparameter space.
    
    Args:
        experiments: List of experiment results
        param1: First hyperparameter name
        param2: Second hyperparameter name
        metric: Metric to visualize
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    param1_values = []
    param2_values = []
    metric_values = []
    
    for exp in experiments:
        if 'hyperparameters' in exp:
            p1 = exp['hyperparameters'].get(param1)
            p2 = exp['hyperparameters'].get(param2)
            
            if p1 is not None and p2 is not None:
                m = exp.get('validation_metrics', {}).get(metric)
                if m is not None:
                    param1_values.append(p1)
                    param2_values.append(p2)
                    metric_values.append(m)
    
    if param1_values and param2_values:
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(
            param1_values, param2_values, bins=20, weights=metric_values
        )
        counts, _, _ = np.histogram2d(param1_values, param2_values, bins=20)
        
        # Average metric per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            hist = np.divide(hist, counts)
            hist = np.nan_to_num(hist)
        
        # Plot heatmap
        im = ax.imshow(hist.T, origin='lower', aspect='auto', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      cmap='viridis', interpolation='nearest')
        
        ax.set_xlabel(param1, color=COLORS['text_primary'], fontsize=11)
        ax.set_ylabel(param2, color=COLORS['text_primary'], fontsize=11)
        ax.set_title(f'{metric} Heatmap: {param1} vs {param2}', 
                    color=COLORS['accent_primary'], fontweight='bold', fontsize=12)
        ax.set_facecolor(COLORS['bg_tertiary'])
        ax.tick_params(colors=COLORS['text_primary'])
        
        plt.colorbar(im, ax=ax, label=metric)
    
    return ax


def plot_error_analysis(
    error_analysis: Dict,
    ax=None
) -> plt.Axes:
    """
    Plot error analysis by root type.
    
    Args:
        error_analysis: Dictionary with error metrics by root type
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    root_types = []
    mae_values = []
    accuracy_values = []
    
    for root_type, metrics in error_analysis.items():
        if 'mae' in metrics and 'accuracy' in metrics:
            root_types.append(root_type.replace('_', ' ').title())
            mae_values.append(metrics['mae'])
            accuracy_values.append(metrics['accuracy'])
    
    if root_types:
        x = np.arange(len(root_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mae_values, width, label='MAE', 
                      color=COLORS['accent_primary'], alpha=0.8)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, accuracy_values, width, label='Accuracy (%)', 
                       color=COLORS['accent_success'], alpha=0.8)
        
        ax.set_xlabel('Root Type', color=COLORS['text_primary'], fontsize=11)
        ax.set_ylabel('MAE', color=COLORS['text_primary'], fontsize=11)
        ax2.set_ylabel('Accuracy (%)', color=COLORS['text_primary'], fontsize=11)
        ax.set_title('Error Analysis by Root Type', 
                    color=COLORS['accent_primary'], fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(root_types, color=COLORS['text_primary'])
        ax.tick_params(colors=COLORS['text_primary'])
        ax2.tick_params(colors=COLORS['text_primary'])
        ax.set_facecolor(COLORS['bg_tertiary'])
        
        ax.legend(loc='upper left', facecolor=COLORS['bg_secondary'], 
                 edgecolor=COLORS['accent_primary'], labelcolor=COLORS['text_primary'])
        ax2.legend(loc='upper right', facecolor=COLORS['bg_secondary'], 
                  edgecolor=COLORS['accent_success'], labelcolor=COLORS['text_primary'])
        
        for spine in ax.spines.values():
            spine.set_color(COLORS['text_primary'])
        for spine in ax2.spines.values():
            spine.set_color(COLORS['text_primary'])
    
    return ax


def plot_architecture_comparison(
    architectures: List[Dict],
    metric: str = 'accuracy',
    ax=None
) -> plt.Axes:
    """
    Plot comparison of different architectures.
    
    Args:
        architectures: List of architecture results with metrics
        metric: Metric to compare
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    arch_names = []
    metric_values = []
    param_counts = []
    
    for arch in architectures:
        name = arch.get('name', 'Architecture')
        arch_names.append(name)
        
        # Get metric value
        val = arch.get('metrics', {}).get(metric)
        if val is None:
            val = arch.get(metric, 0)
        metric_values.append(val)
        
        # Get parameter count for size
        params = arch.get('param_count', arch.get('params', 0))
        param_counts.append(params)
    
    if arch_names:
        x = np.arange(len(arch_names))
        bars = ax.bar(x, metric_values, color=COLORS['accent_primary'], alpha=0.8)
        
        # Add parameter count as text
        for i, (bar, params) in enumerate(zip(bars, param_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{params:,} params',
                   ha='center', va='bottom', fontsize=8, color=COLORS['text_secondary'])
        
        ax.set_xlabel('Architecture', color=COLORS['text_primary'], fontsize=11)
        ax.set_ylabel(metric, color=COLORS['text_primary'], fontsize=11)
        ax.set_title(f'Architecture Comparison: {metric}', 
                    color=COLORS['accent_primary'], fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(arch_names, rotation=45, ha='right', color=COLORS['text_primary'])
        ax.tick_params(colors=COLORS['text_primary'])
        ax.set_facecolor(COLORS['bg_tertiary'])
        
        for spine in ax.spines.values():
            spine.set_color(COLORS['text_primary'])
        
        ax.grid(True, axis='y', color=COLORS['graph_grid'], alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    return ax
