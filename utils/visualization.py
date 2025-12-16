"""
Advanced visualization utilities for optimization results and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from config import COLORS
import optuna

# Set dark theme for all plots
plt.rcParams['figure.facecolor'] = COLORS['bg_tertiary']
plt.rcParams['axes.facecolor'] = COLORS['bg_tertiary']
plt.rcParams['axes.edgecolor'] = COLORS['text_primary']
plt.rcParams['axes.labelcolor'] = COLORS['text_primary']
plt.rcParams['xtick.color'] = COLORS['text_primary']
plt.rcParams['ytick.color'] = COLORS['text_primary']
plt.rcParams['text.color'] = COLORS['text_primary']
plt.rcParams['grid.color'] = COLORS['accent_primary']
plt.rcParams['grid.alpha'] = 0.3

def plot_optimization_convergence(trial_history: List[Dict], ax=None, show_best_so_far=True):
    """
    Plot optimization convergence with all trials and best-so-far line.
    
    Args:
        trial_history: List of dicts with 'trial', 'loss', 'params'
        ax: Matplotlib axes (creates new if None)
        show_best_so_far: Whether to show best-so-far line
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg_tertiary'])
        ax.set_facecolor(COLORS['bg_tertiary'])
    
    if not trial_history:
        return ax, cbar if 'cbar' in locals() else None
    
    trials = sorted(trial_history, key=lambda x: x.get('trial', 0))
    trial_nums = [t.get('trial', i) for i, t in enumerate(trials)]
    losses = [t.get('loss', 0) for t in trials]
    
    # Plot all trials
    ax.scatter(trial_nums, losses, alpha=0.5, color=COLORS['accent_primary'], 
              s=30, label='All Trials', zorder=2)
    
    # Plot best-so-far
    if show_best_so_far and losses:
        best_so_far = []
        best_value = float('inf')
        for loss in losses:
            if loss < best_value:
                best_value = loss
            best_so_far.append(best_value)
        
        ax.plot(trial_nums, best_so_far, color=COLORS['accent_success'], 
               linewidth=2.5, linestyle='--', label='Best So Far', zorder=3)
    
    ax.set_xlabel('Trial Number', color=COLORS['text_primary'], fontsize=11)
    ax.set_ylabel('Validation Loss', color=COLORS['text_primary'], fontsize=11)
    ax.set_title('Optimization Convergence', color=COLORS['accent_primary'], 
                fontsize=14, fontweight='bold')
    ax.tick_params(colors=COLORS['text_primary'])
    ax.grid(True, alpha=0.3, color=COLORS['accent_primary'])
    ax.legend(facecolor=COLORS['bg_secondary'], edgecolor=COLORS['accent_primary'],
             labelcolor=COLORS['text_primary'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    
    return ax


def plot_hyperparameter_importance(study, ax=None, top_n=10):
    """
    Plot hyperparameter importance from Optuna study.
    
    Args:
        study: Optuna study object
        ax: Matplotlib axes (creates new if None)
        top_n: Number of top parameters to show
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg_tertiary'])
        ax.set_facecolor(COLORS['bg_tertiary'])
    
    try:
        importance = optuna.importance.get_param_importances(study)
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        param_names = [p[0] for p in sorted_params]
        importance_values = [p[1] for p in sorted_params]
        
        y_pos = np.arange(len(param_names))
        bars = ax.barh(y_pos, importance_values, color=COLORS['accent_primary'], alpha=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_values)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                   va='center', color=COLORS['text_primary'], fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names, color=COLORS['text_primary'])
        ax.set_xlabel('Importance Score', color=COLORS['text_primary'], fontsize=11)
        ax.set_title('Hyperparameter Importance', color=COLORS['accent_primary'], 
                    fontsize=14, fontweight='bold')
        ax.tick_params(colors=COLORS['text_primary'])
        ax.grid(True, axis='x', alpha=0.3, color=COLORS['accent_primary'])
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Importance calculation unavailable\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes,
               color=COLORS['text_secondary'], fontsize=10)
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    
    return ax


def plot_hyperparameter_heatmap(study, param1: str, param2: str, ax=None):
    """
    Plot 2D heatmap of hyperparameter space.
    
    Args:
        study: Optuna study object
        param1: First hyperparameter name
        param2: Second hyperparameter name
        ax: Matplotlib axes (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['bg_tertiary'])
        ax.set_facecolor(COLORS['bg_tertiary'])
    
    try:
        trials = study.trials
        param1_vals = []
        param2_vals = []
        losses = []
        
        for trial in trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                if param1 in trial.params and param2 in trial.params:
                    val1 = trial.params[param1]
                    val2 = trial.params[param2]
                    # Only include numerical values
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        param1_vals.append(float(val1))
                        param2_vals.append(float(val2))
                        losses.append(trial.value)
        
        if param1_vals and param2_vals and len(param1_vals) > 0:
            # Ensure we have numerical arrays
            param1_vals = np.array(param1_vals, dtype=float)
            param2_vals = np.array(param2_vals, dtype=float)
            losses = np.array(losses, dtype=float)
            
            # Create 2D histogram
            hist, xedges, yedges = np.histogram2d(
                param1_vals, param2_vals, bins=15, weights=losses
            )
            counts, _, _ = np.histogram2d(param1_vals, param2_vals, bins=15)
            
            # Average loss per bin
            with np.errstate(divide='ignore', invalid='ignore'):
                hist = np.divide(hist, counts)
                hist = np.nan_to_num(hist)
            
            # Plot heatmap
            im = ax.imshow(hist.T, origin='lower', aspect='auto',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          cmap='viridis', interpolation='bilinear')
            
            ax.set_xlabel(param1, color=COLORS['text_primary'], fontsize=11)
            ax.set_ylabel(param2, color=COLORS['text_primary'], fontsize=11)
            ax.set_title(f'Loss Heatmap: {param1} vs {param2}',
                        color=COLORS['accent_primary'], fontsize=14, fontweight='bold')
            ax.tick_params(colors=COLORS['text_primary'])
            
            # Use figure from axes to avoid colorbar warning
            fig = ax.figure
            # Remove any existing colorbars from this figure (they're separate axes)
            for i in reversed(range(len(fig.axes))):
                if fig.axes[i] is not ax:
                    # Colorbar axes are typically smaller and positioned differently
                    # Remove any axes that isn't the main plot axes
                    try:
                        fig.axes[i].remove()
                    except:
                        pass
            # Create colorbar with proper spacing to prevent plot shrinking
            # Get current axes position before adding colorbar
            pos = ax.get_position()
            # Create colorbar with fixed positioning
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Average Loss', color=COLORS['text_primary'])
            cbar.ax.tick_params(colors=COLORS['text_primary'])
            # Restore original axes position to prevent shrinking
            ax.set_position(pos)
            # Set fixed colorbar position to prevent it from moving
            # Calculate colorbar position relative to axes (right side with small gap)
            from matplotlib.transforms import Bbox
            cbar_left = pos.x1 + 0.02
            cbar_bottom = pos.y0
            cbar_width = 0.02
            cbar_height = pos.height
            cbar_bbox = Bbox.from_bounds(cbar_left, cbar_bottom, cbar_width, cbar_height)
            cbar.ax.set_position(cbar_bbox)
        elif len(param1_vals) == 0 or len(param2_vals) == 0:
            # Check if parameters are categorical
            sample_trial = next((t for t in trials if t.state == optuna.trial.TrialState.COMPLETE), None)
            if sample_trial:
                val1 = sample_trial.params.get(param1)
                val2 = sample_trial.params.get(param2)
                if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                    ax.text(0.5, 0.5, f'Heatmap requires numerical parameters.\nSelected: {param1}, {param2}\nPlease choose numerical parameters only.',
                           ha='center', va='center', transform=ax.transAxes,
                           color=COLORS['accent_warning'], fontsize=11)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for heatmap',
                           ha='center', va='center', transform=ax.transAxes,
                           color=COLORS['text_secondary'])
            else:
                ax.text(0.5, 0.5, 'Insufficient data for heatmap',
                       ha='center', va='center', transform=ax.transAxes,
                       color=COLORS['text_secondary'])
    except Exception as e:
        ax.text(0.5, 0.5, f'Heatmap unavailable\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes,
               color=COLORS['text_secondary'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    
    # Return axes and colorbar (if created)
    cbar = locals().get('cbar', None)
    return ax, cbar

def plot_pareto_frontier(pareto_front: List[Dict], ax=None):
    """
    Plot Pareto frontier for multi-objective optimization.
    
    Args:
        pareto_front: List of Pareto solutions with 'loss' and 'model_size_mb'
        ax: Matplotlib axes (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg_tertiary'])
        ax.set_facecolor(COLORS['bg_tertiary'])
    
    if not pareto_front:
        return ax
    
    losses = [p['loss'] for p in pareto_front]
    sizes = [p['model_size_mb'] for p in pareto_front]
    
    # Sort by loss
    sorted_indices = np.argsort(losses)
    losses_sorted = [losses[i] for i in sorted_indices]
    sizes_sorted = [sizes[i] for i in sorted_indices]
    
    # Plot Pareto frontier
    ax.plot(losses_sorted, sizes_sorted, 'o-', color=COLORS['accent_primary'],
           linewidth=2, markersize=8, label='Pareto Frontier', zorder=3)
    ax.scatter(losses, sizes, s=100, alpha=0.7, color=COLORS['accent_success'],
              zorder=4, edgecolors=COLORS['accent_primary'], linewidths=1.5)
    
    ax.set_xlabel('Validation Loss', color=COLORS['text_primary'], fontsize=11)
    ax.set_ylabel('Model Size (MB)', color=COLORS['text_primary'], fontsize=11)
    ax.set_title('Pareto Frontier: Loss vs Model Size',
                color=COLORS['accent_primary'], fontsize=14, fontweight='bold')
    ax.tick_params(colors=COLORS['text_primary'])
    ax.grid(True, alpha=0.3, color=COLORS['accent_primary'])
    ax.legend(facecolor=COLORS['bg_secondary'], edgecolor=COLORS['accent_primary'],
             labelcolor=COLORS['text_primary'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    
    return ax


def plot_error_analysis(error_analysis: Dict, ax=None):
    """
    Plot error analysis by root type.
    
    Args:
        error_analysis: Dictionary with error metrics by root type
        ax: Matplotlib axes (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg_tertiary'])
        ax.set_facecolor(COLORS['bg_tertiary'])
    
    root_types = []
    mae_values = []
    accuracy_values = []
    
    for root_type, metrics in error_analysis.items():
        if 'mae' in metrics and 'accuracy' in metrics and metrics.get('count', 0) > 0:
            root_types.append(root_type.replace('_', ' ').title())
            mae_values.append(metrics['mae'])
            accuracy_values.append(metrics['accuracy'])
    
    if root_types:
        x = np.arange(len(root_types))
        width = 0.35
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - width/2, mae_values, width, label='MAE',
                      color=COLORS['accent_primary'], alpha=0.8)
        bars2 = ax2.bar(x + width/2, accuracy_values, width, label='Accuracy (%)',
                       color=COLORS['accent_success'], alpha=0.8)
        
        ax.set_xlabel('Root Type', color=COLORS['text_primary'], fontsize=11)
        ax.set_ylabel('MAE', color=COLORS['text_primary'], fontsize=11)
        ax2.set_ylabel('Accuracy (%)', color=COLORS['text_primary'], fontsize=11)
        ax.set_title('Error Analysis by Root Type',
                    color=COLORS['accent_primary'], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(root_types, color=COLORS['text_primary'])
        ax.tick_params(colors=COLORS['text_primary'])
        ax2.tick_params(colors=COLORS['text_primary'])
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['accent_primary'])
        
        ax.legend(loc='upper left', facecolor=COLORS['bg_secondary'],
                 edgecolor=COLORS['accent_primary'], labelcolor=COLORS['text_primary'])
        ax2.legend(loc='upper right', facecolor=COLORS['bg_secondary'],
                  edgecolor=COLORS['accent_success'], labelcolor=COLORS['text_primary'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    if 'ax2' in locals():
        for spine in ax2.spines.values():
            spine.set_color(COLORS['text_primary'])
    
    return ax


def plot_quadratic_with_roots(a, b, c, true_roots, pred_roots, ax=None):
    """
    Plot quadratic equation curve with true and predicted roots marked.
    
    Args:
        a, b, c: Coefficients
        true_roots: Tuple of (root1, root2) or complex numbers
        pred_roots: Tuple of predicted (r1_real, r2_real, r1_imag, r2_imag)
        ax: Matplotlib axes (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg_tertiary'])
        ax.set_facecolor(COLORS['bg_tertiary'])
    
    # Generate curve
    x = np.linspace(-10, 10, 1000)
    y = a * x**2 + b * x + c
    ax.plot(x, y, color=COLORS['accent_primary'], linewidth=2, label=f'{a}x² + {b}x + {c}')
    
    # Mark true roots
    if isinstance(true_roots[0], complex):
        if abs(true_roots[0].imag) < 1e-6:
            # Real roots
            ax.scatter([true_roots[0].real, true_roots[1].real], [0, 0],
                      s=200, color=COLORS['accent_success'], marker='o',
                      label='True Roots', zorder=5, edgecolors='white', linewidths=2)
    else:
        ax.scatter([true_roots[0], true_roots[1]], [0, 0],
                  s=200, color=COLORS['accent_success'], marker='o',
                  label='True Roots', zorder=5, edgecolors='white', linewidths=2)
    
    # Mark predicted roots as dots
    r1_real, r2_real, r1_imag, r2_imag = pred_roots
    if abs(r1_imag) < 1e-6 and abs(r2_imag) < 1e-6:
        # Real predicted roots - show as dots
        pred_x = [r1_real, r2_real]
        pred_y = [0, 0]
        ax.scatter(pred_x, pred_y, s=200, color=COLORS['accent_warning'], marker='o',
                  label='Predicted Roots', zorder=5, edgecolors='white', linewidths=2, alpha=0.8)
        
        # Draw error lines connecting true to predicted roots
        if isinstance(true_roots[0], complex):
            if abs(true_roots[0].imag) < 1e-6:
                true_x = [true_roots[0].real, true_roots[1].real]
                # Match closest pairs and draw error lines
                for tx in true_x:
                    closest_pred = min(pred_x, key=lambda px: abs(px - tx))
                    error_val = abs(closest_pred - tx)
                    # Draw error line
                    ax.plot([tx, closest_pred], [0, 0], 
                           color=COLORS['accent_error'], linestyle='--', linewidth=2, 
                           alpha=0.6, zorder=3)
                    # Add error annotation
                    mid_x = (tx + closest_pred) / 2
                    ax.annotate(f'{error_val:.3f}', (mid_x, 0.1),
                               ha='center', color=COLORS['accent_error'], fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_secondary'],
                                        edgecolor=COLORS['accent_error'], alpha=0.7))
        else:
            true_x = [true_roots[0], true_roots[1]]
            # Match closest pairs and draw error lines
            for tx in true_x:
                closest_pred = min(pred_x, key=lambda px: abs(px - tx))
                error_val = abs(closest_pred - tx)
                # Draw error line
                ax.plot([tx, closest_pred], [0, 0], 
                       color=COLORS['accent_error'], linestyle='--', linewidth=2, 
                       alpha=0.6, zorder=3)
                # Add error annotation
                mid_x = (tx + closest_pred) / 2
                ax.annotate(f'{error_val:.3f}', (mid_x, 0.1),
                           ha='center', color=COLORS['accent_error'], fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_secondary'],
                                    edgecolor=COLORS['accent_error'], alpha=0.7))
    
    # Zero line
    ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color=COLORS['text_secondary'], linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('x', color=COLORS['text_primary'], fontsize=11)
    ax.set_ylabel('y', color=COLORS['text_primary'], fontsize=11)
    ax.set_title('Quadratic Equation with Roots',
                color=COLORS['accent_primary'], fontsize=14, fontweight='bold')
    ax.tick_params(colors=COLORS['text_primary'])
    ax.grid(True, alpha=0.3, color=COLORS['accent_primary'])
    ax.legend(facecolor=COLORS['bg_secondary'], edgecolor=COLORS['accent_primary'],
             labelcolor=COLORS['text_primary'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['text_primary'])
    
    return ax
