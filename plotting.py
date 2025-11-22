"""
Plotting utilities for interpolation visualization.

This module provides reusable functions for generating plots from interpolation results.
"""

import matplotlib.pyplot as plt
import torch
from typing import Optional, Dict, Any


def plot_interpolation(
    values: torch.Tensor,
    method: str,
    start: float,
    end: float,
    num_points: int,
    output_path: str,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 150,
    **kwargs
) -> None:
    """
    Generate and save a plot for interpolation results.
    
    Args:
        values: Tensor containing interpolated values
        method: Name of the interpolation method used
        start: Starting value for interpolation
        end: Ending value for interpolation
        num_points: Total number of points
        output_path: Path where the plot will be saved
        title: Optional custom title (if None, auto-generated)
        figsize: Figure size tuple (width, height)
        dpi: Resolution for saved figure
        **kwargs: Additional parameters used in interpolation (for title/display)
    
    Returns:
        None (saves plot to file)
    """
    plt.figure(figsize=figsize)
    
    # Generate title if not provided
    if title is None:
        param_str = _format_parameters(method, **kwargs)
        title = f'{method.capitalize()} Interpolation (start={start}, end={end}, num_points={num_points}{param_str})'
    
    plt.plot(values.numpy(), label=method.capitalize(), linewidth=2, marker='o', markersize=4)
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_multiple_methods(
    results: Dict[str, torch.Tensor],
    start: float,
    end: float,
    num_points: int,
    output_path: str,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 150
) -> None:
    """
    Generate and save a comparison plot for multiple interpolation methods.
    
    Args:
        results: Dictionary mapping method names to their interpolated values
        start: Starting value for interpolation
        end: Ending value for interpolation
        num_points: Total number of points
        output_path: Path where the plot will be saved
        title: Optional custom title (if None, auto-generated)
        figsize: Figure size tuple (width, height)
        dpi: Resolution for saved figure
    
    Returns:
        None (saves plot to file)
    """
    plt.figure(figsize=figsize)
    
    if title is None:
        title = f'Comparison of Interpolation Methods (start={start}, end={end}, num_points={num_points})'
    
    for method, values in results.items():
        plt.plot(values.numpy(), label=method.capitalize(), linewidth=2)
    
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def _format_parameters(method: str, **kwargs) -> str:
    """
    Format method-specific parameters as a string for display.
    
    Args:
        method: Interpolation method name
        **kwargs: Method-specific parameters
    
    Returns:
        Formatted parameter string
    """
    param_parts = []
    
    if method == "power" and "p" in kwargs:
        param_parts.append(f", p={kwargs['p']}")
    elif method == "exponential" and "b" in kwargs:
        param_parts.append(f", b={kwargs['b']}")
    elif method == "rho":
        if "rho" in kwargs:
            param_parts.append(f", rho={kwargs['rho']}")
        if kwargs.get("include_zero", False):
            param_parts.append(", include_zero=True")
    
    return "".join(param_parts)

