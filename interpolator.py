"""
A flexible interpolation library for generating intermediate points between start and end values.

This module provides multiple interpolation methods including linear, power, exponential, and rho-based schemes.
"""

import torch
import numpy as np
from typing import Union, Literal


class Interpolator:
    """
    A flexible interpolator class for generating intermediate points between start and end values.
    
    This class supports multiple interpolation schemes suitable for various applications
    including diffusion models, scheduling, and parameter tuning.
    
    Attributes:
        start (float): Starting value for interpolation
        end (float): Ending value for interpolation
        num_steps (int): Total number of output points (output length)
        dtype (torch.dtype): Data type for output tensors
    
    Methods:
        linear(): Linear interpolation from start to end
        power(p: float = 3): Power-based interpolation with adjustable exponent
        exponential(b: float = None): Exponential interpolation with adjustable rate
        rho(rho: float = 7, include_zero: bool = False): Rho-based interpolation
        get_all_methods(): Get results from all interpolation methods
    """
    
    def __init__(
        self,
        start: float,
        end: float,
        num_steps: int,
        dtype: torch.dtype = torch.float64
    ):
        """
        Initialize the Interpolator.
        
        Args:
            start: Starting value for interpolation
            end: Ending value for interpolation
            num_steps: Total number of output points (output length, must be >= 1)
            dtype: Data type for output tensors (default: torch.float64)
        """
        self.start = float(start)
        self.end = float(end)
        self.num_steps = int(num_steps)
        self.dtype = dtype
        
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least 1")
    
    def linear(self) -> torch.Tensor:
        """
        Generate linearly interpolated points between start and end.
        
        Returns:
            torch.Tensor: Tensor of shape (num_steps,) containing interpolated values
                         First value is start, last value is end
        """
        i = torch.arange(self.num_steps, dtype=self.dtype)
        if self.num_steps > 1:
            normalized = i / (self.num_steps - 1)
        else:
            normalized = i
        t_steps = self.start + (self.end - self.start) * normalized
        return t_steps
    
    def power(self, p: float = 3) -> torch.Tensor:
        """
        Generate power-based interpolated points.
        
        This method creates a non-linear interpolation where the distribution
        concentrates more points near the ends or the middle depending on p.
        
        Args:
            p: Power parameter controlling the curve shape (default: 3)
                - Higher p: More concentration at ends
                - Lower p: More uniform distribution
        
        Returns:
            torch.Tensor: Tensor of shape (num_steps,) containing interpolated values
        """
        i = torch.arange(self.num_steps, dtype=self.dtype)
        
        # Power interpolation formula
        # Normalize indices to [0, 1] range
        if self.num_steps > 1:
            normalized = i / (self.num_steps - 1)
        else:
            normalized = i
        
        # Power interpolation formula
        # The formula concentrates points at the ends
        # Create a symmetric curve: low values at middle, high at ends
        power_term = (1 - torch.abs(normalized - 1))**p
        # Normalize power_term to [0, 1] range and map from start to end
        # power_term is 0 at middle, 1 at ends, so we invert to go start->end
        if self.num_steps > 1:
            # Normalize: when normalized=0 or 1, power_term=1; when normalized=0.5, power_term=0.5^p
            # We want to map this to go from start to end
            t_steps = self.start + (self.end - self.start) * power_term
        else:
            t_steps = torch.tensor([self.start], dtype=self.dtype)
        
        return t_steps
    
    def exponential(self, b: float = None) -> torch.Tensor:
        """
        Generate exponentially interpolated points.
        
        This method creates an exponential decay from start to end,
        useful when you want more steps at the beginning.
        
        Args:
            b: Exponential rate parameter (default: (num_steps - 1) * 0.16)
                - Higher b: Slower decay (more gradual)
                - Lower b: Faster decay (more concentration at start)
        
        Returns:
            torch.Tensor: Tensor of shape (num_steps,) containing interpolated values
        """
        if b is None:
            b = max(1, (self.num_steps - 1)) * 0.16
        
        i = torch.arange(self.num_steps, dtype=self.dtype)
        max_index = torch.tensor(max(1, self.num_steps - 1), dtype=self.dtype)
        
        # Exponential interpolation formula
        # exp_term goes from 1 (at i=0) to 0 (at i=max_index)
        # We invert it to go from start to end with more concentration at start
        exp_term = (torch.exp((max_index - i) / b) - 1) / (torch.exp(max_index / b) - 1)
        t_steps = self.start + (self.end - self.start) * (1 - exp_term)
        
        return t_steps
    
    def rho(self, rho: float = 7, include_zero: bool = False) -> torch.Tensor:
        """
        Generate rho-based interpolated points.
        
        This method uses a power transformation with parameter rho,
        commonly used in diffusion models and noise scheduling.
        
        Args:
            rho: Rho parameter controlling the curve shape (default: 7)
                - Higher rho: Different curvature
            include_zero: If True, replaces the last point with zero (default: False)
        
        Returns:
            torch.Tensor: Tensor of shape (num_steps,) containing interpolated values
        """
        step_indices = torch.arange(self.num_steps, dtype=self.dtype)
        
        # Rho-based interpolation formula
        start_power = self.start ** (1 / rho)
        end_power = self.end ** (1 / rho)
        
        if self.num_steps > 1:
            normalized = step_indices / (self.num_steps - 1)
        else:
            normalized = step_indices
        
        t_steps = (start_power + normalized * (end_power - start_power)) ** rho
        
        if include_zero:
            # Replace the last point with zero
            t_steps[-1] = 0.0
        
        return t_steps
    
    def interpolate(
        self,
        method: Literal["linear", "power", "exponential", "rho"] = "linear",
        **kwargs
    ) -> torch.Tensor:
        """
        Unified interface for all interpolation methods.
        
        Args:
            method: Interpolation method to use
                - "linear": Linear interpolation
                - "power": Power-based interpolation (requires 'p' parameter)
                - "exponential": Exponential interpolation (optional 'b' parameter)
                - "rho": Rho-based interpolation (optional 'rho' and 'include_zero' parameters)
            **kwargs: Additional parameters for specific methods
                - For power: p (default: 3)
                - For exponential: b (default: (num_steps - 1) * 0.16)
                - For rho: rho (default: 7), include_zero (default: False)
        
        Returns:
            torch.Tensor: Tensor containing interpolated values
        
        Example:
            >>> interp = Interpolator(0.002, 80, 180)
            >>> linear_values = interp.interpolate("linear")
            >>> power_values = interp.interpolate("power", p=3)
        """
        method = method.lower()
        
        if method == "linear":
            return self.linear()
        elif method == "power":
            p = kwargs.get("p", 3)
            return self.power(p=p)
        elif method == "exponential":
            b = kwargs.get("b", None)
            return self.exponential(b=b)
        elif method == "rho":
            rho = kwargs.get("rho", 7)
            include_zero = kwargs.get("include_zero", False)
            return self.rho(rho=rho, include_zero=include_zero)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available methods: linear, power, exponential, rho"
            )
    
    def get_all_methods(self, **kwargs) -> dict:
        """
        Generate interpolated points using all available methods.
        
        Args:
            **kwargs: Parameters for specific methods
                - p: For power method (default: 3)
                - b: For exponential method (default: (num_steps - 1) * 0.16)
                - rho: For rho method (default: 7)
                - include_zero: For rho method (default: False)
        
        Returns:
            dict: Dictionary containing results for each method
                Keys: 'linear', 'power', 'exponential', 'rho'
        """
        results = {
            "linear": self.linear(),
            "power": self.power(p=kwargs.get("p", 3)),
            "exponential": self.exponential(b=kwargs.get("b", None)),
            "rho": self.rho(
                rho=kwargs.get("rho", 7),
                include_zero=kwargs.get("include_zero", False)
            )
        }
        return results


def interpolate(
    start: float,
    end: float,
    num_steps: int,
    method: Literal["linear", "power", "exponential", "rho"] = "linear",
    dtype: torch.dtype = torch.float64,
    **kwargs
) -> torch.Tensor:
    """
    Convenience function for quick interpolation.
    
    Args:
        start: Starting value
        end: Ending value
        num_steps: Total number of output points (output length)
        method: Interpolation method (default: "linear")
        dtype: Output tensor dtype (default: torch.float64)
        **kwargs: Method-specific parameters
    
    Returns:
        torch.Tensor: Interpolated values
    
    Example:
        >>> values = interpolate(0.0, 1.0, 10, method="linear")
        >>> values = interpolate(0.002, 80, 180, method="power", p=3)
    """
    interp = Interpolator(start, end, num_steps, dtype=dtype)
    return interp.interpolate(method=method, **kwargs)

