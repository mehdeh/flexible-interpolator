"""
A flexible interpolation library for generating intermediate points between start and end values.

This module provides multiple interpolation methods including linear, power, exponential, rho-based, and geometric schemes.
"""

import torch
from typing import Literal, Optional


class Interpolator:
    """
    A flexible interpolator class for generating intermediate points between start and end values.
    
    This class supports multiple interpolation schemes suitable for various applications
    including diffusion models, scheduling, and parameter tuning.
    
    Attributes:
        start (float): Starting value for interpolation
        end (float): Ending value for interpolation
        num_points (int): Total number of output points (including start, end, and all intermediate points)
        dtype (torch.dtype): Data type for output tensors
    
    Methods:
        linear(): Linear interpolation from start to end
        power(p: float = 3): Power-based interpolation with adjustable exponent
        exponential(b: Optional[float] = None): Exponential interpolation with adjustable rate
        rho(rho: float = 7, include_zero: bool = False): Rho-based interpolation
        geometric(): Geometric interpolation using exponential scaling
        get_all_methods(): Get results from all interpolation methods
    """
    
    def __init__(
        self,
        start: float,
        end: float,
        num_points: int,
        dtype: torch.dtype = torch.float64
    ):
        """
        Initialize the Interpolator.
        
        Args:
            start: Starting value for interpolation
            end: Ending value for interpolation
            num_points: Total number of output points including start, end, and intermediate points (must be >= 1)
            dtype: Data type for output tensors (default: torch.float64)
        
        Raises:
            ValueError: If num_points is less than 1
        """
        self.start = float(start)
        self.end = float(end)
        self.num_points = int(num_points)
        self.dtype = dtype
        
        if self.num_points < 1:
            raise ValueError("num_points must be at least 1")
    
    def _normalize_indices(self) -> torch.Tensor:
        """
        Normalize step indices to [0, 1] range.
        
        Returns:
            torch.Tensor: Normalized indices in range [0, 1]
        """
        indices = torch.arange(self.num_points, dtype=self.dtype)
        if self.num_points > 1:
            return indices / (self.num_points - 1)
        return indices
    
    def _map_to_range(self, normalized_values: torch.Tensor) -> torch.Tensor:
        """
        Map normalized values from [0, 1] to [start, end] range.
        
        Args:
            normalized_values: Values in range [0, 1]
        
        Returns:
            torch.Tensor: Values mapped to [start, end] range
        """
        return self.start + (self.end - self.start) * normalized_values
    
    def linear(self) -> torch.Tensor:
        """
        Generate linearly interpolated points between start and end.
        
        Returns:
            torch.Tensor: Tensor of shape (num_points,) containing interpolated values
                         First value is start, last value is end
        """
        normalized = self._normalize_indices()
        return self._map_to_range(normalized)
    
    def power(self, p: float = 3) -> torch.Tensor:
        """
        Generate power-based interpolated points.
        
        This method creates a non-linear interpolation where the distribution
        depends on the power parameter p.
        
        Args:
            p: Power parameter controlling the curve shape (default: 3)
                - Higher p (p > 1): More concentration at start
                - Lower p (0 < p < 1): More concentration at end
                - p = 1: Linear interpolation
                Must be positive
        
        Returns:
            torch.Tensor: Tensor of shape (num_points,) containing interpolated values
                         First value is start, last value is end
        
        Raises:
            ValueError: If p is not positive
        """
        if p <= 0:
            raise ValueError("Power parameter 'p' must be positive")
        
        normalized = self._normalize_indices()
        
        # Power interpolation formula: (i/(n-1))^p
        # When p > 1: concentrates more points near the start
        # When p < 1: concentrates more points near the end
        # When p = 1: linear interpolation
        power_term = normalized ** p
        
        return self._map_to_range(power_term)
    
    def exponential(self, b: Optional[float] = None) -> torch.Tensor:
        """
        Generate exponentially interpolated points.
        
        This method creates an exponential decay from start to end,
        useful when you want more steps at the beginning.
        
        Args:
            b: Exponential rate parameter (default: (num_points - 1) * 0.16)
                - Higher b: Slower decay (more gradual)
                - Lower b: Faster decay (more concentration at start)
                Must be non-zero (can be positive or negative)
        
        Returns:
            torch.Tensor: Tensor of shape (num_points,) containing interpolated values
                         First value is start, last value is end
        
        Raises:
            ValueError: If b is zero
        """
        if b is None:
            b = max(1.0, float(self.num_points - 1)) * 0.16
        else:
            b = float(b)
        
        if b == 0:
            raise ValueError("Exponential rate parameter 'b' must be non-zero")
        
        indices = torch.arange(self.num_points, dtype=self.dtype)
        max_index = torch.tensor(
            max(1.0, float(self.num_points - 1)),
            dtype=self.dtype
        )
        
        # Exponential interpolation formula
        # exp_term goes from 1 (at i=0) to 0 (at i=max_index)
        # We invert it to go from start to end with more concentration at start
        exp_term = (
            (torch.exp((max_index - indices) / b) - 1) /
            (torch.exp(max_index / b) - 1)
        )
        
        normalized = 1 - exp_term
        return self._map_to_range(normalized)
    
    def rho(self, rho: float = 7, include_zero: bool = False) -> torch.Tensor:
        """
        Generate rho-based interpolated points.
        
        This method uses a power transformation with parameter rho,
        commonly used in diffusion models and noise scheduling.
        
        Args:
            rho: Rho parameter controlling the curve shape (default: 7)
                - Higher rho: Different curvature
                Must be non-zero (can be positive or negative)
            include_zero: If True, replaces the last point with zero (default: False)
        
        Returns:
            torch.Tensor: Tensor of shape (num_points,) containing interpolated values
                         First value is start, last value is end (or zero if include_zero=True)
        
        Raises:
            ValueError: If rho is not positive
            ValueError: If include_zero=True but start or end is negative
        """
        if rho <= 0:
            raise ValueError("Rho parameter must be positive and non-zero")
        
        if include_zero and (self.start < 0 or self.end < 0):
            raise ValueError(
                "include_zero=True requires both start and end to be non-negative"
            )
        
        normalized = self._normalize_indices()
        
        # Rho-based interpolation formula
        start_power = self.start ** (1.0 / rho)
        end_power = self.end ** (1.0 / rho)
        
        interpolated_power = start_power + normalized * (end_power - start_power)
        t_steps = interpolated_power ** rho
        
        if include_zero:
            t_steps[-1] = 0.0
        
        return t_steps
    
    def geometric(self) -> torch.Tensor:
        """
        Generate geometrically interpolated points.
        
        This method creates a geometric progression from start to end,
        where values scale exponentially based on the ratio end/start.
        
        Returns:
            torch.Tensor: Tensor of shape (num_points,) containing interpolated values
                         First value is start, last value is end
        
        Raises:
            ValueError: If start is zero (division by zero)
            ValueError: If start and end have opposite signs (would require complex numbers)
        """
        if self.num_points == 1:
            return torch.tensor([self.start], dtype=self.dtype)
        
        if self.start == 0:
            raise ValueError(
                "Geometric interpolation requires start != 0 (division by zero)"
            )
        
        if (self.start > 0 and self.end < 0) or (self.start < 0 and self.end > 0):
            raise ValueError(
                "Geometric interpolation requires start and end to have the same sign "
                "(both positive or both negative)"
            )
        
        normalized = self._normalize_indices()
        
        # Geometric interpolation formula: t_i = start * (end/start) ^ (i/(N-1))
        ratio = self.end / self.start
        t_steps = self.start * (ratio ** normalized)
        
        return t_steps
    
    def interpolate(
        self,
        method: Literal["linear", "power", "exponential", "rho", "geometric"] = "linear",
        **kwargs
    ) -> torch.Tensor:
        """
        Unified interface for all interpolation methods.
        
        Args:
            method: Interpolation method to use
                - "linear": Linear interpolation
                - "power": Power-based interpolation (optional 'p' parameter)
                - "exponential": Exponential interpolation (optional 'b' parameter)
                - "rho": Rho-based interpolation (optional 'rho' and 'include_zero' parameters)
                - "geometric": Geometric interpolation using exponential scaling
            **kwargs: Additional parameters for specific methods
                - For power: p (default: 3)
                - For exponential: b (default: (num_points - 1) * 0.16)
                - For rho: rho (default: 7), include_zero (default: False)
                - Geometric method has no additional parameters
        
        Returns:
            torch.Tensor: Tensor containing interpolated values
        
        Raises:
            ValueError: If method is not recognized
        
        Example:
            >>> interp = Interpolator(0.002, 80, 180)
            >>> linear_values = interp.interpolate("linear")
            >>> power_values = interp.interpolate("power", p=3)
            >>> geometric_values = interp.interpolate("geometric")
        """
        method = method.lower()
        
        method_map = {
            "linear": lambda: self.linear(),
            "power": lambda: self.power(p=kwargs.get("p", 3)),
            "exponential": lambda: self.exponential(b=kwargs.get("b", None)),
            "rho": lambda: self.rho(
                rho=kwargs.get("rho", 7),
                include_zero=kwargs.get("include_zero", False)
            ),
            "geometric": lambda: self.geometric(),
        }
        
        if method not in method_map:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available methods: {', '.join(method_map.keys())}"
            )
        
        return method_map[method]()
    
    def get_all_methods(self, **kwargs) -> dict:
        """
        Generate interpolated points using all available methods.
        
        Args:
            **kwargs: Parameters for specific methods
                - p: For power method (default: 3)
                - b: For exponential method (default: (num_points - 1) * 0.16)
                - rho: For rho method (default: 7)
                - include_zero: For rho method (default: False)
        
        Returns:
            dict: Dictionary containing results for each method
                Keys: 'linear', 'power', 'exponential', 'rho', 'geometric'
        """
        return {
            "linear": self.linear(),
            "power": self.power(p=kwargs.get("p", 3)),
            "exponential": self.exponential(b=kwargs.get("b", None)),
            "rho": self.rho(
                rho=kwargs.get("rho", 7),
                include_zero=kwargs.get("include_zero", False)
            ),
            "geometric": self.geometric(),
        }


def interpolate(
    start: float,
    end: float,
    num_points: int,
    method: Literal["linear", "power", "exponential", "rho", "geometric"] = "linear",
    dtype: torch.dtype = torch.float64,
    **kwargs
) -> torch.Tensor:
    """
    Convenience function for quick interpolation.
    
    Args:
        start: Starting value
        end: Ending value
        num_points: Total number of output points (including start, end, and all intermediate points)
        method: Interpolation method (default: "linear")
        dtype: Output tensor dtype (default: torch.float64)
        **kwargs: Method-specific parameters
    
    Returns:
        torch.Tensor: Interpolated values
    
    Example:
        >>> values = interpolate(0.0, 1.0, 10, method="linear")
        >>> values = interpolate(0.002, 80, 180, method="power", p=3)
    """
    interp = Interpolator(start, end, num_points, dtype=dtype)
    return interp.interpolate(method=method, **kwargs)
