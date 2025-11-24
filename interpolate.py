"""
Command-line interface for the Interpolator library.

This module provides a CLI tool for generating interpolation plots with various methods
and parameters.
"""

import argparse
import os
import sys
from typing import Optional, Dict, Any

from interpolator import Interpolator
from plotting import plot_interpolation


def generate_filename(
    method: str,
    start: float,
    end: float,
    num_points: int,
    **kwargs
) -> str:
    """
    Generate an appropriate filename based on method and parameters.
    
    Args:
        method: Interpolation method name
        start: Starting value
        end: Ending value
        num_points: Number of points
        **kwargs: Method-specific parameters
    
    Returns:
        Generated filename string
    """
    def format_number(num: float) -> str:
        """
        Format number for filename.
        
        - Integers (e.g., 0.0, 10.0) are displayed as integers (0, 10)
        - Decimals (e.g., 0.002) have dots replaced with 'p' (0p002)
        - Negative numbers have dashes replaced with 'neg'
        """
        # Check if number is effectively an integer
        if num == int(num):
            num_str = str(int(num))
        else:
            num_str = str(num)
        
        # Replace dots with 'p' and dashes with 'neg'
        return num_str.replace(".", "p").replace("-", "neg")
    
    # Base filename components
    filename_parts = [
        method,
        f"start{format_number(start)}",
        f"end{format_number(end)}",
        f"points{num_points}"
    ]
    
    # Add method-specific parameters
    if method == "power" and "p" in kwargs:
        filename_parts.append(f"p{format_number(kwargs['p'])}")
    elif method == "exponential" and "b" in kwargs:
        filename_parts.append(f"b{format_number(kwargs['b'])}")
    elif method == "rho":
        if "rho" in kwargs:
            filename_parts.append(f"rho{format_number(kwargs['rho'])}")
        if kwargs.get("include_zero", False):
            filename_parts.append("zero")
    
    # Join parts and add extension
    filename = "_".join(filename_parts) + ".png"
    
    return filename


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate interpolation plots with various methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Linear interpolation
  python interpolate.py --method linear --start 0.0 --end 100.0 --num-points 50
  
  # Power interpolation with custom parameter
  python interpolate.py --method power --start 0.0 --end 100.0 --num-points 50 --p 5
  
  # Exponential interpolation
  python interpolate.py --method exponential --start 0.0 --end 100.0 --num-points 50 --b 15
  
  # Rho interpolation
  python interpolate.py --method rho --start 0.002 --end 80.0 --num-points 180 --rho 7
  
  # Geometric interpolation
  python interpolate.py --method geometric --start 0.0 --end 100.0 --num-points 50
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--method",
        type=str,
        choices=["linear", "power", "exponential", "rho", "geometric"],
        required=True,
        help="Interpolation method to use"
    )
    
    parser.add_argument(
        "--start",
        type=float,
        required=True,
        help="Starting value for interpolation"
    )
    
    parser.add_argument(
        "--end",
        type=float,
        required=True,
        help="Ending value for interpolation"
    )
    
    parser.add_argument(
        "--num-points",
        type=int,
        required=True,
        help="Total number of output points (including start and end)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (if not provided, auto-generated based on parameters)"
    )
    
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="Power parameter for power method (default: 3)"
    )
    
    parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="Exponential rate parameter for exponential method (default: auto-calculated)"
    )
    
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Rho parameter for rho method (default: 7)"
    )
    
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include zero in rho method (only valid for rho method)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for output image (default: 150)"
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments for consistency.
    
    Args:
        args: Parsed arguments namespace
    
    Raises:
        ValueError: If arguments are invalid
    """
    if args.num_points < 1:
        raise ValueError("--num-points must be at least 1")
    
    if args.method == "power" and args.p is not None and args.p <= 0:
        raise ValueError("--p must be positive for power method")
    
    if args.method == "exponential" and args.b is not None and args.b == 0:
        raise ValueError("--b must be non-zero for exponential method")
    
    if args.method == "rho":
        if args.rho is not None and args.rho <= 0:
            raise ValueError("--rho must be positive for rho method")
        if args.include_zero and (args.start < 0 or args.end < 0):
            raise ValueError("--include-zero requires both start and end to be non-negative")
    
    if args.method == "geometric":
        if args.start == 0:
            raise ValueError("--start must be non-zero for geometric method")
        if (args.start > 0 and args.end < 0) or (args.start < 0 and args.end > 0):
            raise ValueError("--start and --end must have the same sign for geometric method")
    
    # Warn about unused parameters
    if args.method != "power" and args.p is not None:
        print(f"Warning: --p parameter is ignored for {args.method} method", file=sys.stderr)
    
    if args.method != "exponential" and args.b is not None:
        print(f"Warning: --b parameter is ignored for {args.method} method", file=sys.stderr)
    
    if args.method != "rho":
        if args.rho is not None:
            print(f"Warning: --rho parameter is ignored for {args.method} method", file=sys.stderr)
        if args.include_zero:
            print(f"Warning: --include-zero parameter is ignored for {args.method} method", file=sys.stderr)
    
    if args.method != "geometric" and args.start == 0:
        # This is just a warning, not an error, since other methods can handle start=0
        pass


def build_method_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build keyword arguments dictionary for the selected method.
    
    Args:
        args: Parsed arguments namespace
    
    Returns:
        Dictionary of method-specific parameters
    """
    kwargs = {}
    
    if args.method == "power":
        if args.p is not None:
            kwargs["p"] = args.p
    
    elif args.method == "exponential":
        if args.b is not None:
            kwargs["b"] = args.b
    
    elif args.method == "rho":
        if args.rho is not None:
            kwargs["rho"] = args.rho
        if args.include_zero:
            kwargs["include_zero"] = True
    
    return kwargs


def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        validate_arguments(args)
        
        # Build method-specific kwargs
        method_kwargs = build_method_kwargs(args)
        
        # Create interpolator
        interp = Interpolator(
            start=args.start,
            end=args.end,
            num_points=args.num_points
        )
        
        # Perform interpolation
        values = interp.interpolate(method=args.method, **method_kwargs)
        
        # Create outputs directory if it doesn't exist
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Generate output filename
        if args.output:
            filename = os.path.basename(args.output)
        else:
            filename = generate_filename(
                method=args.method,
                start=args.start,
                end=args.end,
                num_points=args.num_points,
                **method_kwargs
            )
        
        # Construct full output path in outputs directory
        output_path = os.path.join(outputs_dir, filename)
        
        # Generate plot
        plot_interpolation(
            values=values,
            method=args.method,
            start=args.start,
            end=args.end,
            num_points=args.num_points,
            output_path=output_path,
            dpi=args.dpi,
            **method_kwargs
        )
        
        print(f"Successfully generated plot: {output_path}")
        return 0
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

