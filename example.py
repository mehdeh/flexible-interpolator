"""
Example usage of the Interpolator class.

This script demonstrates various ways to use the Interpolator for different
interpolation methods and scenarios.
"""

import torch
import matplotlib.pyplot as plt
from interpolator import Interpolator


def example_linear():
    """Example 1: Linear interpolation"""
    print("=" * 60)
    print("Example 1: Linear Interpolation")
    print("=" * 60)
    print("Generating plot...")
    
    # Create an interpolator instance
    interp = Interpolator(start=0.0, end=100.0, num_steps=50, dtype=torch.float64)
    
    # Get linear interpolation
    linear_values = interp.linear()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(linear_values.numpy(), label='Linear', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Linear Interpolation (start=0.0, end=100.0, num_steps=50)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('linear_interpolation.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'linear_interpolation.png'")
    plt.close()


def example_power_multiple_params():
    """Example 2: Power interpolation with multiple parameter values"""
    print("\n" + "=" * 60)
    print("Example 2: Power Interpolation with Multiple Parameters")
    print("=" * 60)
    print("Generating plot...")
    
    # Create an interpolator instance with fixed start and end
    interp = Interpolator(start=0.0, end=100.0, num_steps=50, dtype=torch.float64)
    
    # Test different power values (only positive)
    p_values = [1, 2, 3, 5, 7, 10]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    for p in p_values:
        power_values = interp.power(p=p)
        plt.plot(power_values.numpy(), label=f'Power (p={p})', linewidth=2)
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Power Interpolation with Different Parameters (start=0.0, end=100.0, num_steps=50)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('power_interpolation_multiple_params.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'power_interpolation_multiple_params.png'")
    plt.close()


def example_exponential_multiple_params():
    """Example 3: Exponential interpolation with multiple parameter values"""
    print("\n" + "=" * 60)
    print("Example 3: Exponential Interpolation with Multiple Parameters")
    print("=" * 60)
    print("Generating plot...")
    
    # Create an interpolator instance with fixed start and end
    interp = Interpolator(start=0.0, end=100.0, num_steps=50, dtype=torch.float64)
    
    # Test different exponential rate values (including negative)
    b_values = [-30, -20, -10, -5, 5, 10, 20, 50]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    for b in b_values:
        try:
            exp_values = interp.exponential(b=b)
            plt.plot(exp_values.numpy(), label=f'Exponential (b={b})', linewidth=2)
        except Exception as e:
            print(f"  Note: Skipped b={b} due to error: {e}")
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Exponential Interpolation with Different Parameters (start=0.0, end=100.0, num_steps=50)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exponential_interpolation_multiple_params.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'exponential_interpolation_multiple_params.png'")
    plt.close()


def example_all_methods_ascending():
    """Example 4: Comparison of all three methods (start < end)"""
    print("\n" + "=" * 60)
    print("Example 4: All Methods Comparison (Ascending: start < end)")
    print("=" * 60)
    print("Generating plot...")
    
    # Create an interpolator instance (start < end)
    interp = Interpolator(
        start=0.002,
        end=80.0,
        num_steps=50,
        dtype=torch.float64
    )
    
    # Get interpolation results for each method (one setting each)
    linear_values = interp.linear()
    power_values = interp.power(p=3)
    exp_values = interp.exponential(b=15)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    plt.plot(linear_values.numpy(), label='Linear', linewidth=2)
    plt.plot(power_values.numpy(), label='Power (p=3)', linewidth=2)
    plt.plot(exp_values.numpy(), label='Exponential (b=15)', linewidth=2)
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Comparison of All Interpolation Methods (start=0.002, end=80.0, num_steps=50)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('all_methods_comparison_ascending.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'all_methods_comparison_ascending.png'")
    plt.close()


def example_all_methods_descending():
    """Example 5: Comparison of all three methods (start > end)"""
    print("\n" + "=" * 60)
    print("Example 5: All Methods Comparison (Descending: start > end)")
    print("=" * 60)
    print("Generating plot...")
    
    # Create an interpolator instance (start > end)
    interp = Interpolator(
        start=80.0,
        end=0.002,
        num_steps=50,
        dtype=torch.float64
    )
    
    # Get interpolation results for each method (one setting each)
    linear_values = interp.linear()
    power_values = interp.power(p=3)
    exp_values = interp.exponential(b=15)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    plt.plot(linear_values.numpy(), label='Linear', linewidth=2)
    plt.plot(power_values.numpy(), label='Power (p=3)', linewidth=2)
    plt.plot(exp_values.numpy(), label='Exponential (b=15)', linewidth=2)
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Comparison of All Interpolation Methods (start=80.0, end=0.002, num_steps=50)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('all_methods_comparison_descending.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'all_methods_comparison_descending.png'")
    plt.close()


if __name__ == "__main__":
    # Run all examples
    print("\n" + "=" * 60)
    print("Interpolator Library - Examples")
    print("=" * 60)
    
    try:
        example_linear()
        example_power_multiple_params()
        example_exponential_multiple_params()
        example_all_methods_ascending()
        example_all_methods_descending()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - linear_interpolation.png")
        print("  - power_interpolation_multiple_params.png")
        print("  - exponential_interpolation_multiple_params.png")
        print("  - all_methods_comparison_ascending.png")
        print("  - all_methods_comparison_descending.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
