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
    interp = Interpolator(start=0.0, end=100.0, num_points=50, dtype=torch.float64)
    
    # Get linear interpolation
    linear_values = interp.linear()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(linear_values.numpy(), label='Linear', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Linear Interpolation (start=0.0, end=100.0, num_points=50)', fontsize=14)
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
    interp = Interpolator(start=0.0, end=100.0, num_points=50, dtype=torch.float64)
    
    # Test different power values (only positive)
    p_values = [1, 2, 3, 5, 7, 10]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    for p in p_values:
        power_values = interp.power(p=p)
        plt.plot(power_values.numpy(), label=f'Power (p={p})', linewidth=2)
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Power Interpolation with Different Parameters (start=0.0, end=100.0, num_points=50)', fontsize=14)
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
    interp = Interpolator(start=0.0, end=100.0, num_points=50, dtype=torch.float64)
    
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
    plt.title('Exponential Interpolation with Different Parameters (start=0.0, end=100.0, num_points=50)', fontsize=14)
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
        start=20.0,
        end=80.0,
        num_points=50,
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
    plt.title('Comparison of All Interpolation Methods (start=20.0, end=80.0, num_points=50)', fontsize=14)
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
        end=20.0,
        num_points=50,
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
    plt.title('Comparison of All Interpolation Methods (start=80.0, end=20.0, num_points=50)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('all_methods_comparison_descending.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'all_methods_comparison_descending.png'")
    plt.close()


def example_hyperparameter_settings():
    """Example 6: Interpolation with hyperparameter settings from notebook"""
    print("\n" + "=" * 60)
    print("Example 6: Hyperparameter Settings Comparison")
    print("=" * 60)
    print("Generating plot...")
    
    # Parameters from hyperparameter_settings.ipynb
    num_steps = 180
    sigma_min = 0.002
    sigma_max = 80
    rho = 7
    p = 3
    b = num_steps * 0.16  # 28.8
    
    # Create interpolator instances
    interp_180 = Interpolator(
        start=sigma_min,
        end=sigma_max,
        num_points=num_steps,
        dtype=torch.float64
    )
    
    interp_181 = Interpolator(
        start=sigma_min,
        end=sigma_max,
        num_points=num_steps + 1,
        dtype=torch.float64
    )
    
    # Get interpolation results for each method
    # Power and rho: 180 points
    power_values = interp_180.power(p=p)
    rho_values = interp_180.rho(rho=rho, include_zero=False)
    
    # Add zero at the end for rho (matching notebook's t_steps_new)
    rho_values_with_zero = torch.cat([rho_values, torch.zeros_like(rho_values[:1])])
    
    # Exponential: 181 points (matching notebook's t_steps_exp)
    exp_values = interp_181.exponential(b=b)
    
    # Linear: 180 points for comparison
    linear_values = interp_180.linear()
    
    # Plot comparison
    plt.figure(figsize=(12, 7))
    
    plt.plot(linear_values.numpy(), label='Linear (180 points)', linewidth=2)
    plt.plot(power_values.numpy(), label=f'Power p={p} (180 points)', linewidth=2)
    plt.plot(exp_values.numpy(), label=f'Exponential b={b:.1f} (181 points)', linewidth=2)
    plt.plot(rho_values_with_zero.numpy(), label=f'Rho rho={rho} (181 points, with zero)', linewidth=2)
    
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(
        f'Hyperparameter Settings Comparison\n'
        f'(sigma_min={sigma_min}, sigma_max={sigma_max}, num_steps={num_steps})',
        fontsize=14
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization with large range
    
    plt.tight_layout()
    plt.savefig('hyperparameter_settings_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'hyperparameter_settings_comparison.png'")
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
        example_hyperparameter_settings()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - linear_interpolation.png")
        print("  - power_interpolation_multiple_params.png")
        print("  - exponential_interpolation_multiple_params.png")
        print("  - all_methods_comparison_ascending.png")
        print("  - all_methods_comparison_descending.png")
        print("  - hyperparameter_settings_comparison.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
