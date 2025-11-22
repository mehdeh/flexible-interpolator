"""
Example usage of the Interpolator class.

This script demonstrates various ways to use the Interpolator for different
interpolation methods and scenarios.
"""

import torch
import matplotlib.pyplot as plt
from interpolator import Interpolator, interpolate


def example_1_basic_usage():
    """Example 1: Basic usage with different methods"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create an interpolator instance
    interp = Interpolator(start=0.0, end=1.0, num_steps=10)
    
    # Get linear interpolation
    linear_values = interp.linear()
    print(f"\nLinear interpolation (first 5 and last 5 values):")
    print(f"First 5: {linear_values[:5]}")
    print(f"Last 5: {linear_values[-5:]}")
    
    # Get power interpolation
    power_values = interp.power(p=3)
    print(f"\nPower interpolation (p=3, first 5 and last 5 values):")
    print(f"First 5: {power_values[:5]}")
    print(f"Last 5: {power_values[-5:]}")
    
    # Get exponential interpolation
    exp_values = interp.exponential(b=2.0)
    print(f"\nExponential interpolation (b=2.0, first 5 and last 5 values):")
    print(f"First 5: {exp_values[:5]}")
    print(f"Last 5: {exp_values[-5:]}")


def example_2_unified_interface():
    """Example 2: Using the unified interpolate() method"""
    print("\n" + "=" * 60)
    print("Example 2: Unified Interface")
    print("=" * 60)
    
    interp = Interpolator(start=0.002, end=80.0, num_steps=20)
    
    # Use unified interface
    methods = ["linear", "power", "exponential", "rho"]
    for method in methods:
        values = interp.interpolate(method=method, p=3, rho=7)
        print(f"\n{method.capitalize()} interpolation:")
        print(f"  Shape: {values.shape}")
        print(f"  First value: {values[0]:.4f}")
        print(f"  Last value: {values[-1]:.4f}")


def example_3_convenience_function():
    """Example 3: Using the convenience function"""
    print("\n" + "=" * 60)
    print("Example 3: Convenience Function")
    print("=" * 60)
    
    # Quick interpolation without creating a class instance
    values = interpolate(start=0, end=100, num_steps=5, method="linear")
    print(f"\nLinear interpolation from 0 to 100 in 5 steps:")
    print(values)


def example_4_all_methods():
    """Example 4: Get all methods at once"""
    print("\n" + "=" * 60)
    print("Example 4: Get All Methods")
    print("=" * 60)
    
    interp = Interpolator(start=0.002, end=80.0, num_steps=50)
    all_results = interp.get_all_methods(p=3, rho=7)
    
    print("\nAll interpolation methods:")
    for method, values in all_results.items():
        print(f"  {method}: shape={values.shape}, "
              f"range=[{values.min():.4f}, {values.max():.4f}]")


def example_5_visualization():
    """Example 5: Visual comparison of different methods"""
    print("\n" + "=" * 60)
    print("Example 5: Visual Comparison")
    print("=" * 60)
    print("Generating plot...")
    
    # Original parameters from the notebook
    interp = Interpolator(
        start=0.002,
        end=80.0,
        num_steps=180,
        dtype=torch.float64
    )
    
    # Get all interpolation methods
    t_steps_linear = interp.linear()
    t_steps_power = interp.power(p=3)
    t_steps_exp = interp.exponential(b=180 * 0.16)
    t_steps_rho = interp.rho(rho=7, include_zero=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 7))
    
    plt.subplot(2, 1, 1)
    plt.plot(t_steps_linear.numpy(), label='Linear', linewidth=2)
    plt.plot(t_steps_power.numpy(), label='Power (p=3)', linewidth=2)
    plt.plot(t_steps_exp.numpy(), label='Exponential (b=28.8)', linewidth=2)
    plt.plot(t_steps_rho.numpy(), label='Rho (ρ=7)', linewidth=2)
    plt.xlabel('Step Index')
    plt.ylabel('Value')
    plt.title('Comparison of Interpolation Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale plot for better visualization
    plt.subplot(2, 1, 2)
    plt.semilogy(t_steps_linear.numpy(), label='Linear', linewidth=2)
    plt.semilogy(t_steps_power.numpy(), label='Power (p=3)', linewidth=2)
    plt.semilogy(t_steps_exp.numpy(), label='Exponential (b=28.8)', linewidth=2)
    plt.semilogy(t_steps_rho.numpy(), label='Rho (ρ=7)', linewidth=2)
    plt.xlabel('Step Index')
    plt.ylabel('Value (Log Scale)')
    plt.title('Comparison of Interpolation Methods (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'interpolation_comparison.png'")
    plt.show()


def example_6_parameter_effects():
    """Example 6: Effect of different parameters"""
    print("\n" + "=" * 60)
    print("Example 6: Parameter Effects")
    print("=" * 60)
    
    interp = Interpolator(start=0.0, end=1.0, num_steps=50)
    
    # Test different power values
    print("\nPower interpolation with different p values:")
    for p in [1, 2, 3, 5]:
        values = interp.power(p=p)
        print(f"  p={p}: mid-point value = {values[len(values)//2]:.4f}")
    
    # Test different exponential rates
    print("\nExponential interpolation with different b values:")
    for b in [5, 10, 20, 50]:
        values = interp.exponential(b=b)
        print(f"  b={b}: mid-point value = {values[len(values)//2]:.4f}")
    
    # Test different rho values
    print("\nRho interpolation with different rho values:")
    for rho in [2, 5, 7, 10]:
        values = interp.rho(rho=rho)
        print(f"  rho={rho}: mid-point value = {values[len(values)//2]:.4f}")


def example_7_diffusion_model_scenario():
    """Example 7: Real-world scenario - Diffusion model noise scheduling"""
    print("\n" + "=" * 60)
    print("Example 7: Diffusion Model Noise Scheduling")
    print("=" * 60)
    
    # Typical values for diffusion models
    sigma_min = 0.002
    sigma_max = 80.0
    num_steps = 180
    
    interp = Interpolator(
        start=sigma_min,
        end=sigma_max,
        num_steps=num_steps
    )
    
    # Get different scheduling schemes
    schedules = {
        'Linear': interp.linear(),
        'Power': interp.power(p=3),
        'Exponential': interp.exponential(b=num_steps * 0.16),
        'Rho': interp.rho(rho=7, include_zero=False)
    }
    
    print("\nNoise schedules for diffusion model:")
    for name, schedule in schedules.items():
        print(f"\n{name} schedule:")
        print(f"  Steps: {len(schedule)}")
        print(f"  Min: {schedule.min():.4f}")
        print(f"  Max: {schedule.max():.4f}")
        print(f"  First step size: {(schedule[1] - schedule[0]):.4f}")
        print(f"  Last step size: {(schedule[-1] - schedule[-2]):.4f}")


if __name__ == "__main__":
    # Run all examples
    print("\n" + "=" * 60)
    print("Interpolator Library - Examples")
    print("=" * 60)
    
    try:
        example_1_basic_usage()
        example_2_unified_interface()
        example_3_convenience_function()
        example_4_all_methods()
        example_6_parameter_effects()
        example_7_diffusion_model_scenario()
        
        # Visualization example (requires matplotlib)
        try:
            example_5_visualization()
        except Exception as e:
            print(f"\nNote: Visualization skipped ({e})")
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

