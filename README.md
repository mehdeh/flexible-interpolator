# Flexible Interpolator

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A flexible Python library for generating intermediate points between start and end values using various interpolation methods. Perfect for noise scheduling in diffusion models, parameter tuning, and any scenario requiring non-linear interpolation schemes.

This code is provided freely for public use without any license restrictions.

## Features

- 🎯 **Multiple Interpolation Methods**: Linear, Power-based, Exponential, and Rho-based
- 🚀 **Easy to Use**: Simple API with both class-based and function-based interfaces
- 🔧 **Flexible**: Adjustable parameters for fine-tuning each method
- 📊 **Visualization Support**: Built-in plotting capabilities
- 🎨 **PyTorch Compatible**: Native PyTorch tensor support
- 📦 **Lightweight**: Minimal dependencies

## Installation

This library can be used directly by importing the `interpolator.py` file. Simply ensure you have the required dependencies installed:

```bash
pip install torch matplotlib numpy
```

Or install all dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

Then you can directly import and use the interpolator in your Python scripts:
```python
from interpolator import Interpolator
```

## Quick Start

### Basic Usage

```python
from interpolator import Interpolator

# Create an interpolator instance
interp = Interpolator(start=0.0, end=1.0, num_steps=10)

# Linear interpolation
linear_values = interp.linear()

# Power-based interpolation
power_values = interp.power(p=3)

# Exponential interpolation
exp_values = interp.exponential(b=2.0)

# Rho-based interpolation
rho_values = interp.rho(rho=7)
```

### Using the Unified Interface

```python
# All methods through one interface
interp = Interpolator(start=0.002, end=80.0, num_steps=180)

linear = interp.interpolate("linear")
power = interp.interpolate("power", p=3)
exponential = interp.interpolate("exponential", b=28.8)
rho = interp.interpolate("rho", rho=7, include_zero=False)
```

### Convenience Function

```python
from interpolator import interpolate

# Quick interpolation without creating a class
values = interpolate(start=0, end=100, num_steps=5, method="linear")
```

### Get All Methods at Once

```python
interp = Interpolator(start=0.002, end=80.0, num_steps=180)
all_results = interp.get_all_methods(p=3, rho=7, include_zero=False)

# Access results
linear = all_results['linear']
power = all_results['power']
exponential = all_results['exponential']
rho = all_results['rho']
```

## Interpolation Methods

### 1. Linear Interpolation

The simplest method that creates equally spaced points between start and end.

```python
values = interp.linear()
```

**Mathematical Formula:**

For output length $n$ (num_steps), the interpolated values are:

$$t_i = \text{start} + (\text{end} - \text{start}) \cdot \frac{i}{n-1}$$

where $i \in \{0, 1, 2, \ldots, n-1\}$ and $n > 1$. When $n = 1$, $t_0 = \text{start}$.

**Characteristics:**
- Uniform distribution
- Equal step sizes
- Straight line from start to end

### 2. Power-based Interpolation

Creates non-linear interpolation with adjustable power parameter `p`.

```python
values = interp.power(p=3)
```

**Mathematical Formula:**

For output length $n$ (num_steps) and power parameter $p$:

$$t_i = \text{start} + (\text{end} - \text{start}) \cdot \left(1 - \left|\frac{i}{n-1} - 1\right|\right)^p$$

where $i \in \{0, 1, 2, \ldots, n-1\}$ and $n > 1$. When $n = 1$, $t_0 = \text{start}$.

This formula creates a symmetric curve that concentrates more points near the start and end boundaries, with fewer points in the middle region.

**Parameters:**
- `p` (float): Power parameter controlling curve shape
  - Higher `p`: More concentration at ends
  - Lower `p`: More uniform distribution

**Use Cases:**
- When you need more samples near start/end
- Non-uniform sampling strategies

### 3. Exponential Interpolation

Exponential decay from start to end, useful when you need more steps at the beginning.

```python
values = interp.exponential(b=28.8)
```

**Mathematical Formula:**

For output length $n$ (num_steps) and exponential rate parameter $b$:

$$t_i = \text{start} + (\text{end} - \text{start}) \cdot \left(1 - \frac{\exp\left(\frac{n-1 - i}{b}\right) - 1}{\exp\left(\frac{n-1}{b}\right) - 1}\right)$$

where $i \in \{0, 1, 2, \ldots, n-1\}$, $n > 1$, and $b > 0$. When $n = 1$, $t_0 = \text{start}$.

The default value for $b$ is $(n-1) \cdot 0.16$.

This formula creates an exponential decay that concentrates more points near the start value, with the spacing increasing as we approach the end.

**Parameters:**
- `b` (float): Exponential rate parameter (default: `(num_steps - 1) * 0.16`)
  - Higher `b`: Slower decay, more gradual transition
  - Lower `b`: Faster decay, more concentration at start

**Use Cases:**
- Diffusion model noise scheduling
- Learning rate scheduling
- Annealing schedules

### 4. Rho-based Interpolation

Power transformation-based interpolation commonly used in diffusion models.

```python
values = interp.rho(rho=7, include_zero=False)
```

**Mathematical Formula:**

For output length $n$ (num_steps) and rho parameter $\rho$:

$$t_i = \left( \text{start}^{1/\rho} + \frac{i}{n-1} \cdot \left(\text{end}^{1/\rho} - \text{start}^{1/\rho}\right) \right)^\rho$$

where $i \in \{0, 1, 2, \ldots, n-1\}$ and $n > 1$. When $n = 1$, $t_0 = \text{start}$.

If `include_zero=True`, then $t_{n-1} = 0$ (the last point is replaced with zero).

**Parameters:**
- `rho` (float): Rho parameter controlling curve shape (default: 7)
- `include_zero` (bool): If True, replaces the last point with zero (default: False)

**Use Cases:**
- Diffusion model noise schedules
- Score-based generative models
- Advanced scheduling strategies

## Examples

### Example 1: Basic Comparison

```python
from interpolator import Interpolator
import matplotlib.pyplot as plt

interp = Interpolator(start=0.002, end=80.0, num_steps=180)

linear = interp.linear()
power = interp.power(p=3)
exponential = interp.exponential()
rho = interp.rho(rho=7)

plt.plot(linear.numpy(), label='Linear')
plt.plot(power.numpy(), label='Power')
plt.plot(exponential.numpy(), label='Exponential')
plt.plot(rho.numpy(), label='Rho')
plt.legend()
plt.show()
```

### Example 2: Diffusion Model Noise Scheduling

```python
from interpolator import Interpolator

# Typical diffusion model parameters
sigma_min = 0.002
sigma_max = 80.0
num_steps = 180

interp = Interpolator(start=sigma_min, end=sigma_max, num_steps=num_steps)

# Get different scheduling schemes
schedules = {
    'linear': interp.linear(),
    'power': interp.power(p=3),
    'exponential': interp.exponential(b=num_steps * 0.16),
    'rho': interp.rho(rho=7)
}

# Use in your diffusion model
for name, schedule in schedules.items():
    print(f"{name} schedule: {schedule.shape}")
```

### Example 3: Parameter Tuning

```python
# Test different parameters
interp = Interpolator(start=0.0, end=1.0, num_steps=50)

# Different power values
for p in [1, 2, 3, 5]:
    values = interp.power(p=p)
    print(f"p={p}: mid-point = {values[25]:.4f}")

# Different exponential rates
for b in [5, 10, 20, 50]:
    values = interp.exponential(b=b)
    print(f"b={b}: mid-point = {values[25]:.4f}")
```

See `example.py` for more comprehensive examples.

## API Reference

### `Interpolator` Class

#### Constructor

```python
Interpolator(start, end, num_steps, dtype=torch.float64)
```

**Parameters:**
- `start` (float): Starting value
- `end` (float): Ending value
- `num_steps` (int): Total number of output points (output length, must be >= 1)
- `dtype` (torch.dtype): Output tensor dtype (default: torch.float64)

#### Methods

- `linear() -> torch.Tensor`: Linear interpolation
- `power(p=3) -> torch.Tensor`: Power-based interpolation
- `exponential(b=None) -> torch.Tensor`: Exponential interpolation
- `rho(rho=7, include_zero=False) -> torch.Tensor`: Rho-based interpolation
- `interpolate(method, **kwargs) -> torch.Tensor`: Unified interface
- `get_all_methods(**kwargs) -> dict`: Get all methods at once

### `interpolate()` Function

```python
interpolate(start, end, num_steps, method="linear", dtype=torch.float64, **kwargs) -> torch.Tensor
```

Convenience function for quick interpolation without creating a class instance.

## Requirements

- Python 3.7+
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0 (for visualization examples)

## Use Cases

- **Diffusion Models**: Noise scheduling and time step generation
- **Score-based Models**: Noise level scheduling
- **Parameter Tuning**: Non-uniform parameter search spaces
- **Animation**: Smooth transitions between keyframes
- **Signal Processing**: Time-varying parameter generation
- **Machine Learning**: Learning rate scheduling, temperature annealing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This code is provided freely for public use without any license restrictions. Use it as you wish!


## Acknowledgments

This library was inspired by interpolation schemes commonly used in diffusion models and score-based generative models.


## Support

For issues, questions, or contributions, please open an issue on GitHub.

