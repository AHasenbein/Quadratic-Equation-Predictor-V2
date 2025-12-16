"""
Generate synthetic training data for quadratic equations.
Enhanced with stratified sampling, edge cases, and data quality metrics.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from utils.quadratic_utils import solve_quadratic


def generate_training_data(
    num_samples: int = 10000,
    a_range: Tuple[float, float] = (-10, 10),
    b_range: Tuple[float, float] = (-10, 10),
    c_range: Tuple[float, float] = (-10, 10),
    exclude_zero_a: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic training data for quadratic equations.
    
    Returns:
        Dictionary with 'inputs' (coefficients) and 'targets' (roots)
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        # Generate random coefficients
        a = np.random.uniform(a_range[0], a_range[1])
        if exclude_zero_a and abs(a) < 0.01:
            a = 0.01 if a >= 0 else -0.01
        
        b = np.random.uniform(b_range[0], b_range[1])
        c = np.random.uniform(c_range[0], c_range[1])
        
        # Solve for roots
        root1, root2, disc_type = solve_quadratic(a, b, c)
        
        # Store as real and imaginary parts
        if disc_type == "Two real roots":
            inputs.append([a, b, c])
            targets.append([root1.real, root2.real, 0.0, 0.0])  # [r1_real, r2_real, r1_imag, r2_imag]
        elif disc_type == "One real root":
            inputs.append([a, b, c])
            targets.append([root1.real, root1.real, 0.0, 0.0])
        else:  # Complex roots
            inputs.append([a, b, c])
            targets.append([root1.real, root2.real, root1.imag, root2.imag])
    
    return {
        'inputs': np.array(inputs, dtype=np.float32),
        'targets': np.array(targets, dtype=np.float32)
    }


def generate_training_data_around_equation(
    num_samples: int = 10000,
    center_equation: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    a_range: Tuple[float, float] = (-10, 10),
    b_range: Tuple[float, float] = (-10, 10),
    c_range: Tuple[float, float] = (-10, 10),
    exclude_zero_a: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic training data centered around a specific equation.
    Uses a weighted distribution that favors the center equation.
    
    Args:
        num_samples: Number of samples to generate
        center_equation: (a, b, c) tuple for the center equation
        a_range, b_range, c_range: Ranges for coefficients
        exclude_zero_a: Whether to exclude zero values for a
    
    Returns:
        Dictionary with 'inputs' (coefficients) and 'targets' (roots)
    """
    a_center, b_center, c_center = center_equation
    inputs = []
    targets = []
    
    # Calculate range sizes for normalization
    a_range_size = a_range[1] - a_range[0]
    b_range_size = b_range[1] - b_range[0]
    c_range_size = c_range[1] - c_range[0]
    
    for _ in range(num_samples):
        # Use a mix: 30% exactly at center, 50% close to center, 20% random in range
        rand = np.random.random()
        
        if rand < 0.3:
            # 30% exactly at center equation
            a, b, c = a_center, b_center, c_center
        elif rand < 0.8:
            # 50% close to center (within 20% of range)
            a = a_center + np.random.uniform(-0.2 * a_range_size, 0.2 * a_range_size)
            b = b_center + np.random.uniform(-0.2 * b_range_size, 0.2 * b_range_size)
            c = c_center + np.random.uniform(-0.2 * c_range_size, 0.2 * c_range_size)
        else:
            # 20% random in full range
            a = np.random.uniform(a_range[0], a_range[1])
            b = np.random.uniform(b_range[0], b_range[1])
            c = np.random.uniform(c_range[0], c_range[1])
        
        # Clamp to ranges
        a = np.clip(a, a_range[0], a_range[1])
        b = np.clip(b, b_range[0], b_range[1])
        c = np.clip(c, c_range[0], c_range[1])
        
        # Ensure a is not zero
        if exclude_zero_a and abs(a) < 0.01:
            a = 0.01 if a >= 0 else -0.01
        
        # Solve for roots
        root1, root2, disc_type = solve_quadratic(a, b, c)
        
        # Store as real and imaginary parts
        if disc_type == "Two real roots":
            inputs.append([a, b, c])
            targets.append([root1.real, root2.real, 0.0, 0.0])
        elif disc_type == "One real root":
            inputs.append([a, b, c])
            targets.append([root1.real, root1.real, 0.0, 0.0])
        else:  # Complex roots
            inputs.append([a, b, c])
            targets.append([root1.real, root2.real, root1.imag, root2.imag])
    
    return {
        'inputs': np.array(inputs, dtype=np.float32),
        'targets': np.array(targets, dtype=np.float32)
    }

