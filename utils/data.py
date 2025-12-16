"""
Data generation for quadratic equations.
"""

import numpy as np
from typing import Tuple, Dict


def solve_quadratic(a: float, b: float, c: float) -> Tuple[complex, complex, str]:
    """Solve quadratic equation ax² + bx + c = 0."""
    if abs(a) < 1e-10:
        return complex(0, 0), complex(0, 0), "Invalid"
    
    discriminant = b**2 - 4*a*c
    
    if discriminant > 1e-10:
        root1 = (-b + np.sqrt(discriminant)) / (2*a)
        root2 = (-b - np.sqrt(discriminant)) / (2*a)
        return root1, root2, "Two real roots"
    elif abs(discriminant) < 1e-10:
        root = -b / (2*a)
        return root, root, "One real root"
    else:
        real_part = -b / (2*a)
        imag_part = np.sqrt(-discriminant) / (2*a)
        return complex(real_part, imag_part), complex(real_part, -imag_part), "Complex roots"


def generate_data(num_samples: int = 10000, a_range: Tuple[float, float] = (-10, 10),
                   b_range: Tuple[float, float] = (-10, 10), c_range: Tuple[float, float] = (-10, 10)) -> Dict[str, np.ndarray]:
    """Generate uniform training data."""
    inputs, targets = [], []
    
    for _ in range(num_samples):
        a = np.random.uniform(a_range[0], a_range[1])
        if abs(a) < 0.01:
            a = 0.01 if a >= 0 else -0.01
        b = np.random.uniform(b_range[0], b_range[1])
        c = np.random.uniform(c_range[0], c_range[1])
        
        root1, root2, disc_type = solve_quadratic(a, b, c)
        
        if disc_type == "Two real roots":
            targets.append([root1.real, root2.real, 0.0, 0.0])
        elif disc_type == "One real root":
            targets.append([root1.real, root1.real, 0.0, 0.0])
        else:
            targets.append([root1.real, root2.real, root1.imag, root2.imag])
        inputs.append([a, b, c])
    
    return {'inputs': np.array(inputs, dtype=np.float32), 'targets': np.array(targets, dtype=np.float32)}


def generate_stratified_data(num_samples: int = 10000, a_range: Tuple[float, float] = (-10, 10),
                             b_range: Tuple[float, float] = (-10, 10), c_range: Tuple[float, float] = (-10, 10)) -> Dict[str, np.ndarray]:
    """Generate stratified data balanced by root type."""
    inputs, targets = [], []
    samples_per_type = num_samples // 3
    
    # Real roots
    count = 0
    while count < samples_per_type:
        a = np.random.uniform(a_range[0], a_range[1])
        if abs(a) < 0.01:
            a = 0.01 if a >= 0 else -0.01
        b = np.random.uniform(b_range[0], b_range[1])
        c = np.random.uniform(c_range[0], c_range[1])
        discriminant = b**2 - 4*a*c
        if discriminant > 1e-10:
            root1, root2, _ = solve_quadratic(a, b, c)
            inputs.append([a, b, c])
            targets.append([root1.real, root2.real, 0.0, 0.0])
            count += 1
    
    # Complex roots
    count = 0
    while count < samples_per_type:
        a = np.random.uniform(a_range[0], a_range[1])
        if abs(a) < 0.01:
            a = 0.01 if a >= 0 else -0.01
        b = np.random.uniform(b_range[0], b_range[1])
        c = np.random.uniform(c_range[0], c_range[1])
        discriminant = b**2 - 4*a*c
        if discriminant < -1e-10:
            root1, root2, _ = solve_quadratic(a, b, c)
            inputs.append([a, b, c])
            targets.append([root1.real, root2.real, root1.imag, root2.imag])
            count += 1
    
    # Single roots
    count = 0
    while count < samples_per_type:
        a = np.random.uniform(a_range[0], a_range[1])
        if abs(a) < 0.01:
            a = 0.01 if a >= 0 else -0.01
        b = np.random.uniform(b_range[0], b_range[1])
        c = b**2 / (4*a) + np.random.uniform(-0.1, 0.1)
        root1, root2, _ = solve_quadratic(a, b, c)
        inputs.append([a, b, c])
        targets.append([root1.real, root1.real, 0.0, 0.0])
        count += 1
    
    indices = np.random.permutation(len(inputs))
    return {'inputs': np.array(inputs, dtype=np.float32)[indices], 'targets': np.array(targets, dtype=np.float32)[indices]}
