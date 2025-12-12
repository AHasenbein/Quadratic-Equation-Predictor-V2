"""
Utilities for quadratic equation calculations and visualization.
"""

import numpy as np
from sympy import symbols, latex, simplify, sympify
from typing import Tuple, Optional


def parse_equation(equation_str: str) -> Optional[Tuple[float, float, float]]:
    """
    Parse a quadratic equation string and extract coefficients a, b, c.
    Supports formats like: "x^2 + 2x + 1", "ax^2 + bx + c", etc.
    Returns (a, b, c) or None if invalid.
    """
    try:
        # Simple parsing - can be expanded
        x = symbols('x')
        expr = sympify(equation_str.replace('^', '**'))
        expr = simplify(expr)
        
        # Extract coefficients
        coeffs = expr.as_poly(x).all_coeffs()
        if len(coeffs) == 3:
            return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        elif len(coeffs) == 2:
            return float(coeffs[0]), float(coeffs[1]), 0.0
        elif len(coeffs) == 1:
            return float(coeffs[0]), 0.0, 0.0
    except:
        pass
    
    # Try direct coefficient input: "a b c" or "a,b,c"
    try:
        parts = equation_str.replace(',', ' ').split()
        if len(parts) == 3:
            return float(parts[0]), float(parts[1]), float(parts[2])
    except:
        pass
    
    return None


def solve_quadratic(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float], str]:
    """
    Solve quadratic equation ax^2 + bx + c = 0
    Returns (root1, root2, discriminant_type)
    """
    if a == 0:
        return None, None, "Not quadratic"
    
    discriminant = b**2 - 4*a*c
    
    if discriminant > 0:
        root1 = (-b + np.sqrt(discriminant)) / (2*a)
        root2 = (-b - np.sqrt(discriminant)) / (2*a)
        return root1, root2, "Two real roots"
    elif discriminant == 0:
        root = -b / (2*a)
        return root, root, "One real root"
    else:
        real_part = -b / (2*a)
        imag_part = np.sqrt(-discriminant) / (2*a)
        root1 = complex(real_part, imag_part)
        root2 = complex(real_part, -imag_part)
        return root1, root2, "Two complex roots"


def equation_to_latex(a: float, b: float, c: float) -> str:
    """
    Convert coefficients to LaTeX formatted equation.
    """
    x = symbols('x')
    expr = a * x**2 + b * x + c
    return latex(expr)


def generate_equation_data(a: float, b: float, c: float, x_range: Tuple[float, float] = (-10, 10), num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate x and y data points for plotting the quadratic equation.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = a * x**2 + b * x + c
    return x, y

