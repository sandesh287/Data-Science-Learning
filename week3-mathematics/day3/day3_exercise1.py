# Compute Derivatives of Basic Functions

# importing libraries
import numpy as np
import sympy as sp

# define a function
x = sp.Symbol('x')
f = x**3 - 5*x + 7

# compute derivative
derivative = sp.diff(f, x)
print(f"Derivative of Function {f} is: ", derivative)