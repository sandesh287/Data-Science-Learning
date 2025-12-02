# Use sympy to compute second-order derivatives (Hessian Matrix)

# importing library
import sympy as sp

# define a function
x = sp.Symbol('x')
f = x**3 - 5*x + 7

# compute derivatives
first_derivative = sp.diff(f, x)
second_derivative = sp.diff(first_derivative, x)

shortcut_second_derivative = sp.diff(f, x, 2)  # here 2 represents second order derivation
print("Original Function: ", f)
print("First-order derivative: ", first_derivative)
print("Second-order derivative: ", second_derivative)
print("Second Derivative (using sympy directly): ", shortcut_second_derivative)