# Calculus for Machine Learning (Derivatives)

# importing library
# sympy library is used for derivatives problem
import sympy as sp

# Derivatives
x = sp.Symbol('x')
f = x**2
derivative = sp.diff(f, x)
print("Derivative: ", derivative)

# partial Derivatives and Gradients
x, y = sp.symbols('x y')
f = x**2 + y**2
grad_x = sp.diff(f, x)
grad_y = sp.diff(f, y)
print("Partial Derivatives: ", grad_x, grad_y)