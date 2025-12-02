# compute Gradients

# importing library
import sympy as sp

# define a multi variable function
x, y = sp.symbols('x y')
f = x**2 + 3*y**2 -4*x*y

# compute partial derivative
grad_x = sp.diff(f, x)
grad_y = sp.diff(f, y)

print("Original function: ", f)
print("Gradients of X: ", grad_x)
print("Gradients of Y: ", grad_y)