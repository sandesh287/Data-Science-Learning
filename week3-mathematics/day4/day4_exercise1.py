# Calculate Integrals of Simple Functions

# import libraries
import sympy as sp

# define a function
x = sp.Symbol('x')
f = sp.exp(-x)

# compute indefinite integral
indefinite_integral = sp.integrate(f, x)
print("Indefinite Integral: ", indefinite_integral)

# compute definite integral
definite_integral = sp.integrate(f, (x, 0, sp.oo))  # here sp.oo represents infinity
print("Definite Integral: ", definite_integral)