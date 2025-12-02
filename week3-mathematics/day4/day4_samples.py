# Calculus for Ml (Integrals and Optimization)

# importing sympy
import sympy as sp

x = sp.Symbol('x')
f = x**2
definite_integral = sp.integrate(f, (x, 0, 2))  # integrating f from 0 to 2
indefinite_integral = sp.integrate(f, x)
print(f"Definite Integral of {f} is: ", definite_integral)
print(f"Indefinite Integral of {f} is: ", indefinite_integral)