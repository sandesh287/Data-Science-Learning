# Side-by-side comparison plots for all five distributions
# using subplot


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, expon, poisson, uniform


# creating a figure with 5 rows and 1 column (long layout) (vertical layout)
# plt.figure(figsize=(12, 18))

# creating a figure with 1 row and 5 column (wide layout) (horizontal layout)
plt.figure(figsize=(20, 4))


# 1. Uniform Distribution (continuous)
x = np.linspace(0, 1, 200)
pdf = uniform.pdf(x, loc=0, scale=1)

# plt.subplot(5, 1, 1)  # for long layout
plt.subplot(1, 5, 1)  # for wide layout
plt.plot(x, pdf, label='Uniform PDF', color='blue')
plt.title("Uniform Distribution (PDF)")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)


# 2. Normal Distribution (continuous)
x = np.linspace(-4, 4, 400)
normal_pdf = norm.pdf(x, loc=0, scale=1)

# plt.subplot(5, 1, 2)  # for long layout
plt.subplot(1, 5, 2)  # for wide layout
plt.plot(x, normal_pdf, color='red')
plt.title("Normal Distribution (PDF)  (mean=0, std=1)")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)


# 3. Binomial Distribution (discrete)
n = 10
p = 0.5
x_binom = np.arange(0, n+1)
pmf_binom = binom.pmf(x_binom, n, p)

# plt.subplot(5, 1, 3)  # for long layout
plt.subplot(1, 5, 3)  # for wide layout
plt.bar(x_binom, pmf_binom, color='green', alpha=0.7)
plt.title("Binomial Distribution PMF (n=10, p=0.5)")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.grid(True)


# 4. Exponential Distribution (continuous)
x_exp = np.linspace(0, 6, 300)
pdf_exp = expon.pdf(x_exp, scale=1)

# plt.subplot(5, 1, 4)  # for long layout
plt.subplot(1, 5, 4)  # for wide layout
plt.plot(x_exp, pdf_exp, color='purple')
plt.title("Exponential Distribution (λ=1) PDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)


# 5. Poisson Distribution (discrete)
lam = 3
x_pois = np.arange(0, 15)
pmf_pois = poisson.pmf(x_pois, lam)

# plt.subplot(5, 1, 5)  # for long layout
plt.subplot(1, 5, 5)  # for wide layout
plt.bar(x_pois, pmf_pois, color='orange', alpha=0.7)
plt.title("Poisson Distribution PMF (λ=3)")
plt.xlabel("Number of Events")
plt.ylabel("Probability")
plt.grid(True)


# altogether
plt.tight_layout()
plt.show()