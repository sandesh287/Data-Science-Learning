# Explore other distributions (eg. normal, binomial) using Python
# Normal distributon (continuous), Binomial distribution (discrete), Exponential distribution (continuous), Poisson distribution (discrete), Uniform distribution (continuous)


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, expon, poisson, uniform


# 1. Normal Distribution (continuous)
# Mean = 0, std = 1 (standard normal)

# Normal distribution: PDF
x = np.linspace(-4, 4, 400)
pdf = norm.pdf(x, loc=0, scale=1)
plt.plot(x, pdf)
plt.title('PDF of Normal Distribution (mean=0, std=1)')
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.show()

# Histogram of simulated data
data = np.random.normal(0, 1, 10000)
plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue')
plt.plot(x, pdf, color='red')
plt.title("Normal Distribution (Simulated Data)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()


# 2. Binomial Distribution (discrete)
# Eg. Flip a coin 10 times, probability of heads = 0.5

n = 10  # number of trials
p = 0.5  # probability of success

x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)

# Bar plot
plt.bar(x, pmf, alpha=0.7)
plt.title("PMF of Binomial Distribution (n=10, p=0.5)")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.show()


# 3. Exponential Distribution (continuous)
# used for modeling waiting times. λ=1

x = np.linspace(0, 6, 200)
pdf = expon.pdf(x, scale=1)

# Visualization
plt.plot(x, pdf)
plt.title("PDF of Exponential Distribution (λ=1)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()


# 4. Poisson Distribution (discrete)
# Eg. average rate λ=3 events per interval

lambda_val = 3
x = np.arange(0, 15)
pmf = poisson.pmf(x, lambda_val)

# Bar plot
plt.bar(x, pmf, alpha=0.7, color='green')
plt.title("PMF of Poisson Distribution (λ=3)")
plt.xlabel("Number of Events")
plt.ylabel("Probability")
plt.show()


# 5. Uniform Distribution (continuous)

x = np.linspace(0, 1, 100)
pdf = uniform.pdf(x, loc=0, scale=1)

# Visualization
plt.plot(x, pdf, color='red')
plt.title("PDF of Uniform Distribution (0 to 1)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()