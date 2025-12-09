# Conduct sampling and create a Report
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# importing libraries
import pandas as pd
import numpy as np
from scipy.stats import norm, t

# load iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# sampling
sample = df["sepal_length"].sample(30, random_state=42)

# sample statistics
mean = sample.mean()
std = sample.std()
n = len(sample)

# confidence interval
z_value = norm.ppf(0.975)
margin_of_error = z_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)

print("Sample Mean: ", mean)
print(f"95% Confidence Interval: ({ci[0].item()}, {ci[1].item()})")