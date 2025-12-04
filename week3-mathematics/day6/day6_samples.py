# Statistics Fundamentals

# importing libraries
from statistics import mode

# Central Tendency
data = [10, 20, 30, 40, 50]
mean = sum(data) / len(data)
print("Mean: ",mean)

sorted_data = sorted(data)
median = sorted_data[len(data) // 2] if len(data) % 2 != 0 else \
  (sorted_data[len(data) // 2 - 1] + sorted_data[len(data) // 2]) / 2
print("Median: ", median)

print("Mode: ", mode(data))

# Dispersion
variance = sum((x - mean) ** 2 for x in data) / len(data)
print("Variance: ", variance)

standard_deviation = variance ** 0.5
print("Standard Deviation: ", standard_deviation)


# Confidence Interval
import scipy.stats as stats

data = [10, 20, 30, 40, 50]
mean = sum(data) / len(data)

variance = sum((x - mean) ** 2 for x in data) / len(data)
standard_deviation = variance ** 0.5

sample_mean = mean
z_score = 1.96

confidence_interval = (sample_mean - z_score * (standard_deviation / len(data) ** 0.5),
                       sample_mean + z_score * (standard_deviation / len(data) ** 0.5))
print("95% confidence interval: ", confidence_interval)