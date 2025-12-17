# 1. Create new feature from a date column (eg. day of the week, month, year)
# 2. Apply polynomial transformations to a dataset and compare model performance before and after transformation
# Objective: To derive new features from a date column in a dataset and then apply polynomial transformation  and evaluate their impact on model performance
# Dataset: Bike sharing dataset



# libraries
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Load Bike Sharing dataset
df = pd.read_csv('bike_sharing_daily.csv')


# Display dataset Information
print('Dataset Info:\n')
print(df.info())

# Preview the first five rows
print('\nDataset Preview:\n')
print(df.head())


# Convert dteday to datetime
df['dteday'] = pd.to_datetime(df['dteday'])


# Create new features
df['day_of_week'] = df['dteday'].dt.day_name()
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year


# Display the new features
print('\nNew Features derived from Date Column:\n')
print(df[['dteday', 'day_of_week', 'month', 'year']].head())



# Apply the polynomial transformation
# Polynomial Transformation helps capture the non-linear relationship in numerical features such as temperature that we have.

# Select fearture and target
X = df[['temp']]
y = df['cnt']


# Apply Polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)


# Display the transformed feature
print('\nOriginal and Polynomial Features:\n')
print(pd.DataFrame(X_poly, columns=['temp', 'temp^2']).head())



# Compare Model performance before and after the transformation

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_poly_train, X_poly_test = train_test_split(X_poly, test_size=0.2, random_state=42)


# Train and evaluate model with original features
model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
mse_original = mean_squared_error(y_test, y_pred_original)


# Train and evaluate model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_pred_poly = model_poly.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)


# Compare Results
print(f'MSE Original: {mse_original:.2f}')
print(f'MSE Polynomial: {mse_poly:.2f}')

print(f'Since {mse_poly:.2f} > {mse_original:.2f}, Polynomial is way better than the Original features')