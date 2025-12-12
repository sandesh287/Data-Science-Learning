# Feature Importance with Lasso:
  # Use Lasso Regression to perform feature selection and identify the most relevant predictors

# Lasso Feature importance (Feature Selection)


# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


# 1. load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame


# Features and target
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']


# 2. (Optional) Create Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

feature_names = poly.get_feature_names_out(X.columns)

print('Original Features: ', X.shape[1])
print('Polynomial Features: ', X_poly.shape[1])


# 3. Scale features
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X_poly)


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 5. Lasso Regression for Feature Selection
lasso_model = Lasso(alpha=0.01, max_iter=10000)
lasso_model.fit(X_train, y_train)


# 6. Get Feature Importance (absolute coefficients)
coeffs = lasso_model.coef_
importance = np.abs(coeffs)

# Create DataFrames
feature_importance = pd.DataFrame({
  'feature': feature_names,
  'importance': importance
})

# Sort by importance
feature_importance = feature_importance.sort_values(by='importance', ascending=False)

print(f'\nTop 20 most important features (Lasso):\n {feature_importance.head(20)}')


# Visualization
top_n = 20  # display top 20
plt.figure(figsize=(14,8))
plt.barh(
  feature_importance['feature'].head(top_n),
  feature_importance['importance'].head(top_n)
)
plt.gca().invert_yaxis()  # most important at top
plt.xlabel('Importance (|Lasso Coefficient|)')
plt.title('Top Feature Importances using Lasso Regression')
plt.show()