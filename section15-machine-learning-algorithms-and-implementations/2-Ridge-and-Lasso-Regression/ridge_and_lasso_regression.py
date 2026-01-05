# Ridge and Lasso Regression
# Ridge and Lasso Regression are regularization techniques applied to Linear Regression to prevent overfitting by penalizing large coefficients:
  # Ridge Regression adds an L2 penalty (sum of squared coefficients)
  # Lasso Regression adds an L1 penalty (sum of absolute values of coefficients), which can lead to feature selection by shrinking some coefficients to zero.




# libraries
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np



# Sample data (eg. house size vs. house price)
X = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450], [1425], [1700]])
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Ridge Regression
ridge_model = Ridge(alpha=1.0)   # initializes ridge regression, alpha controls the strength of L2 regularization with higher values applying a stronger penalty to larger coefficients, alpha=1.0 is often a good starting point

ridge_model.fit(X_train, y_train)   # trains the ridge model on training data. The model learns the relationship between house size and price, fitting a line that minimizes error while applying the L2 regularization penalty

ridge_pred = ridge_model.predict(X_test)   # uses the trained ridge model to make predictions on the test set (X_test)

ridge_mse = mean_squared_error(y_test, ridge_pred)

print(f'\nRidge Mean Squared Error: {ridge_mse}')



# Lasso Regression
lasso_model = Lasso(alpha=0.1)   # initializes lasso regression, alpha controls the L1 regularization strength with larger values, increasing the penalty on the coefficients. alpha=0.1 applies a mild regularization

lasso_model.fit(X_train, y_train)   # trains the lasso model on training data. Like Ridge, the model learns the relationship between house size and price, but Lasso may shrink some coefficients to zero, effectively selecting features and reducing complexity.

lasso_pred = lasso_model.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_pred)

print(f'\nLasso Mean Squared Error: {lasso_mse}')