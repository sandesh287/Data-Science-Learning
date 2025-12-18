# Train and compare LightGBM, CatBoost, and XGBoost models on a dataset, focusing on their ability to handle large datasets and categorical data
# Dataset: Titanic Dataset

# Need to install python 3.12 for catboost, as 3.14 doesn't support it.
# I created a virtual environment for python 3.12 and installed all the packages required to run this file



# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# Load titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)


# Select features and target
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'


# Handle missing values
df.fillna({'Age': df['Age'].median()}, inplace=True)
df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)


# Encode categorical variables
label_encoders = {}
for col in ['Sex', 'Embarked']:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col])
  label_encoders[col] = le


# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training Data Shape: {X_train.shape}')
print(f'test Data Shape: {X_test.shape}')



# Train LightGBM model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)


# Predict and evaluate
lgb_pred = lgb_model.predict(X_test)
print(f'LightGBM Accuracy: {accuracy_score(y_test, lgb_pred)}')



# Train CatBoost model
cat_features = ['Pclass', 'Sex', 'Embarked']
cat_model = CatBoostClassifier(cat_features=cat_features, verbose=0)
cat_model.fit(X_train, y_train)


# Predict and evaluate
cat_pred = cat_model.predict(X_test)
print(f'CatBoost Accuracy: {accuracy_score(y_test, cat_pred)}')



# Train XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)


# Predict and evaluate
xgb_pred = xgb_model.predict(X_test)
print(f'XGBoost Accuracy: {accuracy_score(y_test, xgb_pred)}')


# Interpreting Result:
print('Here, we can see that accuracy for CatBoost is 0.8156, LightGBM is 0.8044 and XGBoost is 0.7709. Hence, we can conclude that CatBoost has the best one for this particular dataset, LightGBM is second best one and the worst that we have here is XGBoost with only 0.77 accuracy.')



# Experiment with handling categorical data with CatBoost

# Train CatBoost without encoding categorical features
cat_model_native = CatBoostClassifier(cat_features=['Sex', 'Embarked'], verbose=0)
cat_model_native.fit(X_train, y_train)


# Predict and evaluate
cat_pred_native = cat_model_native.predict(X_test)
print(f'CatBoost Native Accuracy: {accuracy_score(y_test, cat_pred_native)}')


# Interpreting result
print('Since we can clearly see that the CatBoost model without encoding (i.e. Native CatBoost) is also giving the same accuracy as CatBoost model with encoding, hence we can say that we need not encode dataset while using CatBoost.')