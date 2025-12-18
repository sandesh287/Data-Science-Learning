# Ensemble Learning and Model Comparison
# Train and compare multiple ensemble models on a real-world dataset, analyzing their performance under balanced and imbalanced conditions
# Dataset: Telco-Customer-Churn Dataset, which predicts whether a customer will leave a service provider based on demograpghic and usage data



# libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score


# Load dataset
df = pd.read_csv('Telco-Customer-Churn.csv')


# Display dataset info and preview
print('Dataset Info:\n')
print(df.info())
print('\nClass Distribution:\n')
print(df['Churn'].value_counts())
print('\nSample Data:\n', df.head())


# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)


# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
  if column != 'Churn':
    df[column] = label_encoder.fit_transform(df[column])


# Encode target variable
df['Churn'] = label_encoder.fit_transform(df['Churn'])


# Scale numerical features
scalar = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_features] = scalar.fit_transform(df[numerical_features])


# Features and Target
X = df.drop(columns=['Churn'])
y = df['Churn']


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Display class distribution after SMOTE
print('\nClass Distribution after SMOTE:\n')
print(pd.Series(y_train_resampled).value_counts())



# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])


# Train XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])


# Train LightGBM model
lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_lgb = lgb_model.predict(X_test)
roc_auc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:,1])


# Classification Reports
print('Classification Reports:\n')
print('Random Forest Report:\n', classification_report(y_test, y_pred_rf))
print('XGBoost Report:\n', classification_report(y_test, y_pred_xgb))
print('LightGBM Report:\n', classification_report(y_test, y_pred_lgb))


# ROC-AUC Comparison
print('ROC-AUC Scores:\n')
print(f'Random Forest: {roc_auc_rf}')
print(f'XGBoost: {roc_auc_xgb}')
print(f'LightGBM: {roc_auc_lgb}')