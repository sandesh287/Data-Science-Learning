# 1. Apply One-Hot Encoding and Label Encoding to a dataset with categorical variables
# 2. Experiment with different encoding techniques and observe their impact on model performance
# Dataset: Titanic Dataset
# url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'



# libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load Titanic Dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Display dataset information
print('Dataset Information: \n')
print(df.info())

# Preview the first 5 rows
print('\nDataset Preview: \n', df.head())


# Handle Missing values before encoding
# Fill missing value 'Age' with median and 'Embarked' with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)



# Apply One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Display the encoded dataset
print('\nOne-Hot Encoded Dataset:\n')
print(df_one_hot.head())



# Apply Label Encoding (applied to nominal or ordinal features like Pclass)
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# Display encoded dataset
print('\nLabel Encoded Dataset:\n')
print(df[['Pclass', 'Pclass_encoded']].head())



# Apply Frequency Encoding (for high cardinality features like Ticket)
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# Display frequency encoded feature
print('\nFrequency Encoded Feature:\n')
print(df[['Ticket', 'Ticket_frequency']].head())



# To compare model performance before and after encoding
# Define features and target variables
X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = df['Survived']


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
print('Accuracy with One-Hot Encoding: ', accuracy_score(y_test, y_pred))