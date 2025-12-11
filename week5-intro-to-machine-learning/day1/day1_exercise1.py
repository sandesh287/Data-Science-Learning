# 1. Define Features and Target Variables
# 2. Split Data into Training and Testing Sets
# 3. Visualize the Dataset
# Objective 1: work on real-world dataset to identify features and target variables
# Objective 2: split the data into 80% training and 20% testing set using train_test_split
# Objective 3: explore the data and visualize relationship between features and target variable
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# Define features and target (total_bill, size as features), (tip as target)
features = df[['total_bill', 'size']]
target = df['tip']

print('Features: \n', features.head())  # head() displays first 5 items in the list
print('Target: \n', target.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Training Data Set: ", X_train.shape)
print("Testing Data Set: ", X_test.shape)

# Visualize the relationships
sns.pairplot(df, x_vars=['total_bill', 'size'], y_vars='tip', height=7, aspect=0.8, kind='scatter')
plt.title('Features vs Target Relationships')
plt.show()