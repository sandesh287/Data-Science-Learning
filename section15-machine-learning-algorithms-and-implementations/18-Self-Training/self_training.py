# Self-Training
# Self-Training is the semi-supervised learning approach that leverages a small labeled dataset alongside a larger unlabeled dataset. The model is initially trained on labeled data, and then it makes predictions on the unlabeled data. The confident predictions (those with high certainty) are then added to the labeled dataset, and the process is repeated to improve the model.




# libraries
from sklearn.ensemble import RandomForestClassifier   # used for supervised classification task
from sklearn.datasets import make_classification   # generates a synthetic dataset for classification tasks
from sklearn.model_selection import train_test_split   # splits the dataset into training and testing subsets
from sklearn.metrics import accuracy_score   # calculates the accuracy of predictions
import numpy as np   # for numerical operations like array manipulation



# Generate a synthetic dataset
X, y = make_classification(n_samples=200, n_features=5, random_state=42)   # generates a synthetic dataset with 200 samples, 5 features using random_state=42 for reproducibility

X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.7, random_state=42)   # it splits the dataset into labeled and unlabeled portions, with 30% labeled for initial training and 70% unlabeled. "X_labeled, y_labeled", holds the labeled data and "X_unlabeled", contains the unlabeled data. Its labels are ignored for the self training approach here.



# Initialize and train the model with labeled data
model = RandomForestClassifier(random_state=42)   # initialize random forest classifier with "random_state=42" to ensure reproducibility

model.fit(X_labeled, y_labeled)   # trains the model on the initially labeled data



# Perform self-training on unlabeled data
# sets a loop to repeat the self-training process 5 times, allowing the model to iteratively expand its label dataset by adding confident predictions
for _ in range(5):
  # Predict the probabilities on unlabeled data
  probs = model.predict_proba(X_unlabeled)   # This uses model to predict class probabilities for each sample in unlabeled dataset
  high_confidence_idx = np.where(np.max(probs, axis=1) > 0.75)[0]   # Select confidence predictions. This finds indices of samples where the model's maximum probability exceeds 0.9, including high confidence in these predictions
  
  
  if len(high_confidence_idx)  == 0:
    print('No high-confidence samples found. Stopping self-training.')
    break
  
  
  # Add high-confidence predictions to labeled data
  X_labeled = np.vstack([X_labeled, X_unlabeled[high_confidence_idx]])   # This adds the high confidence samples from X_unlabeled to the X_labeled dataset using np.vstack.
  
  y_labeled = np.hstack([y_labeled, model.predict(X_unlabeled[high_confidence_idx])])   # This adds the corresponding predicted labels for these high confidence samples to y_labeled using np.hstack
  
  
  # Remove confident samples from the unlabeled dataset
  X_unlabeled = np.delete(X_unlabeled, high_confidence_idx, axis=0)   # This will remove the confident samples from X_unlabeled leaving only uncertain samples for the next iteration
  
  
  # Re-train the model on the expanded labeled dataset
  model.fit(X_labeled, y_labeled)   # re-train the model on the expanded labeled dataset, incorporating new samples from each iteration



# Final evaluation on a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# model.fit(X_train, y_train)   # trains the model in full training set


y_pred = model.predict(X_test)   # predicts labels for test set


accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')   # prints the final accuracy score, providing a performance measure for the self-trained model