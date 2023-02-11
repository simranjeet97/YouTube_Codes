import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create KNN classifier using Euclidean distance metric
knn_euclidean = KNeighborsClassifier(metric='euclidean')
knn_euclidean.fit(X_train, y_train)
y_pred_euclidean = knn_euclidean.predict(X_test)

# Create KNN classifier using Manhattan distance metric
knn_manhattan = KNeighborsClassifier(metric='manhattan')
knn_manhattan.fit(X_train, y_train)
y_pred_manhattan = knn_manhattan.predict(X_test)

# Calculate accuracy of predictions
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)
accuracy_manhattan = accuracy_score(y_test, y_pred_manhattan)

# Compare accuracy of predictions
print("Accuracy using Euclidean distance metric:", accuracy_euclidean)
print("Accuracy using Manhattan distance metric:", accuracy_manhattan)

# Output
# Accuracy using Euclidean distance metric: 0.9666666666666667
# Accuracy using Manhattan distance metric: 1.0