import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load actual values
y_test = pd.read_csv("../models/y_test.csv")

# Load predicted values
predictions = pd.read_csv("random_forest_predictions.csv")

# Extract columns
y_true = y_test.iloc[:,0]
y_pred = predictions.iloc[:,0]

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Classification report
report = classification_report(y_true, y_pred)
print("\nClassification Report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()