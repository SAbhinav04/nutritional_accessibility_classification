import pandas as pd
from knn_distance import get_neighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score

X_train = pd.read_csv("models/X_train.csv").values
y_train = pd.read_csv("models/y_train.csv").values.flatten()

X_test = pd.read_csv("models/X_test.csv").values
y_test = pd.read_csv("models/y_test.csv").values.flatten()


print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


def predict_class(neighbors):

    counts = {}

    for label in neighbors:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1

    return max(counts, key=counts.get)


k = 3
predictions = []

# Using only first 500 test samples
for i, test_point in enumerate(X_test[:500]):

    if i % 10 == 0:
        print("Processing test sample:", i)

    neighbors = get_neighbors(X_train, y_train, test_point, k)

    prediction = predict_class(neighbors)

    predictions.append(prediction)

accuracy = accuracy_score(y_test[:500], predictions)
precision = precision_score(y_test[:500], predictions, average="macro")
recall = recall_score(y_test[:500], predictions, average="macro")


print("\nKNN Results")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
