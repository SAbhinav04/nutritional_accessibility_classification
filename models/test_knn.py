import pandas as pd
from knn_distance import get_neighbors


# load dataset
X = pd.read_csv("dataset/features_X.csv")
y = pd.read_csv("dataset/target_y.csv")


# convert to numpy
X = X.values
y = y.values.flatten()


test_point = X[0]

neighbors = get_neighbors(X, y, test_point, k=3)

print("Nearest Neighbors:", neighbors)