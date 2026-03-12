import numpy as np
import pandas as pd


# ---------------------------------
# 1. Euclidean Distance Calculation
# ---------------------------------
def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    """

    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2

    return np.sqrt(distance)


# ---------------------------------
# 2. Find K Nearest Neighbors
# ---------------------------------
def get_neighbors(X_train, y_train, test_point, k=3):

    distances = []

    for i in range(len(X_train)):

        dist = euclidean_distance(test_point, X_train[i])

        distances.append((dist, y_train[i]))

    # sort by distance
    distances.sort(key=lambda x: x[0])

    neighbors = []

    for i in range(k):
        neighbors.append(distances[i][1])

    return neighbors