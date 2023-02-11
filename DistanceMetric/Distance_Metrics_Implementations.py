import numpy as np

# Euclidean Distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Manhattan Distance
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# Minkowski Distance
def minkowski_distance(x1, x2, p):
    return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)

# Hamming Distance
def hamming_distance(x1, x2):
    return np.sum(x1 != x2)

# Jaccard Distance
def jaccard_distance(x1, x2):
    intersect = np.sum(x1 * x2)
    union = np.sum(x1) + np.sum(x2) - intersect
    return 1 - intersect / union

# Cosine Similarity
def cosine_similarity(x1, x2):
    return np.dot(x1, x2) / (np.sqrt(np.dot(x1, x1)) * np.sqrt(np.dot(x2, x2)))

# Mahalanobis Distance
def mahalanobis_distance(x1, x2, VI):
    delta = x1 - x2
    return np.sqrt(np.dot(np.dot(delta, VI), delta))
