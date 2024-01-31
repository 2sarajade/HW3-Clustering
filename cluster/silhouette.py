import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # error handling
        if X.shape[0] != len(y):
            raise ValueError("Dimensionsn of inputs do not match")

        # calculate scores
        scores = []
        for i in range(len(y)):
            a = self._calculate_a(i, X, y)
            b = self._calculate_b(i, X, y)
            max_ab = max(a, b)
            if max_ab != 0:
                scores.append((b-a)/max_ab)
            else:
                scores.append(0)
        return scores

    def _calculate_a(self, i, X, y):
        i_cluster = y[i]
        i_point = X[i,]
        distances = [np.linalg.norm(i_point-X[point,]) for point in range(len(y)) if point != i and y[point] == i_cluster]
        #if distances:
        return np.mean(distances)
       # else:
        #    return 0.0
    
    def _calculate_b(self, i, X, y):
        i_cluster = y[i]
        i_point = X[i,]
        b_values = []
        for cluster_label in np.unique(y):
            if cluster_label != i_cluster:
                distances = [np.linalg.norm(i_point-X[point]) for point in range(len(y)) if y[point] == cluster_label]
                b_values.append(np.mean(distances))
        #if b_values:
        return min(b_values)
        #else:
        #    return 0.0