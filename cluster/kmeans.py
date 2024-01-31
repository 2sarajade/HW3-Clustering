import numpy as np
from scipy.spatial.distance import cdist
import random


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        #error handling
        if type(k) is not int or k < 1:
            raise ValueError("invalid input for k")
        if max_iter < 0:
            raise ValueError("invalid input for max_iter")
        if tol < 0:
            raise ValueError("invalid input for tol")

        #initialize params
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centers = []

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        #error handling
        self.point_dims = mat.shape[1]

        # kmeans++ initialization
        first_center = random.choice(range(self.k))
        self.centers.append(mat[first_center,])
        for _ in range(1, self.k):
            distances, centroids = self._calc_distances(mat)
            new_centroid_index = random.choices(range(mat.shape[0]), weights = distances, k = 1)
            self.centers.append(mat[new_centroid_index,])

        # kmeans algorithm
        iter = 0
        diff = np.inf
        while iter < self.max_iter and diff > self.tol:
            distances, centroids = self._calc_distances(mat)
            old_centers = self.centers.copy()
            for i in range(len(self.centers)):
                ind_points_in_clust = [j for j in range(mat.shape[0]) if centroids[j] == i]
                points_in_clust = mat[ind_points_in_clust, ]
                self.centers[i] = np.mean(points_in_clust, axis = 0)
            sum_dist = 0
            for i in range(len(self.centers)):
                sum_dist += np.linalg.norm(self.centers[i]-old_centers[i])
            diff = sum_dist
            iter += 1
        self.error = diff


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        #error handling
        if mat.shape[1] != self.point_dims:
            raise ValueError("points do not have same dimensions as training set")
        
        #predict
        distances, centroids = self._calc_distances(mat)
        return centroids


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centers

    def _calc_distances(self, points):
        distances = []
        centroids = []
        for i in range(np.shape(points)[0]):
            min_distance = np.inf
            closest_centroid = 0
            for j in range(len(self.centers)):
                center = self.centers[j]
                distance = np.linalg.norm(center-points[i,])
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = j
            distances.append(min_distance)
            centroids.append(closest_centroid)
        return distances, centroids
