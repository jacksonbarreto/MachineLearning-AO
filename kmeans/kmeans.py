import numpy as np
from numpy.linalg import norm


def find_closest_cluster(distance):
    return np.argmin(distance, axis=1)


class KMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.centroids = None
        self.labels = None
        self.error = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def __initialize_centroids__(self, x):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(x.shape[0])
        centroids = x[random_idx[:self.n_clusters]]
        return centroids

    def __compute_centroids__(self, x, labels):
        centroids = np.zeros((self.n_clusters, x.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(x[labels == k, :], axis=0)
        return centroids

    def __compute_distance__(self, x, centroids):
        distance = np.zeros((x.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(x - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def __compute_sse__(self, x, labels, centroids):
        distance = np.zeros(x.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(x[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, x):
        self.centroids = self.__initialize_centroids__(x)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.__compute_distance__(x, old_centroids)
            self.labels = find_closest_cluster(distance)
            self.centroids = self.__compute_centroids__(x, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.__compute_sse__(x, self.labels, self.centroids)

    def predict(self, x):
        distance = self.__compute_distance__(x, self.centroids)
        return find_closest_cluster(distance)
