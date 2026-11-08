"""Batched CPU versions of the pixel-distance classifiers in main.py."""

from collections import Counter

import numpy as np


def squared_distances(queries, references, reference_norms):
    """Use ||x-y||² = ||x||² + ||y||² - 2 x·y without a 3D temporary."""
    distances = queries @ references.T
    distances *= -2.0
    distances += np.einsum("ij,ij->i", queries, queries)[:, None]
    distances += reference_norms[None, :]
    # Cancellation can produce tiny negative values for coincident points.
    np.maximum(distances, 0.0, out=distances)
    return distances


def query_array(X, features):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != features:
        raise ValueError("Queries must have shape (images, training features).")
    if not np.isfinite(X).all():
        raise ValueError("Queries must contain finite values.")
    return X


class OptimizedKNearestNeighbors:
    def __init__(self, k=3, batch_size=128):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        self.k = k
        self.batch_size = batch_size
        self.X = None

    def fit(self, points):
        if not points:
            raise ValueError("Training data must not be empty.")
        categories = sorted(points)
        self.X = np.concatenate([points[c] for c in categories]).astype(np.float64)
        self.labels = np.concatenate([
            np.full(len(points[c]), c) for c in categories
        ])
        if self.X.ndim != 2 or len(self.X) < self.k or not np.isfinite(self.X).all():
            raise ValueError("Training data must be finite and contain at least k images.")
        self.norms = np.einsum("ij,ij->i", self.X, self.X)
        return self

    def kneighbors(self, X, k=None, progress=False):
        if self.X is None:
            raise ValueError("Fit the classifier before predicting.")
        k = self.k if k is None else k
        if not isinstance(k, int) or not 1 <= k <= len(self.X):
            raise ValueError("k must be between 1 and the training set size.")
        X = query_array(X, self.X.shape[1])
        neighbors = np.empty((len(X), k), dtype=np.int64)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            distances = squared_distances(X[start:stop], self.X, self.norms)
            for offset, row in enumerate(distances):
                threshold = np.partition(row, k - 1)[k - 1]
                # Include all boundary ties before ordering by distance/category/index.
                candidates = np.flatnonzero(row <= threshold)
                order = np.lexsort((candidates, self.labels[candidates], row[candidates]))
                neighbors[start + offset] = candidates[order[:k]]
            if progress:
                print(f"k-NN search: {stop}/{len(X)} images", flush=True)
        return neighbors

    def predict_for_k(self, X, ks=(1, 3), progress=False):
        """Reuse one exhaustive search for several values of k."""
        if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
            raise ValueError("Provide positive integer neighbor counts.")
        indices = self.kneighbors(X, k=max(ks), progress=progress)
        labels = self.labels[indices]
        return {
            k: np.array([
                Counter(row[:k]).most_common(1)[0][0] for row in labels
            ], dtype=self.labels.dtype)
            for k in ks
        }

    def predict_many(self, X):
        return self.predict_for_k(X, ks=(self.k,))[self.k]

    def predict(self, new_point):
        return self.predict_many(np.asarray(new_point)[None, :])[0]


class OptimizedNearestClassCentroid:
    def __init__(self, batch_size=128):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        self.batch_size = batch_size
        self.centroids = None

    def fit(self, points):
        if not points or any(len(values) == 0 for values in points.values()):
            raise ValueError("Every training class must contain at least one image.")
        # Preserve the original centroid calculation and class tie order.
        self.labels = np.array(list(points))
        self.centroids = np.array([
            np.mean(values, axis=0) for values in points.values()
        ], dtype=np.float64)
        if self.centroids.ndim != 2 or not np.isfinite(self.centroids).all():
            raise ValueError("Training data must be finite feature vectors.")
        self.norms = np.einsum("ij,ij->i", self.centroids, self.centroids)
        return self

    def predict_many(self, X):
        if self.centroids is None:
            raise ValueError("Fit the classifier before predicting.")
        X = query_array(X, self.centroids.shape[1])
        predictions = np.empty(len(X), dtype=self.labels.dtype)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            distances = squared_distances(X[start:stop], self.centroids, self.norms)
            predictions[start:stop] = self.labels[np.argmin(distances, axis=1)]
        return predictions

    def predict(self, new_point):
        return self.predict_many(np.asarray(new_point)[None, :])[0]
