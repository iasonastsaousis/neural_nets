import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --- data ---

POINTS = {
    "blue": [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
    "red": [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]],
}

NEW_POINT = [3, 3]

CLASS_COLORS = {"blue": "#104DCA", "red": "#FF0000"}


# --- shared math ---

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


# --- models ---

class KNearestNeighbors:

    def __init__(self, k=3):
        self.k = k
        self.points = None

    def fit(self, points):
        self.points = points

    def k_nearest(self, new_point):
        """Return the k closest training points as (distance, category, point)."""
        distances = []

        for category, class_points in self.points.items():
            for point in class_points:
                distance = euclidean_distance(point, new_point)
                distances.append((distance, category, point))

        return sorted(distances)[: self.k]

    def predict(self, new_point):
        categories = [category for _, category, _ in self.k_nearest(new_point)]
        return Counter(categories).most_common(1)[0][0]


class NearestClassCentroid:

    def __init__(self):
        self.centroids = None

    def fit(self, points):
        # Example for this dataset: {"blue": [2.0, 2.6], "red": [4.8, 5.4]}
        self.centroids = {
            category: np.mean(class_points, axis=0)
            for category, class_points in points.items()
        }

    def predict(self, new_point):
        return min(
            self.centroids,
            key=lambda category: euclidean_distance(
                self.centroids[category], new_point
            ),
        )


# --- visualization ---

def setup_axes(ax, title):
    ax.set_facecolor("black")
    ax.grid(True, color="#323232")
    ax.figure.set_facecolor("#121212")
    ax.tick_params(axis="x", color="white")
    ax.tick_params(axis="y", color="white")
    ax.set_title(title, color="white")


def plot_training_points(ax, points):
    for category, class_points in points.items():
        for x, y in class_points:
            ax.scatter(x, y, color=CLASS_COLORS[category], s=60)


def plot_query_point(ax, new_point, predicted_class):
    ax.scatter(
        new_point[0],
        new_point[1],
        color=CLASS_COLORS[predicted_class],
        marker="*",
        s=200,
        zorder=100,
    )


def plot_knn(ax, points, new_point, clf):
    predicted_class = clf.predict(new_point)

    setup_axes(ax, f"KNN (k={clf.k}) - predicts {predicted_class}")
    plot_training_points(ax, points)
    plot_query_point(ax, new_point, predicted_class)

    # Only the neighbors that actually voted get a line.
    for distance, category, point in clf.k_nearest(new_point):
        ax.plot(
            [new_point[0], point[0]],
            [new_point[1], point[1]],
            color=CLASS_COLORS[category],
            linestyle="--",
            linewidth=1,
        )


def plot_centroid(ax, points, new_point, clf):
    predicted_class = clf.predict(new_point)

    setup_axes(ax, f"Nearest class centroid - predicts {predicted_class}")
    plot_training_points(ax, points)
    plot_query_point(ax, new_point, predicted_class)

    for category, centroid in clf.centroids.items():
        ax.scatter(
            centroid[0],
            centroid[1],
            color=CLASS_COLORS[category],
            marker="X",
            s=120,
            zorder=90,
        )
        ax.plot(
            [new_point[0], centroid[0]],
            [new_point[1], centroid[1]],
            color=CLASS_COLORS[category],
            linestyle="--",
            linewidth=1,
        )


# --- experiment runner ---

def main():
    knn = KNearestNeighbors(k=3)
    knn.fit(POINTS)

    centroid_clf = NearestClassCentroid()
    centroid_clf.fit(POINTS)

    print("KNN prediction:      ", knn.predict(NEW_POINT))
    print("Centroid prediction: ", centroid_clf.predict(NEW_POINT))

    fig, (ax_knn, ax_centroid) = plt.subplots(1, 2, figsize=(12, 6))
    fig.set_facecolor("#121212")

    plot_knn(ax_knn, POINTS, NEW_POINT, knn)
    plot_centroid(ax_centroid, POINTS, NEW_POINT, centroid_clf)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
