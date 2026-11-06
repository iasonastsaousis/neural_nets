import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import pickle

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


# --- load dataset ---

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict





# --- experiment runner ---

def main():

    # Loading the Dataset

    path = Path(__file__).resolve().parent.parent / "dataset" / "cifar-100-python"
    training_dataset = unpickle(path / "train")
    validation_dataset = unpickle(path / "test")
    meta = unpickle(path / "meta")

    class_names = [
        name.decode("utf-8")
        for name in meta[b"fine_label_names"]
    ]

    # --------------------------------------------------- #

    # Exploring the Dataset

    # 3072 columns because we have 32x32 images, 3 color channels RGB
    # 32x32x3 = 1024x3 = 3072 features

    X = training_dataset[b"data"]
    y = np.array(training_dataset[b"fine_labels"])

    print(X.shape)  # X is a vector with dimensions 50000 
                    #(row = number of images), 3072 (column = number of features)
    print(y.shape)  # Y is a 50000x1 vector (a class for each image)


    # Plotting 16 images
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):

        image = X[i].reshape(3, 32, 32).transpose(1, 2, 0)

        ax.imshow(image)
        ax.set_title(class_names[y[i]])
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Making the dataset compatible with our classifiers

    points = {}

    for category in np.unique(y):
        points[category] = X[y == category]

    # Now points is a dict with:
    # { 0: [500x3076]
    #   1: [500x3076]} etc
    print(len(points))       # 100
    print(points[0].shape)   # (500, 3072)
    print(points[19].shape)  # (500, 3072)

    ncc = NearestClassCentroid()

    ncc.fit(points)

    print(len(ncc.centroids))
    print(ncc.centroids[0].shape)

    X_test = validation_dataset[b"data"]
    y_test = np.array(validation_dataset[b"fine_labels"])

    new_image = X_test[0]

    prediction = ncc.predict(new_image)

    print("Predicted:", class_names[prediction])
    print("Actual:", class_names[y_test[0]])

    image = new_image.reshape(3, 32, 32).transpose(1, 2, 0)

    plt.imshow(image)
    plt.title(
        f"Predicted: {class_names[prediction]}\n"
        f"Actual: {class_names[y_test[0]]}"
    )
    plt.axis("off")
    plt.show()
    



if __name__ == "__main__":
    main()