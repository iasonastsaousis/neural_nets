import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import pickle

# --- data loading ---

def unpickle(file):
    with open(file, 'rb') as fo:
        dataset = pickle.load(fo, encoding='bytes')
    return dataset


def load_dataset(path):
    training_dataset = unpickle(path / "train")
    test_dataset = unpickle(path / "test")
    meta = unpickle(path / "meta")
    class_names = [name.decode("utf-8") for name in meta[b"fine_label_names"]]
    return training_dataset, test_dataset, class_names


# --- data preprocessing ---

def preprocess_data(dataset):
    # Use the same numeric type and pixel scale for training and test images.
    X = dataset[b"data"].astype(np.float32) / 255.0
    y = np.array(dataset[b"fine_labels"])
    return X, y


def group_by_class(X, y):
    points = {}
    for category in np.unique(y):
        points[category] = X[y == category]
    return points


# --- classifiers and shared math ---

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

        # Exclude image arrays from comparisons when distance and category tie.
        return sorted(distances, key=lambda item: (item[0], item[1]))[: self.k]

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


# --- fitting ---

def fit_classifiers(points):
    classifiers = {
        "1-NN": KNearestNeighbors(k=1),
        "3-NN": KNearestNeighbors(k=3),
        "NCC": NearestClassCentroid(),
    }
    for classifier in classifiers.values():
        classifier.fit(points)
    return classifiers


# --- evaluation ---

def evaluate_classifier(classifier, X_test, y_test):
    """Return predictions and accuracy for an already fitted classifier."""
    if len(X_test) == 0:
        raise ValueError("The evaluation set must contain at least one image.")
    if len(X_test) != len(y_test):
        raise ValueError("Each evaluation image must have one label.")

    predictions = []
    correct_predictions = 0

    for i in range(len(X_test)):
        prediction = classifier.predict(X_test[i])
        predictions.append(prediction)
        if prediction == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(X_test)
    return np.array(predictions), accuracy


# --- visualization ---

def plot_predictions(X, y, class_names, predictions=None, model_name=None):
    """Show up to 16 images, optionally with already computed predictions."""
    count = min(16, len(X))
    if count == 0:
        return

    columns = min(4, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.5 * columns, 2.5 * rows), squeeze=False
    )
    if model_name is not None:
        fig.suptitle(model_name)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= count:
            continue

        image = X[i].reshape(3, 32, 32).transpose(1, 2, 0)
        ax.imshow(image)
        if predictions is None:
            ax.set_title(class_names[y[i]])
        else:
            ax.set_title(
                f"Predicted: {class_names[predictions[i]]}\n"
                f"Actual: {class_names[y[i]]}"
            )

    fig.tight_layout()
    plt.show()


# --- experiment runner ---

def main():
    path = Path(__file__).resolve().parent.parent / "dataset" / "cifar-100-python"
    training_dataset, test_dataset, class_names = load_dataset(path)
    X_train, y_train = preprocess_data(training_dataset)
    X_test, y_test = preprocess_data(test_dataset)

    # print(X_train.shape)  # (images, features): 32 * 32 * 3 = 3072 features
    # print(y_train.shape)  # One class label per image.
    # plot_predictions(X_train[:16], y_train[:16], class_names)

    points = group_by_class(X_train, y_train) # returns dict with {0: [500x3072], 1: [500x3072]}
    # print(len(points))
    # print(points[0].shape)
    # print(points[19].shape)

    classifiers = fit_classifiers(points)
    results = {}
    for name, classifier in classifiers.items():
        print(f"Evaluating {name} on {len(y_test)} test images...", flush=True)
        predictions, accuracy = evaluate_classifier(classifier, X_test, y_test)
        results[name] = predictions
        print(f"{name} test accuracy: {accuracy:.2%} ({len(y_test)} images)", flush=True)

    for name, predictions in results.items():
        plot_predictions(
            X_test[:16], y_test[:16], class_names, predictions[:16], model_name=name
        )


if __name__ == "__main__":
    main()
