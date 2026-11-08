"""Build report figures from saved measurements, without rerunning classifiers."""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    report = Path(__file__).resolve().parent
    root = report.parent
    figures = report / "figures"
    figures.mkdir(exist_ok=True)
    summary = json.loads((root / "results" / "cpu_benchmark.json").read_text())
    predictions = np.load(root / "results" / "predictions.npz")
    with (root / "dataset" / "cifar-100-python" / "test").open("rb") as stream:
        test = pickle.load(stream, encoding="bytes")
    labels = np.array(test[b"fine_labels"])
    np.testing.assert_array_equal(labels, predictions["y_true"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    names = ["1-NN", "3-NN", "NCC"]
    colors = ["#17395c", "#147d83", "#b16c13"]

    figure, axis = plt.subplots(figsize=(5.4, 3.2), layout="constrained")
    values = [summary["full"][name]["accuracy"] * 100 for name in names]
    bars = axis.bar(names, values, color=colors, width=0.6)
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)
    axis.axhline(1, color="#777777", linestyle="--", linewidth=1)
    axis.text(2.35, 1.4, "1%", color="#555555", ha="right", fontsize=8)
    axis.set(ylabel="Accuracy (%)", ylim=(0, 21))
    axis.grid(axis="y", alpha=0.16)
    axis.set_axisbelow(True)
    figure.savefig(figures / "accuracy.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(3, 4, figsize=(10.4, 7.8))
    class_names = predictions["class_names"]
    selections = {}
    for row, (name, key) in enumerate(zip(names, ("knn1", "knn3", "ncc"))):
        predicted = predictions[key]
        assert int(np.sum(predicted == labels)) == summary["full"][name]["correct"]
        correct = np.flatnonzero(predicted == labels)[:2]
        wrong = np.flatnonzero(predicted != labels)[:2]
        indices = np.concatenate((correct, wrong))
        selections[name] = indices.tolist()
        for column, index in enumerate(indices):
            axis = axes[row, column]
            image = test[b"data"][index].reshape(3, 32, 32).transpose(1, 2, 0)
            axis.imshow(image, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"#{index} · {class_names[labels[index]].replace('_', ' ')}\n"
                f"→ {class_names[predicted[index]].replace('_', ' ')}",
                fontsize=9, color="#147d83" if column < 2 else "#b23a35",
            )
            if column == 0:
                axis.set_ylabel(name, fontsize=13, labelpad=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(figures / "classification_examples.pdf", bbox_inches="tight")
    plt.close(figure)
    (figures / "example_indices.json").write_text(json.dumps(selections, indent=2) + "\n")


if __name__ == "__main__":
    main()
