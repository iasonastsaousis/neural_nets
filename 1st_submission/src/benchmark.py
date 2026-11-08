"""Compare original and optimized CPU classifiers; optionally evaluate all tests."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from main import (KNearestNeighbors, NearestClassCentroid, evaluate_classifier,
                  group_by_class, load_dataset, preprocess_data)
from optimized import OptimizedKNearestNeighbors, OptimizedNearestClassCentroid


def timed(function):
    start = perf_counter()
    value = function()
    return value, perf_counter() - start


def score(predictions, labels):
    correct = int(np.count_nonzero(predictions == labels))
    return {"correct": correct, "count": len(labels), "accuracy": correct / len(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive")

    root = Path(__file__).resolve().parent.parent
    output = root / "results"
    output.mkdir(exist_ok=True)
    summary_path = output / "cpu_benchmark.json"
    if summary_path.exists():
        raise FileExistsError("results/cpu_benchmark.json already exists; preserve this run first.")
    train, test, names = load_dataset(root / "dataset" / "cifar-100-python")
    X_train, y_train = preprocess_data(train)
    X_test, y_test = preprocess_data(test)
    del train, test
    points = group_by_class(X_train, y_train)
    rng = np.random.default_rng(args.seed)
    # One test example per class makes this fixed trial cover all 100 classes.
    trial_indices = np.array([rng.choice(np.flatnonzero(y_test == c))
                              for c in np.unique(y_test)])
    X_trial, y_trial = X_test[trial_indices], y_test[trial_indices]
    source_hashes = {name: hashlib.sha256((root / "src" / name).read_bytes()).hexdigest()
                     for name in ("main.py", "optimized.py", "benchmark.py")}
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
    cpu = next((line.split(":", 1)[1].strip() for line in cpu_lines
                if line.startswith("model name")), platform.machine())
    summary = {
        "config": {"seed": args.seed, "batch_size": args.batch_size,
                   "blas_threads": args.threads, "device": "CPU", "cpu": cpu,
                   "python": platform.python_version(), "numpy": np.__version__,
                   "train_shape": list(X_train.shape), "test_shape": list(X_test.shape),
                   "input_dtype": str(X_train.dtype), "distance_dtype": "float64",
                   "trial_indices": trial_indices.tolist(), "trial_selection": "one per class",
                   "timing": "one warmed prediction pass, wall-clock seconds; excludes IO/plots",
                   "source_sha256": source_hashes},
        "original": {}, "optimized": {},
    }

    def save():
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    with threadpool_limits(limits=args.threads):
        summary["config"]["threadpools"] = [
            {key: pool.get(key) for key in ("internal_api", "num_threads", "architecture", "version")}
            for pool in threadpool_info()
        ]
        originals = {"1-NN": KNearestNeighbors(1), "3-NN": KNearestNeighbors(3),
                     "NCC": NearestClassCentroid()}
        reference_predictions = {}
        for name, model in originals.items():
            _, fit_seconds = timed(lambda: model.fit(points))
            model.predict(X_trial[0])
            print(f"Original {name}: 100-image trial...", flush=True)
            (predictions, accuracy), seconds = timed(
                lambda: evaluate_classifier(model, X_trial, y_trial))
            reference_predictions[name] = predictions
            summary["original"][name] = {
                "fit_seconds": fit_seconds, "trial_seconds": seconds,
                **score(predictions, y_trial),
            }
            print(f"  {seconds:.3f} s, accuracy {accuracy:.2%}", flush=True)

        knn = OptimizedKNearestNeighbors(batch_size=args.batch_size)
        ncc = OptimizedNearestClassCentroid(batch_size=args.batch_size)
        _, knn_fit = timed(lambda: knn.fit(points))
        _, ncc_fit = timed(lambda: ncc.fit(points))
        knn.predict_for_k(X_trial[:2])
        ncc.predict_many(X_trial[:2])
        for name, predict in (
                ("1-NN", lambda: knn.predict_for_k(X_trial, (1,))[1]),
                ("3-NN", lambda: knn.predict_for_k(X_trial, (3,))[3]),
                ("NCC", lambda: ncc.predict_many(X_trial))):
            predictions, seconds = timed(predict)
            agreement = int(np.count_nonzero(predictions == reference_predictions[name]))
            summary["optimized"][name] = {
                "fit_seconds": ncc_fit if name == "NCC" else knn_fit,
                "trial_seconds": seconds, **score(predictions, y_trial),
                "reference_agreement": agreement,
                "trial_speedup": summary["original"][name]["trial_seconds"] / seconds,
            }
            print(f"Optimized {name}: {seconds:.3f} s, agreement {agreement}/100", flush=True)
            if agreement != len(y_trial):
                save()
                raise RuntimeError("Investigate reference disagreement before the full run.")

        _, joint_seconds = timed(lambda: knn.predict_for_k(X_trial))
        summary["joint_knn_trial_seconds"] = joint_seconds
        save()
        if args.full:
            estimate = joint_seconds * len(X_test) / len(X_trial)
            print(f"Projected shared full k-NN pass: {estimate:.1f} s", flush=True)
            if estimate > 900:
                raise RuntimeError("CPU full-run estimate exceeds 15 minutes; reassess acceleration.")
            knn_predictions, joint_seconds = timed(
                lambda: knn.predict_for_k(X_test, progress=True))
            ncc_predictions, ncc_seconds = timed(lambda: ncc.predict_many(X_test))
            (original_ncc, _), original_ncc_seconds = timed(
                lambda: evaluate_classifier(originals["NCC"], X_test, y_test))
            summary["full"] = {
                "1-NN": score(knn_predictions[1], y_test),
                "3-NN": score(knn_predictions[3], y_test),
                "NCC": {**score(ncc_predictions, y_test), "seconds": ncc_seconds,
                        "reference_agreement": int(np.count_nonzero(ncc_predictions == original_ncc))},
                "shared_knn_seconds": joint_seconds,
                "original_NCC": {**score(original_ncc, y_test), "seconds": original_ncc_seconds},
                "original_knn_full_run": "not executed; only trial timings are measured",
            }
            np.savez_compressed(output / "predictions.npz", y_true=y_test,
                                knn1=knn_predictions[1], knn3=knn_predictions[3],
                                ncc=ncc_predictions, original_ncc=original_ncc,
                                trial_indices=trial_indices, class_names=np.array(names))
            save()
            print(json.dumps(summary["full"], indent=2), flush=True)
        print(f"Saved results to {summary_path.name}", flush=True)


if __name__ == "__main__":
    main()
