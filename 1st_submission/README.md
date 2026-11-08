# CIFAR-100: 1-NN, 3-NN and NCC

The Greek report is `report/report.pdf`; its source is `report/report.tex`.
Both implementations use all 50,000 training images and the official 10,000-image
test split, without PCA or a neural network.

## Files

- `src/main.py`: original explanatory implementation and evaluation/plotting pipeline.
- `src/optimized.py`: separate batched CPU implementations with deterministic tie rules.
- `src/benchmark.py`: reproducible reference comparison and full optimized evaluation.
- `src/test_optimized.py`: small reference, tie, and batch-boundary checks.
- `results/cpu_benchmark.json`: measured settings, timings, scores and source hashes.
- `results/predictions.npz`: full predictions and true labels, in original test order.
- `report/build_figures.py`: regenerate report figures from recorded results.

## Measured results

| Model | Correct / 10,000 | Accuracy |
|---|---:|---:|
| 1-NN | 1,755 | 17.55% |
| 3-NN | 1,748 | 17.48% |
| NCC | 1,004 | 10.04% |

The shared optimized 1-NN/3-NN search took 10.319 seconds on the recorded CPU,
excluding data loading and fitting. It includes neighbor selection, voting and
progress printing. This is one measured run, not a cross-machine guarantee.
Optimized NCC prediction took 0.0567 seconds.

On the fixed 100-image reference trial (one image per class), all three models
matched their originals on every prediction. NCC additionally matched its
original on all 10,000 test images. The original k-NN implementations were not
run on all test images; their full-run times are estimates, not measurements.
Distance arithmetic in the optimized models is float64; inputs and original
centroid means use float32. Mathematical equivalence is not a claim of bitwise
agreement for every possible nearly tied distance.

## Reproduce

Use Python 3 with NumPy, Matplotlib, and threadpoolctl. The measured environment
used Python 3.10.12, NumPy 1.24.3 and eight OpenBLAS threads; full metadata is in
the JSON. Keep the official CIFAR-100 Python files under `dataset/cifar-100-python/`.
The dataset is available at https://www.cs.toronto.edu/~kriz/cifar.html.

Run from this directory:

```bash
python3 -B -m unittest discover -s src -p test_optimized.py -v
MPLBACKEND=Agg python3 -B src/benchmark.py --full --threads 8 --batch-size 128
python3 -B report/build_figures.py
```

The benchmark refuses to overwrite `results/cpu_benchmark.json`. Before rerunning,
archive the existing `results/` directory under a different name; the runner
creates a fresh one. Keep the archived results if comparing timings between runs.
The report's numerical text describes the supplied run; if results change, update
that text together with the figures.

Running `python3 -B src/main.py` instead executes the original, substantially
slower implementation and opens plots after evaluation. No CUDA installation is
needed for either CPU implementation.

## Compile the report

Use XeLaTeX, IEEEtran, TikZ, listings, polyglossia and the FreeSerif, FreeSans and
DejaVu Sans Mono fonts. From `report/`, run:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
```

The supplied IEEE journal sample was used for the report structure and is
preserved under `report/template/`. The compiled PDF has no personal author name.
