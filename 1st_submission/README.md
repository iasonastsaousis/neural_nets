# Setup and execution

Run the commands below from the extracted submission directory.

## Install dependencies

Use Python 3 and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib threadpoolctl
```

## Prepare CIFAR-100

Download the **CIFAR-100 Python version** from the
[official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html).
Extract it so these files are present:

```text
dataset/cifar-100-python/train
dataset/cifar-100-python/test
dataset/cifar-100-python/meta
```

The dataset is not included in the submission.

## Run the playground

```bash
python playground/playground.py
```

This displays k-NN and NCC on a small two-dimensional example. It does not
require CIFAR-100.

## Run the original classifiers

```bash
python src/main.py
```

This evaluates 1-NN, 3-NN and NCC on the full test set and then displays
prediction examples. The original k-NN implementation can take a long time.

## Recreate the full evaluation using the optimized classifiers

```bash
MPLBACKEND=Agg python src/benchmark.py --full --threads 8 --batch-size 128
```

This runs reference agreement checks and evaluates all 10,000 test images
using all 50,000 training images. It creates the `results/` directory with
new measurements and predictions; these generated files are not included
in the submission. No GPU is required.

Before repeating the command, rename the existing `results/` directory to
preserve the previous run. The runner refuses to overwrite an existing
`results/cpu_benchmark.json`.

## Run the automated checks

```bash
python -m unittest discover -s src -p test_optimized.py -v
```
