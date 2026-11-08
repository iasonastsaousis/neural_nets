# Classifier experiments

## Package Instructions

- Follow the parent workspace's learning-first collaboration rules.
- `main.py` loads CIFAR-100, displays images, and implements k-NN and NCC.
- Preserve the user's classifier implementations during preprocessing changes.
- For a noninteractive smoke check, use `MPLBACKEND=Agg python3 -B src/main.py`
  from the repository root with NumPy and Matplotlib available.

---

## Progress Log

### Branch: master

#### Phase 1: Floating-point image preprocessing

Design decisions approved by the user before implementation:
1. Convert training and test image arrays to `float32` and divide by 255
   to represent pixel intensities in [0, 1].
2. Apply this to model inputs; retain integer class labels and original dataset files.

- [x] Apply the same preprocessing to both image arrays.
- [x] Check data type, range, and compatibility with the existing pipeline.
- [x] Explain conversion, scaling, and their effect on distance comparisons.

Validation: ran the existing script with a noninteractive plotting backend.
Both image arrays are finite `float32` values spanning [0, 1], with their original
shapes. Grouped training points remain `float32`; labels remain integers.
The existing NCC fitting and single-image prediction completed. This is a smoke
check, not an accuracy experiment. The environment emitted an unavailable 3D
plotting warning; no 3D plots are used here.

Learning notes: conversion enables floating-point arithmetic; dividing every
feature by 255 changes the common scale. In exact arithmetic, all Euclidean
distances shrink by the same factor, preserving nearest-neighbor and nearest-
centroid decisions. Neural-network training depends on numeric scale; [0, 1]
is a common input convention, not a guarantee of better accuracy.

#### Phase 2: Organize the experiment into functions

Design decisions approved by the user before implementation:
1. Keep the pipeline in `main.py`, separating loading, preprocessing, shared
   math/classifiers, fitting, evaluation, and visualization.
2. Use option A: refactor first and retain one-image-at-a-time prediction.
   Optimized batch prediction is a later learning step.
3. Fit the requested 1-NN, 3-NN, and NCC instances with the same training points.
4. Preserve the current NCC single-image demonstration while restructuring.
   Building the multi-image accuracy loop together follows this refactor.

- [x] Extract loading, preprocessing, and grouping functions.
- [x] Extract fitting and visualization functions; make `main()` orchestrate them.
- [x] Verify the existing NCC result and preprocessing survive the refactor.
- [x] Learning follow-up: build the accuracy loop and timing on a small trial.
- [x] Learning follow-up: address the existing k-NN sorting issue before evaluation.

Validation: ran the before/after scripts using the real dataset with a
noninteractive plotting backend. Printed output, plotted image arrays, and plot
titles matched. The math and classifier definitions are unchanged. Checked the
extracted preprocessing/grouping/fitting helpers on small arrays, including
both k values and shared training data. `git diff --check` passed. The existing
environment warning about unavailable 3D plotting remains unrelated to these
2D plots. No full accuracy run or optimized batch prediction was added.

#### Phase 3: First NCC evaluation

Design decisions approved by the user before implementation:
1. Implement the discussed `evaluate_classifier(classifier, X_test, y_test)`
   function with an ordinary prediction loop, correct-answer counter, and accuracy.
2. Return predictions and accuracy; reuse predictions for visualization.
3. Evaluate NCC on the test set first. Keep k-NN evaluation and timing for later.

- [x] Implement the evaluator and connect it to the NCC pipeline.
- [x] Check prediction alignment, accuracy, and invalid empty/mismatched inputs.
- [x] Run the full NCC test-set evaluation and record the observed result.

Validation: controlled predictions checked accuracy for partly correct, entirely
correct, and entirely incorrect answers; empty/mismatched inputs raise errors.
The full current pipeline completed on all 10,000 CIFAR-100 test images with
NCC accuracy 10.04% (1,004 correct). Verified the 16 plotted images have prediction
and actual-label titles using a noninteractive backend. `git diff --check` passed.
Timing remains pending from Phase 2; no k-NN test-set evaluation was run.

#### Phase 4: Evaluate 1-NN and 3-NN

Implementation decisions for the requested extension:
1. Reuse `evaluate_classifier` for each fitted model, preserving the ordinary
   one-image-at-a-time prediction selected by the user.
2. Fix the known sort failure by comparing distance and category only, retaining
   their existing ordering without comparing the image arrays on ties.
3. Print accuracy and sample count for each model; retain predictions for plots
   labeled with the model name. Finish evaluation before showing any plots.
4. Verify on a small common test subset against all 50,000 training images and
   measure its elapsed time before considering expensive full k-NN runs.

- [x] Fix distance ties and connect all three classifiers to evaluation.
- [x] Verify neighbor voting, accuracy, and model-labeled visualization.
- [x] Measure a small real-data trial and report its scope and runtime.

Validation: controlled cases cover equal-distance points within a category,
category ordering on distance ties, different 1-NN/3-NN votes, and accuracy.
Ran the real `main()` with a temporary test harness limiting only the test split
to its first 20 images; all 50,000 training examples were retained. Verified all
three model-labeled plots occur after evaluations and scores agree with the
returned predictions. `git diff --check` passed.

Observed trial results (not estimates of full-test accuracy):
- 1-NN: 3/20 correct (15%), 4.948 seconds for evaluation.
- 3-NN: 4/20 correct (20%), 4.920 seconds for evaluation.
- NCC: 1/20 correct (5%), 0.009 seconds for evaluation.

Linear extrapolation suggests roughly 41 minutes per full k-NN test pass on this
machine. Full k-NN runs were not launched. The normal script still evaluates all
10,000 test images; no subset limit was added to project code. Timing was measured
in the verification harness, not added to the evaluator's return contract.

#### Phase 5: Separate optimized classifiers and measured report results

Approved scope: the user requested parallel Greek report drafting, NEW optimized
classifiers alongside the original, measured CPU acceleration, and GPU work only
if CPU acceleration is insufficient for the deadline.

Implementation decisions recorded before coding:
1. Preserve `main.py`; add optimized CPU classes in `optimized.py` using batched
   Euclidean distance matrix calculations and deterministic nearest-neighbor
   selection. No PCA, reduced training set, or approximate search.
2. Use float64 internally for distance matrix arithmetic to reduce cancellation;
   describe this numerical difference and check predictions against the original.
3. Reuse the nearest three neighbors to produce both 1-NN and 3-NN predictions.
4. Add a reproducible benchmark runner with a fixed-seed common trial, controlled
   BLAS thread count, separate fit/prediction timings, and saved results.
5. Save full-test predictions, measured accuracy/timing, and example figures for
   the report. Never extrapolate a small-trial score as a full-test accuracy.
6. `report/` is owned by the explicitly requested report subagent. It uses the
   provided IEEE template and reports only verified measurements.

- [x] Implement separate optimized models and reproducible benchmark runner.
- [x] Verify toy cases, tie handling, batch boundaries, and reference agreement.
- [x] Measure CPU speed and run the full test set if practical.
- [x] Integrate actual results and figures into the Greek report.
- [x] Compile and inspect the final PDF and accompanying code/results.

Outcome: all four optimized-model tests passed. On a fixed-seed, stratified
100-image sample, all three optimized classifiers matched the originals on
100/100 predictions. Original versus optimized trial prediction times were
24.173/0.0916 s (1-NN), 24.031/0.1095 s (3-NN), and 0.0417/0.000589 s (NCC).
With eight CPU BLAS threads and batches of 128, the shared full 1-NN/3-NN search
took 10.319 s. Full-test scores: 1-NN 1,755/10,000 (17.55%); 3-NN 1,748/10,000
(17.48%); NCC 1,004/10,000 (10.04%). Full NCC agreement with the original was
10,000/10,000. Original full k-NN runs were not performed. GPU work was unnecessary.
Measurements, source hashes and predictions are saved in `../results/`.

The user subsequently changed the report language to English. See
`../report/AGENTS.md` Phase 2. The English PDF was compiled and visually inspected;
the previous Greek PDF and sources were preserved as a separate backup.
