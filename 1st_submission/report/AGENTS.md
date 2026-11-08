# Classifier report

## Package Instructions

- Use the supplied IEEE journal template and XeLaTeX; the current report language is English.
- Keep the report grounded in the assignment, actual source code, and measured results.
- Never present 20-image smoke trials as full-test accuracy or estimates as measurements.
- Keep personal names out of all report artifacts.
- Build from this directory with `latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex`.

---

## Progress Log

### Branch: master

#### Phase 1: Greek report with original and optimized implementations

Approved by the user: prepare the Greek report in parallel, use the parent-directory
IEEE template, cover dataset/preprocessing/algorithms/code/evaluation/discussion,
include explanatory visuals and both implementation versions.

- [x] Inspect the supplied template and prepare Greek typography.
- [x] Write the dataset, preprocessing, algorithms, and code sections.
- [x] Add measured results and actual prediction examples supplied by the primary task.
- [x] Compile and inspect the final PDF.

The report subagent produced the initial four-page draft and then reached its
usage limit. The primary task continues results integration and PDF verification.
The original code is preserved; optimized code lives in `../src/optimized.py`.

#### Phase 2: English report

The user changed the required report language to English. Preserve the Greek
version in `greek-version.zip`; translate prose, diagrams, captions and generated
figure labels without changing the experiments or measured results.

- [x] Preserve the Greek PDF and sources.
- [ ] Translate the report, included sections and figure labels.
- [ ] Rebuild the English PDF and check measurements, references and layout.
- [ ] Include the English report in the final deliverable archive.
