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
- [x] Translate the report, included sections and figure labels.
- [x] Rebuild the English PDF and check measurements, references and layout.
- [x] Include the English report in the final deliverable archive.

Validation: the English PDF has seven pages. All pages were visually inspected;
appendix paragraph boundaries were fixed to prevent listings running outside the
page. Final log has no overfull boxes, missing glyphs or undefined references.
Reported accuracy counts match saved predictions, and all measured code hashes
still match the benchmark JSON. Dataset/reference attribution was checked against
the official CIFAR page. The PDF retains explicit timing-scope and numerical-
agreement limitations; no full original k-NN runtime is presented as measured.

Delivery: `../first_submission_english.zip` contains 21 files: the English PDF,
LaTeX sources, figures, both classifier implementations, tests, benchmark runner,
recorded measurements/predictions and reproduction instructions. ZIP integrity
and exact PDF/source bytes were checked. The Greek backup and dataset files are
excluded from this submission archive; the README explains dataset placement.

#### Phase 3: Automatic PDF preview in VS Code

Requested by the user: configure LaTeX editing so text changes update the PDF.
Use a dedicated report workspace with automatic saves, the existing XeLaTeX
build, and the LaTeX Workshop viewer beside the source. Keep these settings
local to this workspace.

- [x] Verify XeLaTeX and latexmk and install LaTeX Workshop.
- [x] Configure automatic saves, compilation, root detection and PDF preview.
- [x] Verify the configured build and open the report workspace.

Validation: the configured latexmk/XeLaTeX command rebuilt the seven-page PDF
successfully and generated SyncTeX data. VS Code accepted the command to open
the workspace and report source. Live preview refresh still requires opening
the LaTeX Workshop viewer in the editor (Ctrl+Alt+V).

#### Phase 4: Clarify k-NN tie handling

Requested clarification: separate equal-distance ordering from tied class votes,
illustrate the voting rule, and explain why image arrays are not sorting keys.
This is a wording change describing the existing classifier behavior.

- [x] Replace the ambiguous paragraph and verify the report build.

Validation: XeLaTeX rebuilt the seven-page PDF successfully; no overfull boxes
or undefined references were reported. The existing delivery ZIP remains an
earlier snapshot and does not include this wording change.

#### Phase 5: Explain the squared-distance identity

Approved by the user: add the scalar-expansion analogy, define the squared
norm and dot product, and connect the identity to batched distance computation.

- [x] Add the explanation beside the equation and verify the PDF build.

Validation: the seven-page PDF rebuilt successfully with the added explanation.
