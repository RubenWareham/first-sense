# First-publication preparation verification

Verified on **2026-11-08**, Windows CPU, Python 3.12.3. This slice prepares the source
repository for its first GitHub upload. It adds no model or training method.

## Public-facing files

The README now covers source installation, a synthetic detection quick start,
Python and CLI use, dataset setup, learning from unlabeled images, outputs,
supported environments, and licensing. Contribution guidance addresses human
contributors directly. Local agent guidance is excluded from Git and source
archives, and is not referenced by public usage instructions.

Apache-2.0 was selected for WVision's original code. The canonical Apache license,
`NOTICE`, package SPDX metadata, and third-party notices are present. Optional
upstream packages and model weights retain their terms.

GitHub Actions is configured to run core tests, lint, formatting, typing, and
package builds on Windows and Linux using Python 3.12. The workflow installs no
ML backend and publishes no package. Hosted CI has not been run yet; Linux model
execution and GPU support are not claimed.

## Fresh environment and checks

```powershell
.wvision/clean-env/Scripts/python -m venv .wvision/github-core
.wvision/github-core/Scripts/python -m pip install -e ".[dev]"
.wvision/github-core/Scripts/python -m pytest -q
.wvision/github-core/Scripts/python -m ruff check src examples tests
.wvision/github-core/Scripts/python -m ruff format --check src examples tests
.wvision/github-core/Scripts/python -m mypy
.wvision/github-core/Scripts/python -m pip check
.wvision/github-core/Scripts/python -m build --outdir .wvision/github-dist
.wvision/clean-env/Scripts/python -m pytest tests/test_integration.py -m integration -q --basetemp=.wvision/github-yolo-verified
```

Fresh source installation succeeds without torch, Ultralytics, MMDetection, or
Transformers. All **119 core tests pass**, with eleven integration cases
deselected, in **20.21 seconds**. The fresh environment resolved PyYAML 6.0.3,
Pillow 11.3.0, pytest 9.2.0, Ruff 0.17.4, Mypy 2.4.1, and build 1.6.1. Core-only
type checking initially exposed missing optional torch types; torch now follows
the same isolated typing policy as other optional frameworks. Mypy passes all 26
source modules in both core-only and YOLO environments. Lint/formatting pass for
44 Python files, and the fresh environment's dependency check passes.

Initializing Git also exposed an unborn-repository provenance edge case:
`git rev-parse HEAD` can write `HEAD` while returning a failure status. Environment
capture now checks that status and correctly records `git_commit: null` until a
commit exists. A core regression test and a direct check against this repository
verify the behavior.

Both real YOLO Python/CLI integration cases pass in **42.59 seconds**, with thirteen
existing Matplotlib/Pyparsing deprecation warnings. They exercise train, val,
test, predict, checkpoint fine-tuning, and reload. Other backend training is
unchanged and was not rerun for this publication-only slice; see
[cleanup verification](CLEANUP.md) for the preceding regression evidence.

## Distribution and upload boundary

Source and wheel builds succeed using isolated setuptools 84.2.1. The wheel
contains all three standalone backend workers and Apache license/notice metadata.
The source distribution includes all thirteen model profiles, four dataset
example/templates, examples, documentation, and GitHub configuration. The wheel
does not install repository-relative model/dataset catalogs; README examples use
a source checkout.

Local Git was initialized on `main`. No files have been committed or pushed, and
no remote has been configured. Candidate Git files and distribution contents are
checked for excluded local instructions, environments, generated runs, weights,
caches, and credentials. The [publishing guide](../PUBLISHING.md) contains the remaining
manual steps, including the user's remote URL and authentication.

The independently installed wheel imports from its environment's `site-packages`,
starts the CLI, lists all four model IDs, and exposes Apache-2.0 metadata with no
ML packages installed. The source archive contains 114 files; the wheel contains
34. Source catalogs, all backend workers, and the license/notice files were checked.

The Git upload manifest contains 106 public files. All 29 public Markdown
documents have valid local links; twelve Python blocks parse, 36 CLI examples
parse, and nineteen catalog selections validate. README and workflow configuration
were checked. A separate `dist/WVision-source.zip` contains only the audited Git
upload files under a `WVision/` root, including ignore rules and GitHub configuration.
It excludes Git metadata and local instructions, environments, weights, and runs.

Logs and package build artifacts are under ignored `.wvision/github-*` paths.
