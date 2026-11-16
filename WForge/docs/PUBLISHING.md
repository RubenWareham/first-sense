# First commit for First Sense

Use **`first-sense/` as the only Git root**. WForge is the buildable Python
component under it; do not initialize a nested repository inside WForge.

## What belongs in the commit

Include root README/mission/license/ignore rules, repository agent guidance,
GitHub workflows/templates, and WForge source, model profiles, dataset templates,
examples, tests, constraints, documentation, and notices. The root `.gitignore`
excludes local data (`Data/` and `data/`), environments, generated runs, model
weights, framework caches, credentials, and editor/agent state.

Empty future components are absent. Pose, classification, export/resume, and
runtime deployment remain explicitly planned in [milestones](MILESTONES.md).
Official pretrained DINOv3 verification is an outstanding model acceptance gate,
not a requirement to commit the currently documented implementation.

## Check the package locally

With your core development environment selected, run from **WForge/**:

```text
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src examples tests
python -m ruff format --check src examples tests
python -m mypy
python -m build
```

Default tests exclude real upstream training. Backend execution evidence is in
the [verification index](verification/README.md) and [milestones](MILESTONES.md).
Core CI runs on Windows/Linux with Python 3.12 and publishes no package. Its first
hosted result is only available after pushing; local checks do not claim hosted
CI or Linux/GPU backend certification.

## Review and make the first commit

Run from **first-sense/**, after reviewing the source tree:

```text
git init -b main
git status --short --untracked-files=all
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Initial First Sense training foundation"
```

Confirm the staged paths contain no datasets, weights, generated runs, or local
environments. Review before committing. This guide does not execute those steps.

## Connect GitHub

Create an empty repository with your chosen name and visibility. Leave generated
README/license/ignore options unchecked because they already exist locally.
Then replace the placeholder remote below with your repository URL:

```text
git remote add origin https://github.com/YOUR-USERNAME/first-sense.git
git push -u origin main
```

Git may require your configured author identity and GitHub authentication. Do not
put access tokens into the remote URL. Source installation and model-profile
examples use a repository checkout. The Python source distribution includes
WForge profiles/templates/docs, while the wheel contains the application and
workers. It does not bundle First Sense's root CI files or local datasets.

First Sense's original code uses Apache-2.0. The component retains its build-time
license/notices; upstream dependencies, model weights, and datasets keep their
own terms in [third-party notices](../THIRD_PARTY_NOTICES.md).
