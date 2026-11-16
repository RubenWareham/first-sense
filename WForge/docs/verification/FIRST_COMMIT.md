# First Sense first-commit preparation

Verified locally on **2026-11-16**, Windows, Python 3.12.3. This report describes
the First Sense checkout; the other dated reports in this directory retain the
earlier standalone WVision evidence.

## Resulting source tree

- The implemented `WForge` component and stable `wvision` API/CLI are retained.
- Empty `WVision/api`, `WVision/cli`, `WVision`, and `WRuntime` placeholders are
  removed. Future tasks/runtime remain in the roadmap, not empty source trees.
- Ten historical reports are grouped here with rebased links and an index.
- Root `.gitignore` owns the repository-wide policy; the duplicate component
  ignore file is removed. Repository agent guides remain available to contributors.
- Root Apache-2.0 license matches the component's license. Component license,
  NOTICE, third-party attribution, examples, profiles, constraints, and tests stay.
- Setup/contribution/publishing instructions distinguish the First Sense Git
  root from the WForge package root. The Ultralytics missing-extra message names
  the correct package root. Training algorithms and model capabilities are unchanged.
- Stale machine-local dataset details were removed from the general dataset
  guide. Original data, derived fixtures, weights, environments, and runs remain
  untouched and ignored; they are not source-publication artifacts.

## Local verification

All **148 core tests pass**, with 20 opt-in integration cases deselected. Ruff
checks/formatting pass for 75 Python files; mypy passes for 53 source modules.
Existing real backend evidence is retained in [milestones](../MILESTONES.md).
Heavyweight model/GPU training is not repeated for this publication cleanup.

A prospective-commit audit uses a **separate temporary Git directory** outside
First Sense. It does not initialize or stage the user's actual checkout. The
audited tree contains **153 source files**, including 38 Markdown documents and
16 Python examples. Checks cover all local links, Python example syntax, file
sizes/types, credential-shaped strings, and 13 ignore-policy cases. A temporary
index also passes `git diff --cached --check`. This is a focused source audit,
not a guarantee that every possible credential format is detected.

The policy excludes raw `Data/`, the prepared football label-budget copy under
`.wvision/`, local Python environments, weights/checkpoints, runs, caches, and
editor state. Neither public source files nor package archives include them.
Build/wheel checks verify packaged model factories/workers, profiles in the
source distribution, license metadata, and optional-framework-free core import.

The configured major CI references were checked against official releases:
[checkout v7](https://github.com/actions/checkout/releases/tag/v7.0.1) and
[setup-python v7](https://github.com/actions/setup-python/releases/tag/v7.1.0).
The root workflow runs from WForge and has read-only repository permissions.
**Hosted CI is not verified until a real push runs it.** Backend support is not
inferred from CI or imports.

## Remaining actions

No additional pose/classification/runtime/benchmark feature is required for the
first commit. Official licensed DINOv3 weight acceptance remains open and is
documented as such. The source is an alpha training integration, not a model
accuracy or state-of-the-art benchmark release.

The actual First Sense Git repository, first commit, remote repository choice,
and push are not performed by this preparation. Follow the
[publishing guide](../PUBLISHING.md) from the First Sense root after reviewing
the staged source list. The first hosted CI result will follow the push.
