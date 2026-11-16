# Contributing

Thank you for helping improve First Sense / WForge. Contributions should make existing vision
workflows easier to use, reproduce, or extend.

## Development setup

Use Python 3.12 in an active virtual environment, then run from **First Sense/WForge**
(the package directory containing `pyproject.toml`, not the Git root):

```text
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src examples tests
python -m ruff format --check src examples tests
python -m mypy
python -m build
```

The default tests need no ML framework. Real training tests are opt-in and require
the backend environments documented in the [detection](docs/DEVELOPMENT.md),
[semi-supervised](docs/SEMI_SUPERVISED.md), and [DINOv3](docs/DINOV3.md) guides.
CI checks the core and package build; it does not certify backend training or GPU support.

## Change workflow

1. Describe the problem and expected behavior. Discuss substantial API changes in an issue first.
2. Check upstream documentation and compatibility before selecting dependencies.
3. Implement the smallest working workflow, including clear errors.
4. Run the relevant checks and record the actual commands and results.
5. Update examples and support status in the same change.

Keep changes focused and preserve source datasets and run artifacts. Follow the
[engineering conventions](docs/ENGINEERING.md) and [data/evaluation contract](docs/DATA_AND_EVALUATION.md).
New models must meet the [integration checklist](docs/INTEGRATIONS.md) and use the
[adapter interface](docs/MODEL_ADAPTERS.md). Upstream libraries own architectures,
losses, teacher updates, and training loops.

## Review checklist

- Is the public interface understandable without learning the backend framework?
- Does the change reuse an existing implementation?
- Are task/method/operation capabilities explicit?
- Are validation and test splits protected from training and tuning?
- Are dependencies isolated and upstream behavior covered by meaningful checks?
- Can a fresh user follow the documented example?
- Are limitations and skipped checks visible?

Use the setup and checks in [development instructions](docs/DEVELOPMENT.md). Real training tests are explicitly selected; keep ordinary core checks offline and independent of the backend.

## Reporting issues

Include a minimal reproduction, the model ID or profile path, OS/Python/backend
versions, and the relevant error output. Remove credentials and private data from
shared logs. Synthetic samples are preferred when the original dataset cannot be shared.

## Licensing

Contributions to WVision's original code are accepted under [Apache-2.0](LICENSE).
Keep upstream attribution and license references intact. Do not include model
weights, private datasets, environment folders, or generated training runs in pull requests.

Maintainers preparing the initial upload can follow the [GitHub publishing guide](docs/PUBLISHING.md).
