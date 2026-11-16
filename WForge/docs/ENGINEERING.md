# Engineering rules

## Code structure

- Use Python with a `src/` package layout and `pyproject.toml`.
- Select and document the Python version range after checking the first backend's compatibility. Do not guess CUDA, PyTorch, or backend pins.
- Use type hints for public functions, requests, results, and adapter boundaries.
- Keep functions focused and composition explicit. Prefer a direct function or protocol to inheritance unless shared behavior justifies it.
- Avoid wildcard imports, circular imports, mutable global configuration, import-time work, and broad exception swallowing.
- Explain non-obvious constraints in comments and docstrings. Do not narrate obvious statements.
- Use `pathlib` and portable paths. Core tests must work on Windows and Linux.
- Keep the public namespace small; treat incompatible public changes as documented API changes.

## Dependencies

Keep CLI/configuration requirements small. Put each heavy backend in an optional dependency group and use lazy imports with actionable installation errors. Pin verified backend environments through reproducible constraints or lock files, including accelerator-specific installation instructions. Record the actual tested environment; a lock file does not prove GPU compatibility.

Avoid copying or vendoring upstream source. If a patch is unavoidable, record its source, license, reason, and removal plan. Do not automatically fetch executable code or weights during import or a basic help command.

## Errors and logs

Report what failed, which configuration field or artifact caused it, and what the user can change. Preserve exception causes when wrapping upstream failures. Do not silently ignore unknown fields, missing labels, missing dependencies, unsupported devices, or checkpoint mismatches.

Use structured run metadata and ordinary Python logging. Keep CLI summaries short, make verbosity configurable, and redact tokens and sensitive URL parameters from saved configuration/logs. Library calls must not terminate the host process.

## Verification

Introduce formatting/linting with Ruff, static typing with one documented checker, and tests with pytest in M1. Configure and verify the tools before documenting runnable commands.

Test behavior and risks:

- Configuration precedence, invalid fields, unsupported combinations, and missing extras.
- Split leakage, class mapping, coordinate conversion, and missing-vs-negative annotations.
- Artifact serialization, failed-run status, and non-overwriting output paths.
- The first adapter's real train/val/test/predict path on a tiny owned or redistributable fixture.

Keep core checks CPU-only and offline. Mocks may verify error paths but cannot establish backend support. Mark real backend/GPU checks separately and run them in the pinned environment before publishing support claims. Document unavailable checks and why they were skipped. Use numerical tolerances; do not require bitwise equality for nondeterministic GPU operations.

Avoid expensive training in every pull request. Use a tiny integration smoke test, with larger benchmark jobs scheduled or explicitly run. A tiny smoke test establishes execution, not model quality.

## Documentation maintenance

Every working public workflow needs a minimal example, expected outputs, supported environment, and limitations. Keep planned examples labeled. Update the capability table and milestones only with evidence. New adapters must satisfy [integration acceptance](INTEGRATIONS.md).
