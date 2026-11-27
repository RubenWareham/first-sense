# AGENTS.md

Guidance for coding agents working in the First Sense repository.

First Sense currently implements WForge training/evaluation through the `wvision` Python API and CLI. Detection, supervised 2D pose, Soft Teacher, SimSiam, and DINOv3/DETR have documented support boundaries. Classification and local/edge runtime work remain roadmap items; do not add empty component folders or advertise C++/TensorRT/DeepStream support before implementation. Read `WForge/AGENTS.md` for model/data rules.

## Core principles

**Less is more. Prefer the simplest complete solution. Delete > Replace > Add.**

1. **Fix the cause at its owner.** Change the code responsible for the behavior. Do not hide broken logic behind flags, silent fallbacks, broad exception handlers, or duplicate paths.
2. **Search and reuse first.** Read the relevant code and search the repository before adding helpers, components, dependencies, or workflows. Extend existing implementations. A few clear lines are better than an abstraction used once.
3. **Modify before creating.** Remove obsolete code and in-scope duplication. Add a file or abstraction only when it has a clear responsibility that does not fit an existing owner. Do not build for hypothetical requirements.
4. **Keep scope small.** Complete the requested change without unrelated refactors, speculative options, compatibility layers, or unnecessary configuration. Follow existing conventions.
5. **Validate real behavior.** Run the relevant existing checks and focused validation. Add tests when they cover meaningful new behavior or a regression gap; avoid tests that merely repeat the implementation. State what could not be verified.

## First Sense boundaries

- Keep model-specific details in their adapters. Reuse shared training, evaluation, and deployment workflows instead of copying entire pipelines.
- Implement only supported capabilities. Unsupported model, training, export, or device combinations must fail clearly rather than silently change behavior.
- Keep datasets and annotations intact. Record the configuration, dataset split, and model checkpoint needed to reproduce training and comparisons.
- Make performance claims from measurements. Report hardware, precision, input size, and batch size; distinguish model latency from full-pipeline latency.
- Keep user-facing interfaces simple. Expose settings that serve a real use case and use sensible defaults.

## Working rules

- Read the relevant README, configuration, and nearby code before editing. Treat the repository as the source of truth; do not invent commands, paths, APIs, or supported features.
- Preserve unrelated work. Do not overwrite changes you did not make, commit secrets or datasets, force-push, or push directly to the default branch.
- Before finishing, review the diff for unnecessary additions, dead code, duplication, and unintended behavior changes.
- Summarize what changed, how it was validated, and any remaining limitations. Clearly label assumptions and uncertainty.

**Review gate:** Could this change be solved more clearly by deleting or modifying existing code? If yes, simplify it before adding more.
