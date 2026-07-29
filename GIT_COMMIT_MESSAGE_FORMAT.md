# Git Commit Message Format

All commits in this repository must use a detailed, bracketed format. The
message must explain what changed, why it changed, how it was validated, and
how the change can be safely rolled back.

## Required format

```text
[type][scope] concise subject

[Summary]
Describe the user-visible or repository-level outcome in complete sentences.

[Details]
Explain the implementation, affected files, important design decisions, and
any behavior that future maintainers need to understand.

[Safety]
Describe compatibility considerations, data or API risks, and why the change
is safe to apply.

[Validation]
List the tests, checks, commands, or manual verification performed. If a check
was not run, state that explicitly and explain why.

[Comprehension]
Provide context that helps a reviewer understand the change, its assumptions,
its limitations, and any follow-up considerations.

[Rollback]
Explain how to revert the commit and identify any additional cleanup required.
```

## Subject line

The subject line must follow this structure:

```text
[type][scope] concise subject
```

Use an imperative, specific subject with no trailing period. Keep it concise
enough to understand from `git log --oneline`, while leaving implementation
details for the body.

Common types include:

- `feat` — add a user-facing capability
- `fix` — correct incorrect or broken behavior
- `refactor` — change structure without changing intended behavior
- `perf` — improve runtime, memory, or resource usage
- `test` — add or change tests
- `docs` — change documentation only
- `build` — change dependencies or build configuration
- `ci` — change continuous integration or automation
- `chore` — maintenance that does not fit another type

The scope should identify the affected area, such as `profiler`, `backbone`,
`docs`, `training`, or `dependencies`.

## Body requirements

Every section is required. Use complete sentences and concrete details rather
than generic statements such as “updated code” or “fixed issue.” Include file
names, command names, relevant configuration, and measurable results when they
make the change easier to review.

The `[Validation]` section must distinguish between checks that passed, checks
that failed, and checks that were not run. Do not claim that a test passed if it
was not executed.

The `[Rollback]` section should normally include the commit-specific command:

```bash
git revert <commit-sha>
```

If reverting is not sufficient because of migrations, generated files, or
external state, document the additional rollback steps.

## Complete example

```text
[feat][profiler] support selecting exported models

[Summary]
Allow users to profile any torch.nn.Module exported by models.py by passing
the class name through --model.

[Details]
Added dynamic model discovery and default-constructor instantiation to
profile_backbone.py. Added --list-models for discovering available model
classes and retained YOLOv11Backbone as the default. Per-stage reporting is
performed when the selected model exposes a sequential model attribute.

[Safety]
The change is additive and does not alter model definitions or profiling
formulas. Existing invocations continue to select YOLOv11Backbone by default.
Invalid model names fail before inference with an explicit list of available
models.

[Validation]
Ran `uv run python profile_backbone.py --list-models` and verified that the
exported model classes were listed. Ran profiling for both available models
and ran `uv run python -m py_compile profile_backbone.py`.

[Comprehension]
Models must be exported from models.py and must be constructible without
arguments. MAC counting covers convolution operations and PSA attention matrix
multiplications; elementwise operations remain outside the reported total.

[Rollback]
Run `git revert <commit-sha>`. No migrations, generated artifacts, or external
state require cleanup.
```
