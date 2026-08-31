# Contributing

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) with a concise,
imperative subject:

```text
<type>(<optional scope>): <summary>
```

Allowed types:

- `feat` — new user-facing capability
- `fix` — bug correction
- `docs` — documentation only
- `refactor` — behavior-preserving code restructuring
- `perf` — performance improvement
- `test` — test-only changes
- `build` — packaging or dependency changes
- `ci` — continuous-integration changes
- `chore` — maintenance not covered above
- `style` — formatting-only change
- `revert` — reverses an earlier commit

## Branches

Use one of these prefixes followed by a short kebab-case description:

- `feat/`
- `fix/`
- `release/`
- `hotfix/`
- `docs/`
- `refactor/`
- `test/`
- `chore/`

For example: `feat/cellwise-wbgt-quantiles`.

## Pull requests

Prefer a pull request for any major change or coherent set of major commits.
Keep the PR focused, describe the behavioral impact, and include the relevant
validation results before requesting review.
