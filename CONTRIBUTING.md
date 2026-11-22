# Contributing to Her & Him

Thanks for helping! Her & Him is a unified platform of AI-companion
personality catalogues — two complete catalogues (Her and Him) with a
deterministic, explainable affect engine and mood-driven multilingual
voice.

The most valuable contributions are new personas that pass the
catalogue quality bar, affect-engine improvements with tests, and
voice/language coverage.

## Development setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/gravion-ux/Her--Him.git
cd Her--Him
pip install -e ".[dev]"
```

The package uses a `src/` layout (`src/herhim/`).

## Running the tests

```bash
pytest -v --cov=herhim --cov-report=xml
```

Lint and type checks (CI runs all three, on Python 3.11, 3.12 and
3.13, before the test job):

```bash
ruff check src/herhim tests
mypy src/herhim
```

CI also runs a catalogue-quality job: every persona in both
catalogues is analysed and any critical/high finding fails the build.
Run the same analysis locally before proposing a persona (see the
`quality` job in `.github/workflows/ci.yml`).

## Adding a persona

1. Add the persona to the correct catalogue with complete metadata —
   incomplete entries are exactly what the quality gate rejects.
2. Keep persona content consistent with the existing entries' tone
   and structure; the affect engine consumes the declared traits, so
   trait values must be deliberate, not decorative.
3. Add or update tests covering the new entry, and make sure the
   catalogue-quality analysis reports no critical/high findings.

## Code style

- `ruff` clean and `mypy` clean — both are CI gates, not suggestions.
- Type hints throughout; comments explain *why* an affect threshold
  or weighting exists.
- Tests are plain pytest functions with descriptive names.

## Proposing a bigger change

Changes to the affect engine's model, the catalogue schema, or the
voice pipeline: **open an issue first**. New personas, tests and
documentation can go straight to a pull request.

## Pull request workflow

1. Fork and branch from `main`: `persona/<name>`, `feat/<short-name>`,
   `fix/<short-name>`, `docs/<short-name>`.
2. Run ruff, mypy and the full test suite locally.
3. Open the PR against `main` with: what changed, why, how you
   tested it.
4. CI must be green — lint, type check, tests on 3.11–3.13, the
   catalogue-quality job and the package build.
5. One logical change per PR.
