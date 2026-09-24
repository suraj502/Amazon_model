# Business Entity Resolution

A production-style scaffold for matching business records across Source 1, Source 2, and Source 3 when no shared identifier exists. Source 1 is the reference source, matches are ID-based, and the evaluation contract is entity-level macro F0.5. Countries are treated as open-set data: no country list or class mapping is hardcoded.

## Project Structure

- `config/`: the single source of truth for paths, schema, strategies, model settings, and submission columns.
- `data/raw/` and `data/processed/`: gitignored input and intermediate data locations.
- `src/`: package modules organized by pipeline responsibility.
- `scripts/`: command-line entry points for each stage and the end-to-end runner.
- `tests/`: shared synthetic fixtures, unit contracts, and integration contracts.
- `notebooks/`, `reports/`, and `output/`: working artifacts and generated results.

## Team Ownership

The four pipeline stages map cleanly to separate team folders:

1. **Normalization:** `src/normalization/` cleans names and addresses and profiles source data.
2. **Blocking:** `src/blocking/` creates and unions candidate pairs.
3. **Features and model:** `src/features/` and `src/model/` build candidate features, train, and predict.
4. **Evaluation and submission:** `src/evaluation/` and `src/submission/` score, select thresholds, validate, and serialize results.

Shared configuration and I/O contracts live in `src/utils/`; changes there should be coordinated across the team.

## Setup

```powershell
cd business-entity-resolution
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place competition TSV files under `data/raw/` using the paths in `config/config.yaml`.

## Run the Pipeline

The end-to-end entry point calls normalization, blocking, feature construction, training, and evaluation in order:

```powershell
python scripts/run_pipeline.py
```

The stage modules are deliberately stubs. Implement each stage against the configured ID and column contracts before running the full pipeline.

## Tests

Run the complete test suite with:

```powershell
pytest -q
```

Tests that require real matching, blocking, or ML behavior are marked `pytest.mark.skip(reason="not implemented")` until the corresponding stage is implemented.
