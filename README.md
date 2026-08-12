# liljegren-wbgt-analysis

Analysis helpers and notebooks for comparing WBGT implementations over
HeatStressBench ERA5 results.

## Environment

```powershell
uv sync
```

This creates a project-local `.venv`, installs the package in editable mode,
and synchronizes dependencies with the committed lockfile. Start Jupyter with:

```python
uv run jupyter lab
```

The package is importable without modifying `sys.path`:

```python
from liljegren_analysis import CampaignResults, pooled_sample_variance
```

## Documentation

- [Campaign result scanning and DuckDB relations](docs/results.md)
- [Daily-to-overall pairwise statistics](docs/pairwise-statistics.md)

The intended workflow is:

1. Use `CampaignResults.scan()` to construct filesystem-pruned lazy relations.
2. Use DuckDB SQL to compute compact daily pairwise sufficient statistics.
3. Materialize only the daily statistics as Pandas.
4. Orchestrate rollups in the notebook using the exported pooling calculations.

## Tests

```powershell
uv run python -m unittest discover -s tests -v
```
