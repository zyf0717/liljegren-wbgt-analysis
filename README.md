# liljegren-wbgt-analysis

Analysis helpers and notebooks for comparing WBGT implementations over
HeatStressBench ERA5 results.

## Environment

```powershell
conda env create -f environment.yml
conda activate liljegren-wbgt-analysis
```

The repository is not installed as a package. From a notebook under
`notebooks/`, import directly from `src`:

```python
import sys
from pathlib import Path

repository = Path("..").resolve()
sys.path.insert(0, str(repository / "src"))

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
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
