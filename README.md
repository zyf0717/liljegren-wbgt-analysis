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

from liljegren_analysis import CampaignResults
```

## Results relation

`CampaignResults` maps this filesystem structure to a lazy DuckDB relation:

```text
results/<result_date>/<run_id>/<implementation>/results.parquet
```

The relation contains:

```text
result_date, run_id, implementation,
cycle_ordinal, row_id, row_ordinal, status, value_c
```

Scan one scope:

```python
campaign = (
    repository
    / ".hsb"
    / "campaigns"
    / "era5-2025-all-six-m2-w20-v8"
)

results = CampaignResults.from_campaign(campaign)
relation = results.scan(result_date="2025-01-01", run_id=1)
```

Omitted scope arguments become filesystem globs. `scan()` is lazy; DuckDB does
not execute the query until a terminal operation is requested.

## Combine result directories

Relations must use the same DuckDB connection before they can be unioned:

```python
import duckdb

runs = [
    repository / ".hsb" / "campaigns" / "era5-2025-all-six-m2-w20-v8",
    repository
    / ".hsb"
    / "runs"
    / "era5-2025-meteor-m4-w18"
    / "wbgt-comparison-20260810-231240",
]

con = duckdb.connect()
relations = [
    CampaignResults.from_campaign(run, connection=con).scan(
        result_date="2025-01-01"
    )
    for run in runs
]

all_results = relations[0]
for relation in relations[1:]:
    all_results = all_results.union(relation)
```

Query the combined relation using DuckDB SQL:

```python
summary = all_results.query(
    "results",
    """
    SELECT
        implementation,
        count(*) AS observations,
        avg(value_c) AS mean_c
    FROM results
    WHERE status = 'ok'
    GROUP BY implementation
    ORDER BY implementation
    """,
)

summary.df()
```

Keep filtering, projection, joins, and aggregation in DuckDB. Convert to Pandas
only when the result is bounded; materializing a full day across all seven
implementations produces about 59 million rows.

Close the shared connection when it is no longer needed:

```python
con.close()
```

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
