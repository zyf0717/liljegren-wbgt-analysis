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

### `scan()` API

```python
results.scan(
    result_date=None,
    dates=None,
    start_date=None,
    end_date=None,
    run_id=None,
    run_ids=None,
    implementation=None,
    implementations=None,
    include_filename=False,
)
```

| Argument | Scope | Example | Behavior when omitted |
|---|---|---|---|
| `result_date` | One date directory | `"2025-01-01"` | All dates, unless a range is supplied |
| `dates` | Non-contiguous date directories | `["2025-01-01", "2025-01-07"]` | No explicit date set |
| `start_date` | First date in an inclusive range | `"2025-01-01"` | No range |
| `end_date` | Last date in an inclusive range | `"2025-01-07"` | No range |
| `run_id` | One run/shard directory | `1` or `"0001"` | All runs |
| `run_ids` | Multiple run/shard directories | `[1, 2, 8]` | No explicit run set |
| `implementation` | One implementation directory | `"meteor"` | All implementations |
| `implementations` | Multiple implementation directories | `["heatstress", "meteor"]` | No explicit implementation set |
| `include_filename` | Source-file provenance column | `True` | Filename excluded |

Date arguments accept ISO strings or `datetime.date` values. Use one of an
exact `result_date`, explicit `dates`, or both inclusive range arguments.
Singular and plural run/implementation forms are mutually exclusive. Integer
run IDs are padded to four digits; strings are used unchanged. Duplicate items
in plural selections are removed, and empty selections are rejected.

Common scopes:

```python
# Entire result tree. Keep this lazy and query it carefully.
results.scan()

# All runs and implementations for one date.
results.scan(result_date="2025-01-01")

# One run, all implementations, on one date.
results.scan(result_date="2025-01-01", run_id=1)

# One implementation over all dates and runs.
results.scan(implementation="heatstress")

# Selected non-contiguous dates, runs, and implementations.
results.scan(
    dates=["2025-01-01", "2025-01-07"],
    run_ids=[1, 2, 8],
    implementations=["heatstress", "meteor"],
)

# One implementation and run over an inclusive date range.
results.scan(
    start_date="2025-01-01",
    end_date="2025-01-07",
    run_id=1,
    implementation="meteor",
)
```

Every scan returns these columns:

| Column | Source |
|---|---|
| `result_date` | Date directory, cast to DuckDB `DATE` |
| `run_id` | Run/shard directory, retained as text to preserve leading zeros |
| `implementation` | Implementation directory |
| `cycle_ordinal` | Parquet result column |
| `row_id` | Parquet result identity column |
| `row_ordinal` | Parquet result-order column |
| `status` | Parquet result status |
| `value_c` | Parquet WBGT result in degrees Celsius |
| `filename` | Source Parquet path; present only with `include_filename=True` |

The return value is a `duckdb.DuckDBPyRelation`. Operations such as `.query()`,
`.filter()`, `.project()`, `.aggregate()`, `.order()`, and `.union()` remain
lazy. Terminal methods including `.df()`, `.arrow()`, `.fetchall()`, and
`.fetchone()` execute work. Relations can only be unioned when they were built
from the same DuckDB connection.

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
