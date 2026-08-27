# Campaign result relations

`CampaignResults` maps this filesystem structure to a lazy DuckDB relation:

```text
results/<result_date>/<run_id>/<implementation>/results.parquet
```

From a notebook under `notebooks/`:

```python
from pathlib import Path

from liljegren_analysis import CampaignResults

repository = Path("..").resolve()
```

Open a campaign and construct a scoped relation:

```python
campaign = (
    repository
    / ".hsb"
    / "runs"
    / "wbgt-comparison-20260823-181813"
)

results = CampaignResults.from_campaign(campaign)
relation = results.scan(result_date="2025-01-01", run_id=1)
```

## `scan()` API

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
    quantity=None,
    quantities=None,
    include_filename=False,
)
```

| Argument | Scope | Example | Behavior when omitted |
|---|---|---|---|
| `result_date` | One date directory | `"2025-01-01"` | All dates unless another date scope is supplied |
| `dates` | Non-contiguous dates | `["2025-01-01", "2025-01-07"]` | No explicit date set |
| `start_date` | Inclusive range start | `"2025-01-01"` | No range |
| `end_date` | Inclusive range end | `"2025-01-07"` | No range |
| `run_id` | One run/shard | `1` or `"0001"` | All runs |
| `run_ids` | Multiple runs/shards | `[1, 2, 8]` | No explicit run set |
| `implementation` | One implementation | `"meteor"` | All implementations |
| `implementations` | Multiple implementations | `["heatstress", "meteor"]` | No explicit implementation set |
| `quantity` | One temperature quantity | `"tg"` | All quantities unless `quantities` is supplied |
| `quantities` | Multiple quantities | `["wbgt", "tnwb"]` | All three quantities |
| `include_filename` | Source-file provenance | `True` | Filename excluded |

Date arguments accept ISO strings or `datetime.date` values. Use one of an
exact `result_date`, explicit `dates`, or both range arguments. Singular and
plural run, implementation, and quantity forms are mutually exclusive. Integer
run IDs are padded to four digits. Duplicate plural values are removed; empty
selections are rejected. Quantities use canonical order `wbgt`, `tg`, `tnwb`.

The scanner generates exact path cross-products for scoped values. This is
important because the directory structure is not Hive-style: applying a SQL
date filter to an unrestricted yearly relation does not provide equivalent
file pruning.

Common scopes:

```python
results.scan(result_date="2025-01-01")
results.scan(result_date="2025-01-01", run_id=1)
results.scan(implementation="heatstress")
results.scan(result_date="2025-01-01", quantity="tg")

results.scan(
    dates=["2025-01-01", "2025-01-07"],
    run_ids=[1, 2, 8],
    implementations=["heatstress", "meteor"],
)

results.scan(
    start_date="2025-01-01",
    end_date="2025-01-07",
    run_id=1,
    implementation="meteor",
)
```

Omitting all scopes targets the complete result tree. Keep that relation lazy
and query it carefully.

## Relation schema

| Column | Source |
|---|---|
| `result_date` | Date directory, cast to DuckDB `DATE` |
| `run_id` | Run/shard directory as text, preserving leading zeros |
| `implementation` | Implementation directory |
| `cycle_ordinal` | Parquet result column |
| `row_id` | Parquet result identity |
| `row_ordinal` | Parquet result order |
| `wbgt_status`, `wbgt_c` | WBGT status and value in degrees Celsius |
| `tg_status`, `tg_c` | Globe-temperature status and value in degrees Celsius |
| `tnwb_status`, `tnwb_c` | Natural wet-bulb status and value in degrees Celsius |
| `filename` | Source path, only with `include_filename=True` |

All source files must contain the three identity columns and all six qualified
quantity columns with string statuses and numeric values. Extra columns are
ignored. Legacy `status`/`value_c` files are rejected. A quantity selector
controls the output projection and Parquet column pruning, but schema validation
still requires the complete qualified source schema.

The return value is a `duckdb.DuckDBPyRelation`. `.query()`, `.filter()`,
`.project()`, `.aggregate()`, `.order()`, and `.union()` remain lazy. `.df()`,
`.arrow()`, `.fetchall()`, and `.fetchone()` execute work.

## Combining result directories

Relations must use the same DuckDB connection before they can be unioned:

```python
import duckdb

runs = [
    repository
    / ".hsb"
    / "runs"
    / "wbgt-comparison-20260823-181813",
]

con = duckdb.connect()
relations = [
    CampaignResults.from_campaign(run, connection=con).scan(
        start_date="2025-01-01",
        end_date="2025-01-07",
    )
    for run in runs
]

all_results = relations[0]
for relation in relations[1:]:
    all_results = all_results.union(relation)
```

## Query-local relation name

`DuckDBPyRelation.query(relation_name, sql)` binds the relation to a
query-local virtual table name. The name is arbitrary but must match the SQL
`FROM` clause. It is unrelated to the filesystem `results` directory and does
not create a persistent view. The returned relation remains lazy.

```python
summary = all_results.query(
    "results",  # Query-local name for all_results.
    """
    SELECT implementation, count(*) AS observations, avg(tg_c) AS mean_tg_c
    FROM results
    WHERE tg_status = 'ok' AND tg_c IS NOT NULL
    GROUP BY implementation
    ORDER BY implementation
    """,
)
```

Keep filtering, joins, and aggregation in DuckDB. Convert to Pandas only after
the result is bounded. Close the shared connection when it is no longer needed:

```python
con.close()
```
