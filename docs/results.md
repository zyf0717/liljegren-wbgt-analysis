# HSB result files

`discover_results()` lists trusted HSB Parquet shards below each run directory:

```text
results/<result_date>/<run_id>/<implementation>/results.parquet
```

```python
from datetime import date
from pathlib import Path
from analysis.results import date_range, discover_results, iter_result_batches

files = discover_results(
    (Path("../HeatStressBench/.hsb/runs/example"),),
    date_range(date(2025, 1, 1), date(2025, 1, 7)),
    implementations=("heatstress", "meteor"),
)
for batch in iter_result_batches(files, ("row_id", "wbgt_status", "wbgt_c")):
    print(batch.num_rows)
```

Each `ResultFile` records its path, date, run ID, and implementation. Discovery
is deterministic and rejects overlapping run scopes, which would double-count
observations. Workflows discover their input inventory once and pass the selected
records to computation. HSB schemas and identifier formats are trusted.

`iter_result_batches()` reads only requested columns and yields bounded Arrow
record batches. Temperature calculations retain the workflow's status, missing
value, and support rules. Cross-implementation observation identity is checked
where daily common support is constructed.

Cellwise workflows also accept country caches produced by
`extract_countries_observations()`. Their row-group index limits repeated scans
to the requested date and implementation. Extraction records its resolved
boundary path so country maps use the same geometry and clipping bounds.
