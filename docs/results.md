# HSB result files

`HSBRunResults` resolves validated Parquet shards below:

```text
results/<result_date>/<run_id>/<implementation>/results.parquet
```

Use `files()` to select deterministic file records; it does not read observation
rows or expose a query-engine relation.

```python
from liljegren_analysis.results import HSBRunResults
from liljegren_analysis.io.results import iter_result_batches

files = HSBRunResults.from_hsb_run(hsb_run).files(
    start_date="2025-01-01",
    end_date="2025-01-07",
    implementations=["heatstress", "meteor"],
)
for batch in iter_result_batches(files, ("row_id", "wbgt_status", "wbgt_c")):
    ...
```

`files()` accepts one exact date (`result_date`), explicit dates (`dates`), or
an inclusive date range (`start_date`/`end_date`), plus singular or plural run
and implementation selectors. Integer run IDs are zero-padded to four digits.
Plural selectors are deduplicated and empty selectors are rejected.

Each `ResultFile` carries `path`, `result_date`, `run_id`, and
`implementation`. Source schemas must include `cycle_ordinal`, `row_id`,
`row_ordinal`, and qualified WBGT/Tg/Tnwb status/value columns. Any other
schema is rejected before scanning.

`iter_result_batches()` reads only its requested columns and yields bounded
Arrow record batches. Use it for row-scale processing; use Polars only for the
daily pairwise relational aggregation.
