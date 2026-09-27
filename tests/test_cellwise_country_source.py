from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from iso7243_fixtures import CELLS, NAMES, write_day, write_days

from analysis.cellwise.iso7243 import (
    analyze_cellwise_iso7243_classification,
    country_classification_label,
)
from analysis.cellwise.iso7243_summary import summarize_cellwise_iso7243_classification
from analysis.cellwise.joint_pipeline import (
    compute_joint_cellwise_temperature_quantiles,
)
from analysis.cellwise.spread import analyze_cellwise_quantile_spread
from analysis.comparison.distributions import WBGTHistogramSpec


def cache_scopes(path):
    reader = pq.ParquetFile(path)
    groups = {}
    columns = ["result_date", "implementation", "run_id"]
    for index in range(reader.num_row_groups):
        rows = reader.read_row_group(index, columns=columns).to_pylist()
        for key in {tuple(str(row[name]) for name in columns) for row in rows}:
            groups.setdefault(key, []).append(index)
    return [
        dict(zip(columns, key), row_groups=value)
        for key, value in sorted(groups.items())
    ]


def _country_cache(
    run: Path,
    output: Path,
    *,
    write_statistics: bool = True,
) -> Path:
    tables = []
    for path in sorted((run / "results").glob("*/*/*/results.parquet")):
        table = pq.read_table(path)
        count = table.num_rows
        result_date = date.fromisoformat(path.parents[2].name)
        table = table.append_column(
            "result_date", pa.array([result_date] * count, type=pa.date32())
        )
        table = table.append_column("run_id", pa.array([path.parents[1].name] * count))
        table = table.append_column(
            "implementation", pa.array([path.parent.name] * count)
        )
        table = table.append_column("cell_id", pc_slice(table["row_id"], 28))
        tables.append(table)
    combined = pa.concat_tables(tables)
    order = np.arange(combined.num_rows)
    np.random.default_rng(42).shuffle(order)
    output.parent.mkdir(parents=True)
    pq.write_table(
        combined.take(pa.array(order)),
        output,
        row_group_size=37,
        write_statistics=write_statistics,
    )
    days = sorted(
        {
            path.parents[2].name
            for path in (run / "results").glob("*/*/*/results.parquet")
        }
    )
    boundary = output.parent.parent / "external-boundary.geojson"
    boundary.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "ISO_A2": "XX",
                            "ISO_A3": "XXX",
                            "ADMIN": "Example",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]]
                            ],
                        },
                    }
                ],
            }
        )
    )
    manifest = {
        "schema_version": 4,
        "boundary_path": str(boundary),
        "country_iso_a2": "XX",
        "country_iso_a3": "XXX",
        "country_name": "Example",
        "start_date": days[0],
        "end_date": days[-1],
        "selected_cell_count": len(set(pc_slice(combined["row_id"], 28).to_pylist())),
        "parquet_file": output.name,
        "parquet_size_bytes": output.stat().st_size,
    }
    manifest["selection_bounds"] = [-1.0, -1.0, 1.0, 1.0]
    manifest["scopes"] = cache_scopes(output)
    output.with_name("manifest.json").write_text(json.dumps(manifest))
    return output


def pc_slice(values: pa.ChunkedArray, stop: int) -> pa.ChunkedArray:
    import pyarrow.compute as pc

    return pc.utf8_slice_codeunits(values, 0, stop)


@pytest.mark.parametrize(
    ("selection", "thresholds", "expected"),
    [
        ({"country_iso_a3": "JPN"}, [21, 25, 28, 31], "JPN MoE"),
        (
            {"country_iso_a3": "THA", "country_name": "Thailand"},
            [30, 32, 34],
            "Thailand workload limits",
        ),
        (
            {"country_iso_a3": "USA", "selection_bounds": [-125, 24, -67, 50]},
            [25, 26, 28, 30],
            "CONUS OSHA",
        ),
        ({"country_iso_a3": "XXX", "country_name": "Example"}, [25], "Example"),
    ],
)
def test_country_classification_regime_label(selection, thresholds, expected):
    assert (
        country_classification_label(
            {"country_selection": selection, "thresholds_c": thresholds}
        )
        == expected
    )


def _rewrite_cache(cache: Path, table: pa.Table) -> None:
    pq.write_table(table, cache, row_group_size=37)
    manifest_path = cache.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["parquet_size_bytes"] = cache.stat().st_size
    manifest["scopes"] = cache_scopes(cache)
    manifest_path.write_text(json.dumps(manifest))


def _write_valid_day(run: Path) -> None:
    cells = (
        "era5-global-n000000-e0000000",
        "era5-global-n002500-e0000000",
    )
    for implementation_index, implementation in enumerate(NAMES):
        rows = []
        for cell_index, cell in enumerate(cells):
            for hour in range(2):
                base = 20.0 + cell_index * 10 + hour + implementation_index / 10
                rows.append(
                    {
                        "cycle_ordinal": 1,
                        "row_id": f"{cell}-20250101{hour:02d}",
                        "row_ordinal": cell_index * 24 + hour,
                        "wbgt_status": "ok",
                        "wbgt_c": base,
                        "tg_status": "ok",
                        "tg_c": base + 10,
                        "tnwb_status": "ok",
                        "tnwb_c": base - 5,
                    }
                )
        table = pa.Table.from_pylist(rows)
        for column in ("cycle_ordinal", "row_ordinal"):
            table = table.set_column(
                table.schema.get_field_index(column),
                column,
                table[column].cast(pa.uint32()),
            )
        path = run / "results/2025-01-01/0001" / implementation / "results.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


def test_country_cache_matches_hsb_hourly_classification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    write_day(run)
    cache = _country_cache(run, tmp_path / "country" / "observations.parquet")
    common = dict(
        start_date="2025-01-01",
        end_date="2025-01-01",
        thresholds_c=(25, 26, 28, 30),
        date_workers=1,
        batch_size=11,
    )

    raw = analyze_cellwise_iso7243_classification(
        run, output_root=tmp_path / "raw", **common
    )
    country = analyze_cellwise_iso7243_classification(
        country_cache=cache, output_root=tmp_path / "country-output", **common
    )
    progress = capsys.readouterr().out
    assert "Example WBGT classification: 1 dates, 0 cached, 1 to scan" in progress
    assert "Example WBGT classification 2025-01-01:" in progress

    for name in (
        "cell_pair_differences",
        "cell_lookup",
    ):
        assert pq.read_table((raw / f"{name}.parquet")).equals(
            pq.read_table((country / f"{name}.parquet"))
        )
    raw_non_consensus = pq.read_table((raw / "cellwise_non_consensus.parquet"))
    country_non_consensus = pq.read_table((country / "cellwise_non_consensus.parquet"))
    for column in raw_non_consensus.column_names:
        np.testing.assert_equal(
            np.asarray(raw_non_consensus[column]),
            np.asarray(country_non_consensus[column]),
        )
    assert (raw / "classes.csv").read_text() == (country / "classes.csv").read_text()
    assert (country / "classes.csv").read_text().splitlines() == [
        '"class_id","class_label","threshold_definition"',
        '0,"< 25 C","wbgt_c < 25"',
        '1,"[25, 26) C","25 <= wbgt_c < 26"',
        '2,"[26, 28) C","26 <= wbgt_c < 28"',
        '3,"[28, 30) C","28 <= wbgt_c < 30"',
        '4,">= 30 C","wbgt_c >= 30"',
    ]
    metadata = json.loads((country / "metadata.json").read_text())
    assert metadata["country_selection"]["country_iso_a3"] == "XXX"
    assert metadata["thresholds_c"] == [25.0, 26.0, 28.0, 30.0]
    assert not (country / ".source-index.json").exists()
    weights = tmp_path / "population.parquet"
    pq.write_table(
        pa.table(
            {
                "source_cell_id": [*CELLS, "era5-global-n005000-e0000000"],
                "population": [10, 20, 30, 10_000],
            }
        ),
        weights,
    )
    summary = summarize_cellwise_iso7243_classification(
        country, population_weights_path=weights
    )
    assert summary.total_population == 60
    assert summary.region_label == "Example"


@pytest.mark.parametrize("write_statistics", [True, False])
def test_country_cache_matches_hsb_joint_quantiles(
    tmp_path: Path, write_statistics: bool
) -> None:
    run = tmp_path / "run"
    _write_valid_day(run)
    cache = _country_cache(
        run,
        tmp_path / "country" / "observations.parquet",
        write_statistics=write_statistics,
    )
    spec = WBGTHistogramSpec(-70.0, 70.0, 1.0, 99)
    common = dict(
        start_date="2025-01-01",
        end_date="2025-01-01",
        histogram_spec=spec,
        exact_validation_cells=2,
        date_workers=1,
        replay_workers=1,
    )

    raw_root = compute_joint_cellwise_temperature_quantiles(
        run, output_root=tmp_path / "raw", **common
    )
    country_root = compute_joint_cellwise_temperature_quantiles(
        country_cache=cache, output_root=tmp_path / "country-output", **common
    )

    assert pq.read_table(raw_root / "grid.parquet").equals(
        pq.read_table(country_root / "grid.parquet")
    )
    for implementation in NAMES:
        assert pq.read_table(
            raw_root / "implementations" / f"{implementation}.parquet"
        ).equals(
            pq.read_table(
                country_root / "implementations" / f"{implementation}.parquet"
            )
        )
    population_path = tmp_path / "population.parquet"
    pq.write_table(
        pa.table({"source_cell_id": CELLS, "population": [10, 20, 30]}),
        population_path,
    )
    spread = analyze_cellwise_quantile_spread(
        country_root,
        output_root=tmp_path / "spread",
        weighting="none",
        population_weights_path=population_path,
    )
    assert (spread / "cellwise_quantile_spread.parquet").is_file()


def test_country_cache_requires_exclusive_input_and_output_root(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_day(run)
    cache = _country_cache(run, tmp_path / "country" / "observations.parquet")
    common = dict(
        start_date="2025-01-01",
        end_date="2025-01-01",
        thresholds_c=(25, 26, 28, 30),
    )

    with pytest.raises(ValueError, match="exactly one"):
        analyze_cellwise_iso7243_classification(
            run, country_cache=cache, output_root=tmp_path / "out", **common
        )


def test_country_cache_process_execution_matches_serial(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_days(run, 2)
    cache = _country_cache(run, tmp_path / "country" / "observations.parquet")
    common = dict(
        country_cache=cache,
        start_date="2025-01-01",
        end_date="2025-01-02",
        thresholds_c=(21, 25, 28, 31),
        batch_size=11,
    )
    serial = analyze_cellwise_iso7243_classification(
        output_root=tmp_path / "serial", date_workers=1, **common
    )
    parallel = analyze_cellwise_iso7243_classification(
        output_root=tmp_path / "parallel", date_workers=2, **common
    )
    assert pq.read_table((serial / "cell_pair_differences.parquet")).equals(
        pq.read_table((parallel / "cell_pair_differences.parquet"))
    )
    assert (serial / "classes.csv").read_text() == (
        parallel / "classes.csv"
    ).read_text()
    assert '4,">= 31 C","wbgt_c >= 31"' in (serial / "classes.csv").read_text()


def test_country_cache_change_rejects_checkpoint_reuse(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_day(run)
    cache = _country_cache(run, tmp_path / "country" / "observations.parquet")
    common = dict(
        country_cache=cache,
        output_root=tmp_path / "out",
        start_date="2025-01-01",
        end_date="2025-01-01",
        thresholds_c=(25, 26, 28, 30),
    )
    analyze_cellwise_iso7243_classification(**common)
    table = pq.read_table(cache)
    values = table["wbgt_c"].to_pylist()
    values[0] = 19.5
    _rewrite_cache(
        cache,
        table.set_column(
            table.schema.get_field_index("wbgt_c"),
            "wbgt_c",
            pa.array(values, type=table.schema.field("wbgt_c").type),
        ),
    )
    with pytest.raises(ValueError, match="Incompatible ISO 7243 resume configuration"):
        analyze_cellwise_iso7243_classification(**common)


def test_country_cache_configured_scripts(tmp_path: Path, monkeypatch) -> None:
    import yaml

    from analysis import config
    from scripts import analyze_cellwise_iso7243_classification as iso_script
    from scripts import compute_joint_cellwise_temperature_quantiles as quantiles_script

    run = tmp_path / "run"
    _write_valid_day(run)
    cache = _country_cache(run, tmp_path / "country" / "observations.parquet")
    common = dict(
        country_cache=str(cache),
        start_date="2025-01-01",
        end_date="2025-01-01",
        date_workers=1,
    )
    settings = {
        "quantiles": dict(
            **common,
            output_root=str(tmp_path / "quantiles"),
            exact_validation_cells=0,
            replay_workers=1,
        ),
        "iso7243": dict(
            **common,
            output_root=str(tmp_path / "iso7243"),
            thresholds_c=[25, 26, 28, 30],
        ),
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    quantiles_script.main()
    iso_script.main()
    for workflow in ("quantiles", "iso7243"):
        assert (tmp_path / workflow / "metadata.json").is_file()
        assert (tmp_path / workflow / "settings.json").is_file()
