"""Expand one ERA5 run into the existing workflow argument dictionaries."""

from pathlib import Path

from .config import REFERENCE_IMPLEMENTATION, implementation_names


def expand_run_config(config: dict) -> dict:
    run = config["run"]
    required = {
        "hsb_run_path", "start_date", "end_date", "implementations", "population_raster",
    }
    optional = {"output_root", "boundary_cache", "countries"}
    if not isinstance(run, dict):
        raise ValueError("run must be a mapping")
    if missing := required - run.keys():
        raise ValueError(f"Missing run settings: {sorted(missing)}")
    if unknown := run.keys() - required - optional:
        raise ValueError(f"Unknown run settings: {sorted(unknown)}")
    source = Path(run["hsb_run_path"])
    root = Path(run.get("output_root", Path("results") / source.name))
    dates = {key: run[key] for key in ("start_date", "end_date")}
    targets = list(implementation_names(run["implementations"]))
    common = {**dates, "implementations": targets}
    global_source = {**common, "hsb_run_paths": [str(source)]}
    countries = run.get("countries", {})
    if not isinstance(countries, dict) or any(
        not isinstance(code, str)
        or len(code) != 3
        or not code.isalpha()
        or not code.isupper()
        for code in countries
    ):
        raise ValueError(
            "run.countries must map uppercase ISO3 codes to thresholds in Celsius"
        )
    sections = dict(config)

    def add(name, **defaults):
        overrides = config.get(name, {})
        if not isinstance(overrides, dict):
            raise ValueError(f"{name} must be a mapping")
        sections[name] = {**defaults, **overrides}
        return sections[name]

    population = add(
        "population", input_raster=run["population_raster"],
        output_root=root / "population",
    )
    population_path = Path(population["output_root"]) / "population.parquet"
    daily = add(
        "daily_pairwise", **{**global_source, "implementations": sorted(targets)},
        output_root=root / "pairwise/daily",
    )
    yearly = add(
        "yearly_pairwise", **dates, input_root=daily["output_root"],
        output_root=root / "pairwise/yearly",
    )
    yearly.setdefault("export_implementations", [
        name for name in targets
        if name != yearly.get("reference_implementation", REFERENCE_IMPLEMENTATION)
    ])
    quantiles = add("quantiles", **global_source, output_root=root / "cellwise/quantiles")
    spread = add(
        "quantile_spread", input_root=quantiles["output_root"],
        output_root=root / "cellwise/spread", quantities=["wbgt", "tnwb", "tg"],
        population_weights_path=population_path,
    )
    add("plot_quantile_spread", output_root=spread["output_root"])
    iso = add("iso7243", **global_source, output_root=root / "cellwise/iso7243")
    add(
        "plot_iso7243", input_root=iso["output_root"],
        population_weights_path=population_path,
    )
    add("plot_non_consensus", input_root=iso["output_root"])
    add(
        "plot_non_consensus_stack", input_root=iso["output_root"],
        population_weights_path=population_path,
    )

    steps = {name: [name] for name in (
        "population", "daily_pairwise", "yearly_pairwise", "iso7243", "plot_iso7243",
        "plot_non_consensus", "plot_non_consensus_stack", "quantiles", "quantile_spread",
        "plot_quantile_spread",
    )}
    steps.update(country_cells=[], extract_country=[], country_classification=[])
    if countries:
        boundary = run.get("boundary_cache", "data/administrative_boundaries")
        add(
            "country_cells", country_codes=list(countries), boundary_cache=boundary,
            output=root / "country/cells.csv",
        )
        extraction = add(
            "extract_country", **common, country_codes=list(countries),
            runs=[str(source)], boundary_cache=boundary, output_directory=root / "country",
        )
        steps.update(country_cells=["country_cells"], extract_country=["extract_country"])
        for index, (code, thresholds) in enumerate(countries.items()):
            country = code.lower()
            folder = root / "country" / country
            cache = Path(extraction["output_directory"]) / country / "observations.parquet"
            classification = (
                "country_classification" if index == 0
                else f"country_classification_{country}"
            )
            add(
                classification, **common, cache_path=cache, thresholds_c=thresholds,
                output_root=folder / "classification",
            )
            steps["country_classification"].append(classification)
            country_iso = add(f"iso7243_{country}", **{
                **iso, "hsb_run_paths": None, "country_cache": cache,
                "thresholds_c": thresholds, "output_root": folder / "hourly_classification", "date_workers": 1,
            })
            add(
                f"plot_iso7243_{country}", input_root=country_iso["output_root"],
                population_weights_path=population_path,
            )
            add(f"plot_non_consensus_{country}", input_root=country_iso["output_root"])
            for workflow in ("iso7243", "plot_iso7243", "plot_non_consensus"):
                steps[workflow].append(f"{workflow}_{country}")

    pipeline = add("pipeline", output_root=root / "pipeline", steps=steps)
    if isinstance(pipeline["steps"], list):
        selected = pipeline["steps"]
        if any(not isinstance(name, str) or name not in steps for name in selected):
            raise ValueError("pipeline.steps contains an unknown or unavailable workflow")
        pipeline["steps"] = {name: steps[name] for name in selected}
    return sections
