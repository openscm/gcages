"""
Workflow for climate processor
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pandas as pd
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from gcages.ar6 import (
    AR6Harmoniser,
    AR6Infiller,
    AR6PostProcessor,
    AR6PreProcessor,
    AR6SCMRunner,
    get_ar6_full_historical_emissions,
)
from gcages.ghg_aggregation import calculate_kyoto_ghgs
from gcages.renaming import SupportedNamingConventions, convert_variable_name

MAGICC_OUTPUT_VARIABLES = (
    # GSAT
    "Surface Air Temperature Change",
    # "Raw Surface Air Temperature Change",
    # GMST
    "Surface Air Ocean Blended Temperature Change",
    # ERFs
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Anthropogenic",
    "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|CFC11",
    "Effective Radiative Forcing|CFC12",
    "Effective Radiative Forcing|HCFC22",
    "Effective Radiative Forcing|HFC125",
    "Effective Radiative Forcing|HFC134a",
    "Effective Radiative Forcing|HFC143a",
    "Effective Radiative Forcing|HFC227ea",
    "Effective Radiative Forcing|HFC23",
    "Effective Radiative Forcing|HFC245fa",
    "Effective Radiative Forcing|HFC32",
    "Effective Radiative Forcing|HFC4310mee",
    "Effective Radiative Forcing|CF4",
    "Effective Radiative Forcing|C6F14",
    "Effective Radiative Forcing|C2F6",
    "Effective Radiative Forcing|SF6",
    # Heat uptake
    "Heat Uptake",
    # "Heat Uptake|Ocean",
    # Atmospheric concentrations
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    # Carbon cycle
    "Net Atmosphere to Land Flux|CO2",
    "Net Atmosphere to Ocean Flux|CO2",
    # Permafrost
    "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
)


def run_workflow_ar6(  # noqa: PLR0913, PLR0915
    input_emissions_file_path: Path,
    infilling_database: Path,
    probabilistic_file: Path,
    history_database_path: Path,
    infilling_database_cfcs_path: Path,
    progress: bool = True,
    n_processes: int | None = multiprocessing.cpu_count(),
    magicc_output_variables=MAGICC_OUTPUT_VARIABLES,
    model: str = "magicc",
    model_version: str = "v7.5.3",
    out_variables: list[str] = [
        "AR6 climate diagnostics|Surface Temperature*",
    ],
    num_cfgs: int = 600,
    scenario_batch_size: int = 20,
    **kwargs,
) -> None:
    """
    Run the full AR6 climate assessment workflow for emissions-to-climate outputs.

    This function executes the complete AR6 processing pipeline, including
    preprocessing, harmonisation, infilling, SCM climate model runs (MAGICC),
    post-processing, Kyoto GHG aggregation, and export of results to Excel.
    The workflow follows AR6 conventions for variables, scenarios, and metadata.

    Parameters
    ----------
    input_emissions_file_path
        Path to the input emissions time series CSV file
    outdir_path
        Output directory where processed results are written
    infilling_database
        Path to the AR6 infilling database
    probabilistic_file
        Path to the probabilistic MAGICC configuration JSON file
    history_database_path
        Path to the historical emissions database
    infilling_database_cfcs_path
        Path to the CFC-specific infilling database
    model
        Simple climate model to use
    model_version
        Version of the climate model configuration
    out_variables
        Output climate diagnostic variables requested from the SCM
    num_cfgs
        Number of probabilistic climate model configurations
    scenario_batch_size
        Number of scenarios processed per SCM batch.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    None
        Results are written to disk as Excel files in the output directory
    """
    raw = load_timeseries_csv(
        input_emissions_file_path,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_columns_type=int,
    )
    if raw.empty:
        msg = f"No data loaded from {input_emissions_file_path} ?"
        raise AssertionError(msg)

    raw = update_index_levels_func(
        raw,
        {
            "variable": lambda x: x.replace("AR6 climate diagnostics|", "").replace(
                "|Unhamonized", ""
            )
        },
        copy=False,
    )

    if model_version == "v7.5.3" and model == "magicc":
        # TODO: split out the logic from `load_ar6_magicc_probabilistic_config`
        # and use that here rather than duplicating the code
        with open(probabilistic_file) as fh:
            cfgs_raw = json.load(fh)

        cfgs = [
            {
                "run_id": c["paraset_id"],
                **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
            }
            for c in cfgs_raw["configurations"]
        ]

        startyear = 1750
        endyear = 2105

        common_cfg = {
            "startyear": startyear,
            "endyear": endyear,
            "out_dynamic_vars": magicc_output_variables,
            "out_ascii_binary": "BINARY",
            "out_binary_format": 2,
        }

        # TODO: split out a function for doing this combination
        # rather than duplicating the code
        run_config = [{**common_cfg, **base_cfg} for base_cfg in cfgs]

        magicc_ar6_prob_cfg = {"MAGICC7": run_config}

    pre_processor = AR6PreProcessor.from_ar6_config(
        progress=progress,
        n_processes=n_processes,
    )

    harmoniser = AR6Harmoniser.from_ar6_config(
        ar6_historical_emissions_file=history_database_path,
        progress=progress,
        n_processes=n_processes,
    )

    pre_processed = pre_processor(raw)

    harmonised = harmoniser(pre_processed)

    infiller = AR6Infiller.from_ar6_config(
        ar6_infilling_db_file=infilling_database,
        ar6_infilling_db_cfcs_file=infilling_database_cfcs_path,
        historical_emissions=get_ar6_full_historical_emissions(
            infilling_database_cfcs_path
        ),
        harmonisation_year=2015,
        progress=progress,
        n_processes=n_processes,
    )

    infilled = infiller(harmonised)

    # I have not been able to find:
    # Effective Radiative Forcing|Basket|Non-CO2 Anthropogenic,
    # Raw Surface Temperature (GSAT),
    # Effective Radiative Forcing|Basket|Non-CO2 Greenhouse Gases
    scm_runner = AR6SCMRunner(
        climate_models_cfgs=magicc_ar6_prob_cfg,
        output_variables=magicc_output_variables,
        # historical_emissions=historical_df,  # pandas DataFrame
        run_checks=False,
        harmonisation_year=2015,
        batch_size_scenarios=scenario_batch_size,
        progress=progress,
        n_processes=n_processes,
    )

    scm_results = scm_runner(infilled)
    # Calculating: "Effective Radiative Forcing|Basket|Non-CO2 Anthropogenic"
    diff = scm_results.xs(
        "Effective Radiative Forcing|Anthropogenic", level="variable"
    ) - scm_results.xs("Effective Radiative Forcing|CO2", level="variable")

    diff["variable"] = "Effective Radiative Forcing|Basket|Non-CO2 Anthropogenic"
    diff = diff.set_index("variable", append=True)
    diff = diff.reorder_levels(
        ["climate_model", "model", "region", "run_id", "scenario", "unit", "variable"]
    )
    scm_results = pd.concat([scm_results, diff], axis=0)

    # Calculating: Effective Radiative Forcing|Basket|Non-CO2 Greenhouse Gases
    diff = scm_results.xs(
        "Effective Radiative Forcing|Greenhouse Gases", level="variable"
    ) - scm_results.xs("Effective Radiative Forcing|CO2", level="variable")

    diff["variable"] = "Effective Radiative Forcing|Basket|Non-CO2 Greenhouse Gases"
    diff = diff.set_index("variable", append=True)
    diff = diff.reorder_levels(
        ["climate_model", "model", "region", "run_id", "scenario", "unit", "variable"]
    )
    scm_results = pd.concat([scm_results, diff], axis=0)

    replacements = {
        (
            "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost"
        ): "Net Land to Atmosphere Flux due to Permafrost|CH4",
        (
            "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost"
        ): "Net Land to Atmosphere Flux due to Permafrost|CO2",
        (
            "Effective Radiative Forcing|Anthropogenic"
        ): "Effective Radiative Forcing|Basket|Anthropogenic",
        (
            "Surface Air Ocean Blended Temperature Change"
        ): "Raw Surface Temperature (GMST)",
        (
            "Effective Radiative Forcing|Greenhouse Gases"
        ): "Effective Radiative Forcing|Basket|Greenhouse Gases",
        (
            "Effective Radiative Forcing|Aerosols|Direct Effect|SOx"
        ): "Effective Radiative Forcing|Aerosols|Direct Effect|Sulfur",
        "Surface Air Temperature Change": "Raw Surface Temperature (GSAT)",
    }

    post_processed_list = []

    for magicc_variable in magicc_output_variables:
        output_variable = replacements.get(magicc_variable, magicc_variable)

        post_processor = AR6PostProcessor(
            gsat_assessment_median=0.5,
            gsat_assessment_time_period=tuple(range(1995, 2014 + 1)),
            gsat_assessment_pre_industrial_period=tuple(range(1850, 1900 + 1)),
            quantiles_of_interest=(
                0.05,
                0.10,
                1.0 / 6.0,
                0.17,
                0.25,
                0.33,
                0.50,
                0.66,
                0.67,
                0.75,
                0.83,
                5.0 / 6.0,
                0.90,
                0.95,
            ),
            exceedance_thresholds_of_interest=tuple([1.0 + i * 0.5 for i in range(9)]),
            raw_gsat_variable_in=magicc_variable,
            assessed_gsat_variable=f"{output_variable}",
            run_checks=True,
            progress=progress,
            n_processes=n_processes,
        )
        post_processed_single = post_processor(scm_results)
        post_processed_list.append(
            post_processed_single.timeseries_quantile.loc[:, 1995:]
        )
        if magicc_variable == "Surface Air Temperature Change":
            post_processor = AR6PostProcessor.from_ar6_config(n_processes=n_processes)
            post_processed_temperature_gsat = post_processor(scm_results)
            post_processed_list.append(
                post_processed_temperature_gsat.timeseries_quantile.loc[:, 1995:]
            )

    post_processed = pd.concat(post_processed_list)

    harmonised_kyoto = pd.concat(
        [
            calculate_kyoto_ghgs(harmonised, gwp="AR6GWP100"),
            calculate_kyoto_ghgs(harmonised, gwp="AR5GWP100"),
        ]
    ).rename(
        index=lambda x: "AR6 climate diagnostics|Harmonized|" + x, level="variable"
    )
    infilled_kyoto = pd.concat(
        [
            calculate_kyoto_ghgs(pd.concat([harmonised, infilled]), gwp="AR6GWP100"),
            calculate_kyoto_ghgs(pd.concat([harmonised, infilled]), gwp="AR5GWP100"),
        ]
    ).rename(index=lambda x: "AR6 climate diagnostics|Infilled|" + x, level="variable")
    harmonised_kyoto = harmonised_kyoto.reorder_levels(
        ["model", "scenario", "variable", "region", "unit"]
    )
    infilled_kyoto = infilled_kyoto.reorder_levels(
        ["model", "scenario", "variable", "region", "unit"]
    )

    harmonised_infilled = update_index_levels_func(
        pd.concat(
            [
                infilled,
                harmonised.loc[
                    ~harmonised.index.get_level_values("variable").str.endswith("CO2")
                ],
            ]
        ),
        {
            "variable": lambda x: (
                "AR6 climate diagnostics|Infilled|"
                + convert_variable_name(
                    x,
                    from_convention=SupportedNamingConventions.GCAGES,
                    to_convention=SupportedNamingConventions.IAMC,
                )
            ),
            "unit": lambda x: x.replace("HFC245fa", "HFC245ca").replace(
                "HFC4310", "HFC43-10"
            ),
        },
    )
    harmonised = update_index_levels_func(
        harmonised,
        {
            "variable": lambda x: (
                "AR6 climate diagnostics|Harmonized|"
                + convert_variable_name(
                    x,
                    from_convention=SupportedNamingConventions.GCAGES,
                    to_convention=SupportedNamingConventions.IAMC,
                )
            )
        },
        copy=False,
    )

    res = post_processed.reset_index("quantile")
    res["percentile"] = (res["quantile"] * 100.0).round(1).astype(str)
    res = res.set_index("percentile", append=True).drop("quantile", axis="columns")

    res = res.pix.format(
        variable="AR6 climate diagnostics|{variable}|{climate_model}|{percentile}th Percentile",  # noqa: E501
        drop=True,
    )

    res.columns = res.columns.astype(post_processed.columns.dtype)
    res = res.reorder_levels(["model", "scenario", "variable", "region", "unit"])

    out = (
        pd.concat(
            [
                harmonised,
                harmonised_kyoto,
                harmonised_infilled,
                infilled_kyoto,
                res,
                raw,
            ]
        )
        .reset_index()
        .sort_values(by="variable")
    )

    metadata = pd.DataFrame(
        [
            {
                "harmonisation": "aneris (version: 0.3.1)",
                "infilling": "silicone (version: 1.3.0)",
                "climate-models": "openscm_runner (version: 0.12.1)",
                "workflow": "climate_assessment (version: 0.1.6)",
            }
        ]
    )

    metadata.index = pd.MultiIndex.from_tuples(
        [(out.model[0], out.scenario[0])],
        names=["model", "scenario"],
    )

    return out, post_processed_temperature_gsat, metadata
