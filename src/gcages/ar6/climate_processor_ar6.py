"""
Workflow for climate processor
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
from gcages.ar6.post_processing import set_new_single_value_levels
from gcages.exceptions import MissingOptionalDependencyError
from gcages.post_processing import PostProcessingResult
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import (
    convert_openscm_runner_output_names_to_magicc_output_names,
)


def calculate_kyoto_ghgs(indf: pd.DataFrame, gwp: str = "AR6GWP100"):
    """Calculate aggregate Kyoto GHG emissions in CO2-equivalent units."""
    try:
        import pandas_indexing as pix
        import pint

        pix.set_openscm_registry_as_default()
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "calculate_kyoto_ghgs", requirement="pandas_indexing"
        ) from exc

    if "Emissions|CO2" not in indf.pix.unique("variable"):
        res = (
            indf.loc[
                pix.isin(
                    variable=[
                        "Emissions|CO2|Biosphere",
                        "Emissions|CO2|Fossil",
                    ]
                )
            ]
            .groupby(["model", "scenario", "region", "unit"])
            .sum(min_count=2)
            .pix.assign(variable="Emissions|CO2")
        )
        indf = pix.concat(
            [
                indf,
                res,
            ]
        )

    KYOTO_GHGS = [
        # 'Emissions|CO2|AFOLU',
        # 'Emissions|CO2|Energy and Industrial Processes',
        "Emissions|CO2",
        "Emissions|CH4",
        "Emissions|N2O",
        "Emissions|HFC125",
        "Emissions|HFC134a",
        "Emissions|HFC143a",
        "Emissions|HFC227ea",
        "Emissions|HFC23",
        "Emissions|HFC32",
        "Emissions|HFC4310mee",
        "Emissions|SF6",
        "Emissions|C2F6",
        "Emissions|C6F14",
        "Emissions|CF4",
    ]

    ALL_GHGS = [
        *KYOTO_GHGS,
        "Emissions|CCl4",
        "Emissions|CFC11",
        "Emissions|CFC113",
        "Emissions|CFC114",
        "Emissions|CFC115",
        "Emissions|CFC12",
        "Emissions|CH2Cl2",
        "Emissions|CH3Br",
        "Emissions|CH3CCl3",
        "Emissions|CH3Cl",
        "Emissions|CHCl3",
        "Emissions|HCFC141b",
        "Emissions|HCFC142b",
        "Emissions|HCFC22",
        "Emissions|Halon1202",
        "Emissions|Halon1211",
        "Emissions|Halon1301",
        "Emissions|Halon2402",
        "Emissions|SO2F2",
    ]

    not_handled = set(indf.pix.unique("variable")) - set(KYOTO_GHGS)
    not_handled_problematic = (
        not_handled
        - {
            "Emissions|OC",
            "Emissions|SOx",
            "Emissions|CO2|Biosphere",
            "Emissions|CO",
            "Emissions|NMVOC",
            "Emissions|BC",
            "Emissions|CO2|Fossil",
            "Emissions|NOx",
            "Emissions|NH3",
        }
        # climate-assessment calculate kyoto_ghgs with less gases
        - {
            "Emissions|HFC152a",
            "Emissions|HFC236fa",
            "Emissions|HFC245fa",
            "Emissions|HFC365mfc",
            "Emissions|NF3",
            "Emissions|C3F8",
            "Emissions|C4F10",
            "Emissions|C5F12",
            "Emissions|C7F16",
            "Emissions|C8F18",
            "Emissions|cC4F8",
        }
        - set(ALL_GHGS)
    )
    if not_handled_problematic:
        raise AssertionError(not_handled_problematic)
    # breakpoint()
    with pint.get_application_registry().context(gwp):
        gwp_str = f"{gwp[:3]}-{gwp[3:]}"
        res = (
            indf.loc[pix.isin(variable=KYOTO_GHGS)]
            .pix.convert_unit("MtCO2 / yr")
            .groupby(["model", "scenario", "region", "unit"])
            # .sum(min_count=2)
            .sum()
            .pix.assign(
                variable=f"Emissions|Kyoto Gases ({gwp_str})", unit="Mt CO2-equiv/yr"
            )
        )

    return res


def get_ar6_post_processed_metadata_comparable(res_pp: PostProcessingResult):
    """
    Format AR6 post-processed metadata into a comparable table.
    """
    out_index = ["model", "scenario"]

    # Works only for MAGICC, others need the category in their name
    categories = res_pp.metadata_categories.unstack("metric")
    categories.columns = categories.columns.str.capitalize()
    categories = categories.reset_index(
        categories.index.names.difference(out_index), drop=True
    )

    exceedance_probs_s = update_index_levels_func(
        res_pp.metadata_exceedance_probabilities,
        {"threshold": lambda x: np.round(x, 1)},
    )
    exceedance_probs_s = exceedance_probs_s.pix.format(
        out_name="Exceedance Probability {threshold}C ({climate_model})"
    ).drop(index=[1.0], level="threshold")
    exceedance_probs = exceedance_probs_s.reset_index(
        exceedance_probs_s.index.names.difference([*out_index, "out_name"]), drop=True
    ).unstack("out_name")
    exceedance_probs = exceedance_probs / 100.0

    def get_out_quantile(q: float) -> str:
        half = 0.5
        if q == half:
            return "median"

        return f"p{q*100:.0f}"

    quantile_metadata_l = []
    for v_str, metric_id in (
        ("peak warming", "max"),
        ("year of peak warming", "max_year"),
        ("warming in 2100", 2100),
    ):
        start = res_pp.metadata_quantile[
            res_pp.metadata_quantile.index.get_level_values("metric") == metric_id
        ]
        tmp_a = update_index_levels_func(start, {"quantile": get_out_quantile})
        tmp_a = tmp_a[~tmp_a.index.get_level_values("quantile").duplicated()]
        tmp_b = set_new_single_value_levels(tmp_a, {"v_str": v_str}, copy=False)
        tmp_c = tmp_b.pix.format(out_name="{quantile} {v_str} ({climate_model})")
        tmp_d = tmp_c.reset_index(
            tmp_a.index.names.difference([*out_index, "out_name"]), drop=True
        ).unstack("out_name")

        quantile_metadata_l.append(tmp_d)

    quantile_metadata = pd.concat(quantile_metadata_l, axis="columns")

    out = pd.concat([exceedance_probs, quantile_metadata, categories], axis="columns")
    out = out.rename_axis(index={"model": "Model", "scenario": "Scenario"})

    return out.reset_index()


def run_workflow_ar6(  # noqa: PLR0913, PLR0915
    input_emissions_file_path: Path,
    outdir_path: Path,
    infilling_database: Path,
    probabilistic_file: Path,
    history_database_path: Path,
    infilling_database_cfcs_path: Path,
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
        msg = f"No test data for {model=} ?"
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
            "out_dynamic_vars": convert_openscm_runner_output_names_to_magicc_output_names(  # noqa: E501
                out_variables
            ),
            "out_ascii_binary": "BINARY",
            "out_binary_format": 2,
        }

        run_config = [{**common_cfg, **base_cfg} for base_cfg in cfgs]

        magicc_ar6_prob_cfg = {"MAGICC7": run_config}

    pre_processor = AR6PreProcessor.from_ar6_config(
        n_processes=None,  # not parallel?
        progress=False,
    )

    harmoniser = AR6Harmoniser.from_ar6_config(
        ar6_historical_emissions_file=history_database_path,
        n_processes=None,  # not parallel?
        progress=False,
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
        progress=False,
        n_processes=None,  # not parallel
    )

    infilled = infiller(harmonised)

    OUTPUT_VARIABLES = (
        # GSAT
        "Surface Air Temperature Change",
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
    # I have not been able to find:
    # Effective Radiative Forcing|Basket|Non-CO2 Anthropogenic,
    # Raw Surface Temperature (GSAT),
    # Effective Radiative Forcing|Basket|Non-CO2 Greenhouse Gases
    scm_runner = AR6SCMRunner(
        climate_models_cfgs=magicc_ar6_prob_cfg,
        output_variables=OUTPUT_VARIABLES,
        # historical_emissions=historical_df,  # pandas DataFrame
        run_checks=False,
        harmonisation_year=2015,
        batch_size_scenarios=scenario_batch_size,
    )

    scm_results = scm_runner(infilled)

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
        "Surface Air Temperature Change": "Surface Temperature (GSAT)",
        # "Raw Surface Temperature (GSAT)",
    }

    post_processed_list = []

    for magicc_variable in OUTPUT_VARIABLES:
        output_variable = replacements.get(magicc_variable, magicc_variable)

        post_processor = AR6PostProcessor(
            gsat_assessment_median=0.85,
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
            # n_processes=n_processes,
        )
        post_processed_single = post_processor(scm_results)
        post_processed_list.append(
            post_processed_single.timeseries_quantile.loc[:, 1995:]
        )

        if magicc_variable == "Surface Air Temperature Change":
            metadata = get_ar6_post_processed_metadata_comparable(post_processed_single)
            metadata["harmonisation"] = "aneris (version: 0.3.1)"
            metadata["infilling"] = "silicone (version: 1.3.0)"
            metadata["climate-models"] = "openscm_runner (version: 0.12.1)"
            metadata["workflow"] = "climate_assessment (version: 0.1.6)"

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
        # AR6 put harmonised in the infilled group too for some reason,
        # except CO2
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

    with pd.ExcelWriter(Path(outdir_path) / "data_alloutput.xlsx") as writer:
        out.to_excel(writer, sheet_name="data", index=False)
        metadata.to_excel(writer, sheet_name="meta", index=False)
