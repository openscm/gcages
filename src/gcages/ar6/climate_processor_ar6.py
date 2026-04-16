"""
Workflow for climate processor

TODO: move into climate-processor
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import aneris
import numpy as np
import openscm_runner
import pandas as pd
import pandas_indexing as pix
import silicone
from attrs import evolve
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.index_manipulation import update_index_levels_func
from pyam import IamDataFrame

import gcages
from gcages.ar6 import (
    AR6Harmoniser,
    AR6Infiller,
    AR6PostProcessor,
    AR6PreProcessor,
    AR6SCMRunner,
    get_ar6_full_historical_emissions,
)
from gcages.ar6.post_processing import set_new_single_value_levels
from gcages.ghg_aggregation import calculate_kyoto_ghgs
from gcages.post_processing import PostProcessingResult
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import (
    convert_openscm_runner_output_names_to_magicc_output_names,
)
from gcages.scm_running.magicc import (
    combine_probabilistic_and_common_cfg,
    load_magicc_probabilistic_config,
)

SCM_OUTPUT_VARIABLES_DEFAULT = (
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


def get_climate_model_cfg(
    scm_probabilistic_config_file: Path,
    model: str,
    model_version: str,
    scm_output_variables: tuple[str, ...],
    num_cfgs: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Get simple climate model (SCM) configuration

    Parameters
    ----------
    scm_probabilistic_config_file
        File from which to load the probabilistic config for the SCM

    model
        SCM to run

    model_version
        SCM version to run

    scm_output_variables
        Output variables to get from the SCM

    num_cfgs
        Number of configurations to run with the SCM

        If not supplied, all configurations in `scm_probabilistic_config_file` are run.

    Returns
    -------
    :
        Loaded SCM configurations
    """
    if model_version == "v7.5.3" and model == "magicc":
        cfgs = load_magicc_probabilistic_config(scm_probabilistic_config_file)

        if num_cfgs is not None:
            cfgs = cfgs[:num_cfgs]

        common_cfg = {
            "startyear": 1750,
            "out_dynamic_vars": convert_openscm_runner_output_names_to_magicc_output_names(  # noqa: E501
                scm_output_variables
            ),
            "out_ascii_binary": "BINARY",
            "out_binary_format": 2,
        }

        run_config = combine_probabilistic_and_common_cfg(cfgs, common_cfg=common_cfg)

        res = {"MAGICC7": run_config}

    else:
        raise NotImplementedError(f"{model} {model_version}")

    return res


def add_derived_scm_variables(scm_results_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived simple climate model variables

    Parameters
    ----------
    scm_results_raw
        Raw simple climate model results

    Returns
    -------
    :
        `scm_results_raw` plus the derived variables
    """
    res = pix.concat(
        [
            scm_results_raw,
            (
                scm_results_raw.xs(
                    "Effective Radiative Forcing|Anthropogenic", level="variable"
                )
                - scm_results_raw.xs(
                    "Effective Radiative Forcing|CO2", level="variable"
                )
            ).pix.assign(
                variable="Effective Radiative Forcing|Basket|Non-CO2 Anthropogenic"
            ),
            (
                scm_results_raw.xs(
                    "Effective Radiative Forcing|Greenhouse Gases", level="variable"
                )
                - scm_results_raw.xs(
                    "Effective Radiative Forcing|CO2", level="variable"
                )
            ).pix.assign(
                variable="Effective Radiative Forcing|Basket|Non-CO2 Greenhouse Gases"
            ),
        ]
    )

    return res


def rename_scm_variables(scm_results: pd.DataFrame) -> pd.DataFrame:
    """
    Rename simple climate model variables

    Parameters
    ----------
    scm_results_raw
        Simple climate model results to rename

    Returns
    -------
    :
        `scm_results_raw` with renamed variables
    """
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

    res = scm_results.rename(
        index=lambda x: replacements[x] if x in replacements else x, level="variable"
    )

    return res


def format_scm_timeseries(timeseries_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Format simple climate model timeseries

    Parameters
    ----------
    timeseries_raw
        Raw timeseries data

    Returns
    -------
    :
        Formatted timeseries
    """
    res = timeseries_raw.rename(
        index=lambda x: np.round(x * 100.0, 1), level="quantile"
    )
    res.index = res.index.rename({"quantile": "percentile"})
    res = res.pix.format(
        variable="AR6 climate diagnostics|{variable}|{climate_model}|{percentile}th Percentile",  # noqa: E501
        drop=True,
    )

    return res


def get_kyoto_timeseries(indf: pd.DataFrame, harmonised: bool) -> pd.DataFrame:
    """
    Get kyoto aggregate timeseries

    Parameters
    ----------
    indf
        Input data

    harmonised
        Is this harmonised data?

        If yes, it is treated with a few special bits of behaviour.

    Returns
    -------
    :
        Kyoto aggregate timeseries
    """
    # Not actually the full list, but what has been used in AR6 for some reason
    # (not a major issue, this covers the majority of Kyoto GHG emissions).
    complete_kyoto_ghgs = (
        "Emissions|CO2|Fossil",
        "Emissions|CO2|Biosphere",
        "Emissions|CH4",
        "Emissions|HFC125",
        "Emissions|HFC134a",
        "Emissions|HFC143a",
        "Emissions|HFC227ea",
        "Emissions|HFC23",
        "Emissions|HFC32",
        "Emissions|HFC4310mee",
        "Emissions|N2O",
        "Emissions|C2F6",
        "Emissions|C6F14",
        "Emissions|CF4",
        "Emissions|SF6",
    )
    if harmonised:
        # Only consider what is in the data, don't insist on having everything
        kyoto_ghgs = tuple(
            set(indf.index.get_level_values("variable")).intersection(
                set(complete_kyoto_ghgs)
            )
        )

    else:
        kyoto_ghgs = complete_kyoto_ghgs

    def ckg(indf: pd.DataFrame, gwp: str) -> pd.DataFrame:
        tmp = calculate_kyoto_ghgs(
            indf,
            indf_naming_convention=SupportedNamingConventions.GCAGES,
            kyoto_ghgs=kyoto_ghgs,
            gwp=gwp,
        )
        gwp_str = f"{gwp[:3]}-{gwp[3:]}"
        res = tmp.pix.assign(
            variable=f"Emissions|Kyoto Gases ({gwp_str})", unit="Mt CO2-equiv/yr"
        )

        return res

    kyoto = pix.concat([ckg(indf, gwp=gwp) for gwp in ["AR6GWP100", "AR5GWP100"]])
    if harmonised:
        out_id = "Harmonized"
    else:
        out_id = "Infilled"

    kyoto_out = kyoto.rename(
        index=lambda x: f"AR6 climate diagnostics|{out_id}|{x}",
        level="variable",
    )

    return kyoto_out


def format_emissions(
    indf: pd.DataFrame, name_id: str, include_hfc4310_unit: bool = True
) -> pd.DataFrame:
    """
    Format emissions timeseries

    Parameters
    ----------
    indf
        Input data

    name_id
        ID to include in the output name e.g. "Harmonized", "Infilled"

    include_hfc4310_unit
        Should the unit of HFC4310mee be formatted too?

    Returns
    -------
    :
        Formatted data
    """

    def update_unit(u: str) -> str:
        res = u.replace("HFC245fa", "HFC245ca")
        if include_hfc4310_unit:
            res = res.replace("HFC4310", "HFC43-10")

        return res

    out_emms = update_index_levels_func(
        indf,
        {
            "variable": lambda x: (
                f"AR6 climate diagnostics|{name_id}|"
                + convert_variable_name(
                    x,
                    from_convention=SupportedNamingConventions.GCAGES,
                    to_convention=SupportedNamingConventions.IAMC,
                )
            ),
            "unit": update_unit,
        },
    )

    return out_emms


def get_res_meta(res_pp: PostProcessingResult) -> pd.DataFrame:
    """
    Get metadata for the results

    Parameters
    ----------
    res_pp
        Post-processed results from gcages

    Returns
    -------
    :
        Metadata for the results
    """
    out_index = ["model", "scenario"]

    # Works only for MAGICC, others need the category in their name
    categories = res_pp.metadata_categories.unstack("metric")
    categories.columns = categories.columns.str.capitalize()
    categories = categories.reset_index(
        categories.index.names.difference(out_index), drop=True
    )
    # Put back in C1a and C1b like climate-assessment used to do.
    peak_warming_quantiles = res_pp.metadata_quantile.loc[
        pix.isin(metric="max")
    ].unstack("quantile")
    peak_warming_quantiles_use = peak_warming_quantiles.reset_index(
        peak_warming_quantiles.index.names.difference(["model", "scenario"]), drop=True
    )
    c1a_loc = peak_warming_quantiles_use[0.5] < 1.5  # noqa: PLR2004
    categories.loc[
        (
            categories["Category_name"]
            == "C1: limit warming to 1.5°C (>50%) with no or limited overshoot"
        )
        & c1a_loc,
        "Category_name",
    ] = "C1a"
    categories.loc[
        (
            categories["Category_name"]
            == "C1: limit warming to 1.5°C (>50%) with no or limited overshoot"
        )
        & ~c1a_loc,
        "Category_name",
    ] = "C1b"

    # There must have been a mapping done somewhere between climate-assessment
    # and what appears on the scenario explorer.
    # gcages matches what is on the scenario explorer,
    # so we have to undo that mapping here.
    category_name_mapping = {
        "C8: exceed warming of 4°C (>=50%)": "C8: Above 4.0°C",
        "C7: limit warming to 4°C (>50%)": "C7: Below 4.0°C",
        "C6: limit warming to 3°C (>50%)": "C6: Below 3.0°C",
        "C5: limit warming to 2.5°C (>50%)": "C5: Below 2.5°C",
        "C4: limit warming to 2°C (>50%)": "C4: Below 2°C",
        "C3: limit warming to 2°C (>67%)": "C3: Likely below 2°C",
        "C2: return warming to 1.5°C (>50%) after a high overshoot": "C2: Below 1.5°C with high OS",  # noqa: E501
        "C1b": "C1b: Below 1.5°C with low OS",
        "C1a": "C1a: Below 1.5°C with no OS",
    }
    categories["Category_name"] = categories["Category_name"].map(category_name_mapping)

    exceedance_probs_s = update_index_levels_func(
        res_pp.metadata_exceedance_probabilities,
        {"threshold": lambda x: np.round(x, 1)},
    )
    exceedance_probs_s = exceedance_probs_s.pix.format(
        out_name="Exceedance Probability {threshold}C ({climate_model})"
    )
    exceedance_probs = exceedance_probs_s.reset_index(
        exceedance_probs_s.index.names.difference([*out_index, "out_name"]), drop=True
    ).unstack("out_name")
    exceedance_probs = exceedance_probs / 100.0

    def get_out_quantile(q: float) -> str:
        if q == 0.5:  # noqa: PLR2004
            return "median"

        return f"p{q*100:.0f}"

    quantile_metadata_l = []
    for v_str, metric_id in (
        ("peak warming", "max"),
        ("warming in 2100", 2100),
        ("year of peak warming", "max_year"),
    ):
        start = res_pp.metadata_quantile[
            res_pp.metadata_quantile.index.get_level_values("metric") == metric_id
        ]
        tmp_a = update_index_levels_func(start, {"quantile": get_out_quantile})
        tmp_b = set_new_single_value_levels(tmp_a, {"v_str": v_str}, copy=False)
        tmp_c = tmp_b.pix.format(out_name="{quantile} {v_str} ({climate_model})")
        tmp_d = tmp_c.reset_index(
            tmp_a.index.names.difference([*out_index, "out_name"]), drop=True
        ).unstack("out_name")

        quantile_metadata_l.append(tmp_d)

    quantile_metadata = pd.concat(quantile_metadata_l, axis="columns")

    out = pd.concat([categories, exceedance_probs, quantile_metadata], axis="columns")

    out["harmonization"] = f"aneris (version: {aneris.__version__})"
    out["infilling"] = f"silicone (version: {silicone.__version__})"
    out["climate-models"] = f"openscm_runner (version: {openscm_runner.__version__})"
    out["workflow"] = f"gcages (version: {gcages.__version__})"

    return out


def run_workflow_ar6(  # noqa: PLR0913
    input_emissions: pd.DataFrame,
    historical_emissions_file: Path,
    infilling_database_file: Path,
    infilling_database_cfcs_file: Path,
    scm: str,
    scm_version: str,
    scm_probabilistic_config_file: Path,
    scm_num_cfgs: int | None = None,
    scm_output_variables: tuple[str, ...] = SCM_OUTPUT_VARIABLES_DEFAULT,
    output_variables: list[str] = [
        "AR6 climate diagnostics|Harmonized|Emissions**",
        "AR6 climate diagnostics|Infilled|Emissions**",
        "AR6 climate diagnostics|Surface Temperature**",
    ],
    scenario_batch_size: int = 20,
    progress: bool = True,
    n_processes: int | None = multiprocessing.cpu_count(),
    run_checks: bool = True,
) -> IamDataFrame:
    """
    Run AR6 workflow

    This function is specific to how this is done in the Scenario Explorer.
    For a general solution (that is also used by this function),
    see the [gcages](https://gcages.readthedocs.io/) package.

    Parameters
    ----------
    input_emissions
        Input emissions

    historical_emissions_file
        File containing the historical emissions

    infilling_database_file
        File containing the infilling database

    infilling_database_cfcs_file
        File containing the infilling database for CFCs

    scm
        Simple climate model to run

    scm_version
        Simple climate model version to run

    scm_probabilistic_config_file
        File containing the simple climate model probabilistic config to use

    scm_num_cfgs
        Number of configurations of the simple climate model to run

        If not supplied, all configurations found in `scm_probabilistic_config_file`
        are run.

    scm_output_variables
        Simple climate model output variables to request

    output_variables
        Variables to return from the workflow

    scenario_batch_size
        Number of scenarios to run at a single time

        If this is too big, you can run into memory issues

    progress
        Should progress bars be shown for each step?

    n_processes
        Number of parallel processes to use when processing

    run_checks
        Should checks be run at each step?

    Returns
    -------
    :
        Results of the workflow
    """
    raw = update_index_levels_func(
        input_emissions,
        {
            "variable": lambda x: x.replace("AR6 climate diagnostics|", "").replace(
                "|Unhamonized", ""
            )
        },
        copy=False,
    )

    pre_processor = AR6PreProcessor.from_ar6_config(
        progress=progress,
        n_processes=n_processes,
    )

    harmoniser = AR6Harmoniser.from_ar6_config(
        ar6_historical_emissions_file=historical_emissions_file,
        progress=progress,
        n_processes=n_processes,
        run_checks=run_checks,
    )

    historical_emissions = get_ar6_full_historical_emissions(
        infilling_database_cfcs_file
    )
    infiller = AR6Infiller.from_ar6_config(
        ar6_infilling_db_file=infilling_database_file,
        ar6_infilling_db_cfcs_file=infilling_database_cfcs_file,
        progress=progress,
        n_processes=n_processes,
        run_checks=run_checks,
        harmonisation_year=harmoniser.harmonisation_year,
        historical_emissions=historical_emissions,
    )

    climate_model_cfg = get_climate_model_cfg(
        scm_probabilistic_config_file=scm_probabilistic_config_file,
        model=scm,
        model_version=scm_version,
        scm_output_variables=scm_output_variables,
        num_cfgs=scm_num_cfgs,
    )

    # Also required for running MAGICC, but handled elsewhere in climate-processor
    # os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
    scm_runner = AR6SCMRunner(
        climate_models_cfgs=climate_model_cfg,
        output_variables=scm_output_variables,
        force_interpolate_to_yearly=True,
        batch_size_scenarios=scenario_batch_size,
        db=None,
        res_column_type=int,
        progress=progress,
        n_processes=n_processes,
        run_checks=run_checks,
        historical_emissions=historical_emissions,
        harmonisation_year=harmoniser.harmonisation_year,
    )

    # Somewhat inexplicably, the quantiles for metadata
    # are not the same as the quantiles for timeseries.
    quantiles_metadata = (
        0.05,
        0.10,
        0.17,
        0.25,
        0.33,
        0.50,
        0.66,
        0.67,
        0.75,
        0.83,
        0.90,
        0.95,
    )
    quantiles_timeseries = (
        *quantiles_metadata,
        1.0 / 6.0,
        5.0 / 6.0,
    )

    post_processor = AR6PostProcessor.from_ar6_config(
        exceedance_thresholds_of_interest=tuple([1.5 + i * 0.5 for i in range(8)]),
        quantiles_of_interest=quantiles_timeseries,
        run_checks=run_checks,
        progress=progress,
        n_processes=n_processes,
    )

    pre_processed = pre_processor(raw)
    harmonised = harmoniser(pre_processed)
    infilled = infiller(harmonised)
    complete = pix.concat([harmonised, infilled])
    scm_results_raw = scm_runner(complete)
    post_processed = post_processor(scm_results_raw)

    scm_results_raw = add_derived_scm_variables(scm_results_raw)
    scm_results_raw_quantiles = fix_index_name_after_groupby_quantile(
        groupby_except(
            scm_results_raw,
            "run_id",
        ).quantile(quantiles_timeseries),  # type: ignore # pandas-stubs confused
        new_name="quantile",
    )
    scm_results_raw_quantiles = rename_scm_variables(scm_results_raw_quantiles)

    timeseries_scm = format_scm_timeseries(
        pix.concat([scm_results_raw_quantiles, post_processed.timeseries_quantile])
    )

    harmonised_kyoto_out = get_kyoto_timeseries(harmonised, harmonised=True)
    infilled_kyoto_out = get_kyoto_timeseries(complete, harmonised=False)
    harmonised_out = format_emissions(
        harmonised,
        "Harmonized",
        # Only done for infilled for some reason
        include_hfc4310_unit=False,
    )
    infilled_out = format_emissions(complete, "Infilled")

    res_df = (
        pix.concat(
            [
                harmonised_out,
                harmonised_kyoto_out,
                infilled_out,
                infilled_kyoto_out,
                timeseries_scm,
            ]
        )
        .sort_index(axis="columns")
        .loc[:, 1995:]
    )
    res = IamDataFrame(res_df).filter(variable=output_variables)

    post_processed_for_metadata = evolve(
        post_processed,
        # Drop out the quantiles that aren't included in metadata
        metadata_quantile=post_processed.metadata_quantile.loc[
            post_processed.metadata_quantile.index.get_level_values("quantile").isin(
                quantiles_metadata
            )
        ],
    )
    metadata = get_res_meta(post_processed_for_metadata)

    res.set_meta(metadata)

    return res
