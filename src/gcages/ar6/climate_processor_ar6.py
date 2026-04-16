"""
Workflow for climate processor
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

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
from pandas_openscm.io import load_timeseries_csv
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


def run_workflow_ar6(  # noqa: PLR0913
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
    # TODO: change to default being None i.e. use everything in probabilistic_file
    num_cfgs: int = 600,
    scenario_batch_size: int = 20,
    **kwargs,
) -> IamDataFrame:
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
    :
        Results
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

    historical_emissions = get_ar6_full_historical_emissions(
        infilling_database_cfcs_path
    )
    infiller = AR6Infiller.from_ar6_config(
        ar6_infilling_db_file=infilling_database,
        ar6_infilling_db_cfcs_file=infilling_database_cfcs_path,
        harmonisation_year=2015,
        progress=progress,
        n_processes=n_processes,
        run_checks=True,
        historical_emissions=historical_emissions,
    )

    scm_runner = AR6SCMRunner(
        climate_models_cfgs=magicc_ar6_prob_cfg,
        output_variables=magicc_output_variables,
        harmonisation_year=2015,
        batch_size_scenarios=scenario_batch_size,
        progress=progress,
        n_processes=n_processes,
        run_checks=True,
        historical_emissions=historical_emissions,
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
        run_checks=True,
        progress=progress,
        n_processes=n_processes,
    )

    pre_processed = pre_processor(raw)

    harmonised = harmoniser(pre_processed)

    infilled = infiller(harmonised)

    complete = pix.concat([harmonised, infilled])

    scm_results_raw = scm_runner(complete)

    # Add derived variables
    scm_results_raw = pix.concat(
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

    scm_results_raw_quantiles = fix_index_name_after_groupby_quantile(
        groupby_except(
            scm_results_raw,
            "run_id",
        ).quantile(quantiles_timeseries),  # type: ignore # pandas-stubs confused
        new_name="quantile",
    )

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

    scm_results_raw_quantiles = scm_results_raw_quantiles.rename(
        index=lambda x: replacements[x] if x in replacements else x, level="variable"
    )

    post_processed = post_processor(scm_results_raw)

    timeseries_scm = pd.concat(
        [
            scm_results_raw_quantiles,
            post_processed.timeseries_quantile,
        ]
    )

    timeseries_scm = timeseries_scm.rename(
        index=lambda x: np.round(x * 100.0, 1), level="quantile"
    )
    timeseries_scm.index = timeseries_scm.index.rename({"quantile": "percentile"})

    timeseries_scm = timeseries_scm.pix.format(
        variable="AR6 climate diagnostics|{variable}|{climate_model}|{percentile}th Percentile",  # noqa: E501
        drop=True,
    )
    timeseries_scm.columns = timeseries_scm.columns.astype(
        scm_results_raw.columns.dtype
    )

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

    def ckg(
        indf: pd.DataFrame, gwp: str, kyoto_ghgs: tuple[str, ...] = complete_kyoto_ghgs
    ) -> pd.DataFrame:
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

    kyoto_ghgs_harmonised = tuple(
        set(harmonised.index.get_level_values("variable")).intersection(
            set(complete_kyoto_ghgs)
        )
    )
    harmonised_kyoto = pix.concat(
        [
            ckg(
                harmonised,
                gwp=gwp,
                # For harmonisation, use whatever is there rather than checking strictly
                kyoto_ghgs=kyoto_ghgs_harmonised,
            )
            for gwp in ["AR6GWP100", "AR5GWP100"]
        ]
    )
    harmonised_kyoto_out = harmonised_kyoto.rename(
        index=lambda x: f"AR6 climate diagnostics|Harmonized|{x}",
        level="variable",
    )

    complete_kyoto = pix.concat(
        [ckg(complete, gwp=gwp) for gwp in ["AR6GWP100", "AR5GWP100"]]
    )
    infilled_kyoto_out = complete_kyoto.rename(
        index=lambda x: f"AR6 climate diagnostics|Infilled|{x}",
        level="variable",
    )

    def get_output_emissions(
        indf: pd.DataFrame, name_id: str, include_hfc4310_unit: bool = True
    ) -> pd.DataFrame:
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

    harmonised_out = get_output_emissions(
        harmonised,
        "Harmonized",
        # Only done for infilled for some reason
        include_hfc4310_unit=False,
    )
    infilled_out = get_output_emissions(complete, "Infilled")

    res = IamDataFrame(
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

    post_processed_for_metadata = evolve(
        post_processed,
        metadata_quantile=post_processed.metadata_quantile.loc[
            post_processed.metadata_quantile.index.get_level_values("quantile").isin(
                quantiles_metadata
            )
        ],
    )
    metadata = get_res_meta(post_processed_for_metadata)

    res.set_meta(metadata)

    return res


def get_res_meta(res_pp: PostProcessingResult):
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
    c1a_loc = peak_warming_quantiles_use[0.5] < 1.5
    categories["Category_name"].loc[
        (
            categories["Category_name"]
            == "C1: limit warming to 1.5°C (>50%) with no or limited overshoot"
        )
        & c1a_loc
    ] = "C1a"
    categories["Category_name"].loc[
        (
            categories["Category_name"]
            == "C1: limit warming to 1.5°C (>50%) with no or limited overshoot"
        )
        & ~c1a_loc
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
