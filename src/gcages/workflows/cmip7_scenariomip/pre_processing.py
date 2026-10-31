"""
Pre-processing for the CMIP7 ScenarioMIP workflow

This adds the steps (including model-specific fixes to the raw data)
that were applied in CMIP7 ScenarioMIP around
[CMIP7ScenarioMIPPreProcessor][gcages.cmip7_scenariomip.pre_processing.CMIP7ScenarioMIPPreProcessor].
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from attrs import define
from pandas_openscm.index_manipulation import (
    set_index_levels_func,
    update_index_levels_func,
    update_levels_from_other,
)

from gcages.cmip7_scenariomip.pre_processing import (
    CMIP7ScenarioMIPPreProcessingResult,
    CMIP7ScenarioMIPPreProcessor,
    ReaggregatorBasic,
)
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_complete_timeseries_index,
    get_internal_consistency_checking_index,
    get_required_timeseries_index,
)
from gcages.completeness import get_missing_levels
from gcages.pandas_indexing_helpers import import_pandas_indexing
from gcages.workflows.cmip7_scenariomip.inputs import load_harmonisation_methods

CO2_AFOLU_RAW_NAME = "Emissions|CO2|AFOLU"
"""Name of CO2 AFOLU emissions in the raw data"""

CO2_AFOLU_GCAGES_NAME = "Emissions|CO2|Biosphere"
"""Name of CO2 AFOLU emissions using gcages' naming convention"""


def get_model_prefix(model: str) -> str:
    """
    Get the prefix which identifies a model's regions

    In CMIP7 ScenarioMIP, a model's regions were identified
    as the regions which start with the first word of the model's name.

    Parameters
    ----------
    model
        Model

    Returns
    -------
    :
        Prefix which identifies the model's regions
    """
    return model.split(" ", maxsplit=1)[0]


@functools.cache
def get_used_variables() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Get the variables which are used in the workflow

    These are:

    - the variables used to create the region-sector (i.e. gridding) emissions
      during pre-processing
      (including the variables used to check internal consistency)
    - the globally reported CO2 AFOLU emissions
      (see [override_co2_afolu_with_reported_global_values][(m).])
    - the globally reported emissions of species which aren't in the
      region-sector emissions (e.g. F-gases).
      Pre-processing passes these straight through to the global-only emissions.
      They can only be harmonised if there is a harmonisation method for them,
      so we only include the ones in the recorded harmonisation methods.

    Any other timeseries in the raw data don't affect the results of the workflow.

    Returns
    -------
    :
        Variables used at the World level and variables used at the model region level
    """
    placeholder_region = "__model_region__"
    used = get_complete_timeseries_index((placeholder_region,)).append(  # type: ignore # pandas-stubs confused
        get_internal_consistency_checking_index((placeholder_region,))
    )
    regions = used.get_level_values("region")
    variables = used.get_level_values("variable")

    model_region_variables = tuple(
        sorted(set(variables[regions == placeholder_region]))
    )

    region_sector_species = {v.split("|")[1] for v in model_region_variables}
    global_methods = load_harmonisation_methods("harmonisation-methods_global.csv")
    global_only_species_variables = {
        v
        for v in global_methods.index.get_level_values("variable").unique()
        if v.split("|")[1] not in region_sector_species
    }

    world_variables = tuple(
        sorted(
            {
                *variables[regions == "World"],
                CO2_AFOLU_RAW_NAME,
                *global_only_species_variables,
            }
        )
    )

    return world_variables, model_region_variables


def get_used_locator(df: pd.DataFrame) -> np.typing.NDArray[np.bool_]:
    """
    Get a locator for the timeseries which are used in the workflow

    These are the timeseries of the variables from [get_used_variables][(m).]
    in the regions in which they are used
    (with model regions identified as in [get_model_prefix][(m).]).

    Parameters
    ----------
    df
        Raw data for a single model

    Returns
    -------
    :
        Locator for the timeseries in `df` which are used in the workflow
    """
    model_l = df.index.get_level_values("model").unique()
    if len(model_l) != 1:
        msg = f"`df` must contain data for a single model. Received {model_l=}"
        raise ValueError(msg)

    model_prefix = get_model_prefix(model_l[0])
    world_variables, model_region_variables = get_used_variables()

    variables = df.index.get_level_values("variable")
    regions = df.index.get_level_values("region")

    res: np.typing.NDArray[np.bool_] = np.asarray(
        ((regions == "World") & variables.isin(world_variables))
        | (
            regions.str.startswith(model_prefix)
            & variables.isin(model_region_variables)
        )
    )

    return res


def apply_model_specific_fixes(model_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Apply the model-specific fixes that were applied in CMIP7 ScenarioMIP

    These fix known issues with the reporting of specific models.

    Parameters
    ----------
    model_df
        Data for a single model

    model
        Model

    Returns
    -------
    :
        Data with the fixes applied
    """
    pix = import_pandas_indexing("apply_model_specific_fixes")

    res = model_df

    if model.startswith("AIM"):
        # AIM accidentally reports zeroes between model years for Carbon Removal,
        # so only keep the model years (every five years) for Carbon Removal
        # (the zeroes are then replaced by interpolation).
        res = pd.concat(
            [
                res.loc[pix.ismatch(variable=["Emissions**"])]
                .loc[:, 2015:2100:1]
                .dropna(how="all", axis="columns"),
                res.loc[pix.ismatch(variable=["Carbon Removal**"])]
                .loc[:, 2015:2100:5]
                .dropna(how="all", axis="columns"),
            ]
        )

    if model.startswith("GCAM"):
        # GCAM reports 'Australia and New Zealand' for some scenarios
        # and 'Australia_and New Zealand' for others.
        res = update_index_levels_func(
            res,
            {"region": lambda x: x.replace("Australia_and", "Australia and")},
        )

    return res


def drop_unexpected_reporting(model_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Drop reporting that we don't expect, as was done in CMIP7 ScenarioMIP

    Parameters
    ----------
    model_df
        Data for a single model

    model
        Model

    Returns
    -------
    :
        Data without the unexpected reporting
    """
    pix = import_pandas_indexing("drop_unexpected_reporting")

    if model.startswith("GCAM"):
        # GCAM reports sub-categories of C2F6 that we've never heard of
        res: pd.DataFrame = model_df.loc[~pix.ismatch(variable="Emissions|C2F6|**")]

        return res

    return model_df


def assume_zero_for_missing_reporting(
    model_df: pd.DataFrame, reaggregator: ReaggregatorBasic
) -> pd.DataFrame:
    """
    Assume zero for any required timeseries that aren't reported

    Parameters
    ----------
    model_df
        Data for a single model

    reaggregator
        Re-aggregator that will be used during pre-processing

    Returns
    -------
    :
        `model_df`, with zeros for any required timeseries that weren't reported
    """
    pix = import_pandas_indexing("assume_zero_for_missing_reporting")

    required_index = get_required_timeseries_index(
        model_regions=reaggregator.model_regions,
        world_region=reaggregator.world_region,
        region_level=reaggregator.region_level,
        variable_level=reaggregator.variable_level,
    )

    # Guess units based on the units used for each species
    variable_unit_map = {
        "|".join(v.split("|")[:2]): u
        for v, u in model_df.index.droplevel(
            model_df.index.names.difference(["variable", "unit"])  # type: ignore # pandas-stubs confused
        )
        .drop_duplicates()
        .to_list()
    }

    def guess_unit(v_in: str) -> str | None:
        """Guess the unit of a given variable"""
        for k, v in variable_unit_map.items():
            if v_in.startswith(f"{k}|") or v_in == k:
                return v  # type: ignore # pandas-stubs confused

        return None

    res_l = []
    for (model, scenario), sdf in model_df.groupby(["model", "scenario"]):
        missing = get_missing_levels(
            sdf.index,  # type: ignore # pandas-stubs confused
            required_index,
            unit_col=reaggregator.unit_level,
        )

        zeros = pd.DataFrame(
            np.zeros((missing.shape[0], sdf.shape[1])),
            columns=sdf.columns,
            index=update_levels_from_other(
                missing, {reaggregator.unit_level: ("variable", guess_unit)}
            ),
        )
        zeros = set_index_levels_func(
            zeros, {"model": model, "scenario": scenario}
        ).reorder_levels(sdf.index.names)

        res_l.append(pix.concat([sdf, zeros]))

    res: pd.DataFrame = pix.concat(res_l)

    return res


def override_co2_afolu_with_reported_global_values(
    pre_processed: CMIP7ScenarioMIPPreProcessingResult, model_df: pd.DataFrame
) -> CMIP7ScenarioMIPPreProcessingResult:
    """
    Override the global CO2 AFOLU emissions with the globally reported values

    This is done even if the globally reported values
    aren't consistent with the region-sector reporting,
    as was done in CMIP7 ScenarioMIP.

    Parameters
    ----------
    pre_processed
        Pre-processing result

    model_df
        Data that was pre-processed

    Returns
    -------
    :
        Pre-processing result with CO2 AFOLU overridden
    """
    pix = import_pandas_indexing("override_co2_afolu_with_reported_global_values")

    reported = model_df.loc[pix.isin(variable=CO2_AFOLU_RAW_NAME, region="World")]

    def override(df: pd.DataFrame, variable: str) -> pd.DataFrame:
        res: pd.DataFrame = pix.concat(
            [
                df.loc[~pix.isin(variable=variable)],
                pix.assignlevel(reported, variable=variable),
            ]
        )

        return res

    return CMIP7ScenarioMIPPreProcessingResult(
        assumed_zero_emissions=pre_processed.assumed_zero_emissions,
        gridding_workflow_emissions=pre_processed.gridding_workflow_emissions,
        global_workflow_emissions=override(
            pre_processed.global_workflow_emissions, CO2_AFOLU_GCAGES_NAME
        ),
        global_workflow_emissions_raw_names=override(
            pre_processed.global_workflow_emissions_raw_names, CO2_AFOLU_RAW_NAME
        ),
    )


def combine_pre_processing_results(
    results: list[CMIP7ScenarioMIPPreProcessingResult],
) -> CMIP7ScenarioMIPPreProcessingResult:
    """
    Combine pre-processing results

    Parameters
    ----------
    results
        Results to combine

    Returns
    -------
    :
        Combined result
    """
    pix = import_pandas_indexing("combine_pre_processing_results")

    assumed_zero_l = [
        r.assumed_zero_emissions
        for r in results
        if r.assumed_zero_emissions is not None
    ]

    return CMIP7ScenarioMIPPreProcessingResult(
        assumed_zero_emissions=pix.concat(assumed_zero_l) if assumed_zero_l else None,
        gridding_workflow_emissions=pix.concat(
            [r.gridding_workflow_emissions for r in results]
        ),
        global_workflow_emissions=pix.concat(
            [r.global_workflow_emissions for r in results]
        ),
        global_workflow_emissions_raw_names=pix.concat(
            [r.global_workflow_emissions_raw_names for r in results]
        ),
    )


@define
class CMIP7ScenarioMIPWorkflowPreProcessor:
    """
    Pre-processor for the CMIP7 ScenarioMIP workflow

    For each model, we:

    1. keep data from 2015 to 2100 (inclusive) only
    1. (optionally) apply model-specific fixes to the reporting
       ([apply_model_specific_fixes][(m).]
       and [drop_unexpected_reporting][(m).])
    1. interpolate the data to fill gaps in the reporting
       (this ensures that the CDR-emissions correction works
       even if Carbon Removal data is reported at a different time resolution)
    1. (optionally) assume zero for any required timeseries that aren't reported
       ([assume_zero_for_missing_reporting][(m).])
    1. pre-process with
       [CMIP7ScenarioMIPPreProcessor][gcages.cmip7_scenariomip.pre_processing.CMIP7ScenarioMIPPreProcessor]
    1. (optionally) override the global CO2 AFOLU emissions
       with the globally reported values
       ([override_co2_afolu_with_reported_global_values][(m).])

    The model's regions are the regions
    which start with the first word of the model's name
    ([get_model_prefix][(m).]).
    """

    apply_model_specific_fixes: bool = True
    """Should model-specific fixes be applied?"""

    assume_zero_for_missing_reporting: bool = True
    """Should we assume zero for any required timeseries that aren't reported?"""

    use_reported_global_co2_afolu: bool = True
    """
    Should the global CO2 AFOLU emissions be the globally reported values?

    If `False`, the global CO2 AFOLU emissions are the sum
    of the region-sector reporting.
    """

    run_checks: bool = False
    """
    Should [CMIP7ScenarioMIPPreProcessor][gcages.cmip7_scenariomip.pre_processing.CMIP7ScenarioMIPPreProcessor] run checks?

    These were turned off in CMIP7 ScenarioMIP,
    so that pre-processing passes even if reporting is inconsistent.
    """  # noqa: E501

    progress: bool = True
    """Should progress bars be shown?"""

    def __call__(
        self, in_emissions: pd.DataFrame
    ) -> CMIP7ScenarioMIPPreProcessingResult:
        """
        Pre-process

        Parameters
        ----------
        in_emissions
            Raw emissions to pre-process

            These can include data from more than one model.

        Returns
        -------
        :
            Pre-processed emissions
        """
        return combine_pre_processing_results(
            [
                self.pre_process_model(mdf, model=str(model))
                for model, mdf in in_emissions.groupby("model")
            ]
        )

    def pre_process_model(
        self, model_df: pd.DataFrame, model: str
    ) -> CMIP7ScenarioMIPPreProcessingResult:
        """
        Pre-process data for a single model

        Parameters
        ----------
        model_df
            Raw emissions for a single model

        model
            Model

        Returns
        -------
        :
            Pre-processed emissions
        """
        model_df = model_df.loc[:, 2015:2100].dropna(how="all", axis="columns")
        if model_df.empty:
            msg = f"No data between 2015 and 2100 for {model=}"
            raise ValueError(msg)

        model_df.columns.name = "year"

        if self.apply_model_specific_fixes:
            model_df = apply_model_specific_fixes(model_df, model=model)

        model_df = model_df.T.interpolate(method="index").T

        pix = import_pandas_indexing("CMIP7ScenarioMIPWorkflowPreProcessor")

        model_prefix = get_model_prefix(model)
        model_regions = tuple(
            r for r in pix.uniquelevel(model_df, "region") if r.startswith(model_prefix)
        )
        if not model_regions:
            msg = f"No regions start with {model_prefix=} ({model=})"
            raise ValueError(msg)

        reaggregator = ReaggregatorBasic(model_regions=model_regions)

        if self.assume_zero_for_missing_reporting:
            model_df = assume_zero_for_missing_reporting(
                model_df, reaggregator=reaggregator
            )

        if self.apply_model_specific_fixes:
            model_df = drop_unexpected_reporting(model_df, model=model)

        pre_processor = CMIP7ScenarioMIPPreProcessor(
            reaggregator=reaggregator,
            run_checks=self.run_checks,
            progress=self.progress,
            # Pre-processing is always done in serial, as in CMIP7 ScenarioMIP.
            # Running in parallel doesn't help much here
            # (we already pre-process one model at a time)
            # and doesn't work at the moment,
            # because the re-aggregator's tolerances are pint quantities
            # which can't be unpickled in the worker processes
            # (the worker processes don't use the unit registry
            # that defines e.g. 'BC').
            n_processes=None,
        )
        res = pre_processor(model_df)

        if self.use_reported_global_co2_afolu:
            res = override_co2_afolu_with_reported_global_values(res, model_df=model_df)

        return res
