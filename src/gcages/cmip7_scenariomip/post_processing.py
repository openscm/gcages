"""
Post-processing configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import multiprocessing
from typing import cast

import numpy as np
import pandas as pd
from attrs import define
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.index_manipulation import (
    update_index_levels_func,
)

from gcages.ar6.post_processing import (
    categorise_scenarios,
    get_exceedance_probabilities,
    get_exceedance_probabilities_over_time,
    get_temperatures_in_line_with_assessment,
)
from gcages.index_manipulation import set_new_single_value_levels
from gcages.post_processing import PostProcessingResult


@define
class CMIP7ScenarioMIPPostProcessor:
    """
    CMIP7 ScenarioMIP fast-track post-processor
    """

    gsat_variable_name: str
    """The name of the GSAT variable"""

    gsat_in_line_with_assessment_variable_name: str
    """The name of the GSAT variable once its been aligned with the assessment"""

    gsat_assessment_median: float
    """
    Median of the GSAT assessment
    """

    gsat_assessment_time_period: tuple[int, ...]
    """
    Time period over which the GSAT assessment applies
    """

    gsat_assessment_pre_industrial_period: tuple[int, ...]
    """
    Pre-industrial time period used for the GSAT assessment
    """

    percentiles_to_calculate: tuple[float, ...] = (0.05, 0.33, 0.5, 0.67, 0.95)
    """Percentiles to calculate and include in the output"""

    exceedance_global_warming_levels: tuple[float, ...] = (1.5, 2.0, 2.5)
    """
    Global-warming levels against which to calculate exceedance probabilities
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    n_processes: int = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to 1 to process in serial.
    """

    def __call__(self, in_df: pd.DataFrame) -> PostProcessingResult:
        """
        Do the post-processing

        Parameters
        ----------
        in_df
            Data to post-process

        Returns
        -------
        :
            Post-processed results
        """
        if self.run_checks:
            self._check_in_df(in_df)

        temperatures_in_line_with_assessment = update_index_levels_func(
            get_temperatures_in_line_with_assessment(
                in_df.loc[
                    in_df.index.get_level_values("variable") == self.gsat_variable_name
                ],
                assessment_median=self.gsat_assessment_median,
                assessment_time_period=self.gsat_assessment_time_period,
                assessment_pre_industrial_period=self.gsat_assessment_pre_industrial_period,
                group_cols=["climate_model", "model", "scenario"],
            ),
            {"variable": lambda x: self.gsat_in_line_with_assessment_variable_name},
        )

        # Quantiles
        temperatures_in_line_with_assessment_quantiles = (
            fix_index_name_after_groupby_quantile(
                groupby_except(
                    temperatures_in_line_with_assessment,
                    "run_id",
                ).quantile(list(self.percentiles_to_calculate)),  # type: ignore # pandas-stubs confused
                new_name="quantile",
            )
        )

        # Exceedance probabilities, peak warming and categorisation
        exceedance_probabilities_over_time = get_exceedance_probabilities_over_time(
            temperatures_in_line_with_assessment,
            exceedance_thresholds_of_interest=self.exceedance_global_warming_levels,
            group_cols=["model", "scenario", "climate_model"],
            unit_col="unit",
            groupby_except_levels="run_id",
        )
        exceedance_probabilities = get_exceedance_probabilities(
            temperatures_in_line_with_assessment,
            exceedance_thresholds_of_interest=self.exceedance_global_warming_levels,
            group_cols=["model", "scenario", "climate_model"],
            unit_col="unit",
            groupby_except_levels="run_id",
        )

        # Peak Warming
        peak_warming_df = set_new_single_value_levels(
            temperatures_in_line_with_assessment.max(axis="columns").to_frame("value"),
            {"metric": "max"},
        )
        peak_warming_quantiles_df = fix_index_name_after_groupby_quantile(
            groupby_except(peak_warming_df, "run_id").quantile(
                np.array(self.percentiles_to_calculate)
            ),
            new_name="quantile",
        )
        # Extract Series for categorization and final result
        peak_warming_quantiles = peak_warming_quantiles_df["value"]

        # EOC Warming
        eoc_warming_df = set_new_single_value_levels(
            temperatures_in_line_with_assessment[2100].to_frame("value"),
            {"metric": 2100},
        )
        eoc_warming_quantiles_df = fix_index_name_after_groupby_quantile(
            groupby_except(eoc_warming_df, "run_id").quantile(
                np.array(self.percentiles_to_calculate)
            ),
            new_name="quantile",
        )
        eoc_warming_quantiles = eoc_warming_quantiles_df["value"]

        # Peak Year
        peak_warming_year_df = set_new_single_value_levels(
            update_index_levels_func(
                temperatures_in_line_with_assessment.idxmax(axis="columns").to_frame(
                    "value"
                ),
                {"unit": lambda x: "yr"},
            ),
            {"metric": "max_year"},
        )
        peak_warming_year_quantiles_df = fix_index_name_after_groupby_quantile(
            groupby_except(peak_warming_year_df, "run_id").quantile(
                np.array(self.percentiles_to_calculate)
            ),
            new_name="quantile",
        )
        peak_warming_year_quantiles = peak_warming_year_quantiles_df["value"]

        # Categorisation
        categories = categorise_scenarios(
            peak_warming_quantiles=cast(pd.DataFrame, peak_warming_quantiles),
            eoc_warming_quantiles=cast(pd.DataFrame, eoc_warming_quantiles),
            group_levels=["climate_model", "model", "scenario"],
            quantile_level="quantile",
        )

        # Metadata Compilation
        metadata_run_id = pd.concat(
            [
                peak_warming_df["value"],
                eoc_warming_df["value"],
                peak_warming_year_df["value"],
            ]
        )
        metadata_quantile = pd.concat(
            [peak_warming_quantiles, eoc_warming_quantiles, peak_warming_year_quantiles]
        )

        # Compile climate output result
        timeseries_run_id = pd.concat([temperatures_in_line_with_assessment])
        timeseries_quantile = pd.concat(
            [temperatures_in_line_with_assessment_quantiles]
        )
        timeseries_exceedance_probabilities = pd.concat(
            [exceedance_probabilities_over_time]
        )

        metadata_exceedance_probabilities = exceedance_probabilities
        metadata_categories = categories

        res = PostProcessingResult(
            timeseries_run_id=timeseries_run_id,
            timeseries_quantile=timeseries_quantile,
            timeseries_exceedance_probabilities=timeseries_exceedance_probabilities,
            metadata_run_id=metadata_run_id,
            metadata_quantile=metadata_quantile,
            metadata_exceedance_probabilities=metadata_exceedance_probabilities,
            metadata_categories=metadata_categories,
        )

        return res

    @classmethod
    def from_cmip7_scenariomip_config(cls) -> CMIP7ScenarioMIPPostProcessor:
        """
        Initialise from the config used in CMIP7 ScenarioMIP

        Returns
        -------
        :
            Initialised post-processor
        """
        return cls(
            gsat_variable_name="Surface Air Temperature Change",
            gsat_in_line_with_assessment_variable_name="Surface Temperature (GSAT)",
            gsat_assessment_median=0.85,
            gsat_assessment_time_period=tuple(range(1995, 2014 + 1)),
            gsat_assessment_pre_industrial_period=tuple(range(1850, 1900 + 1)),
            percentiles_to_calculate=(
                0.05,
                0.10,
                1.0 / 6.0,
                0.33,
                0.5,
                0.67,
                5.0 / 6.0,
                0.90,
                0.95,
            ),
            exceedance_global_warming_levels=(1.0, 4.01, 0.5),
            run_checks=True,
        )

    def _check_in_df(self, in_df: pd.DataFrame) -> None:
        """
        Perform checks on the input DataFrame
        """
        # Check for known variable names
        # Ensure that the variable we expect to process is actually present
        available_vars = in_df.index.get_level_values("variable").unique()
        if self.gsat_variable_name not in available_vars:
            msg_tuple = (
                f"Required variable '{self.gsat_variable_name}' not found in input.",
                f" Available variables: {available_vars.tolist()}",
            )
            raise ValueError(msg_tuple)

        # Check for usable time axis
        # Ensure columns are integers (years) and not empty
        if in_df.columns.empty:
            msg = "Input DataFrame has no time columns."
            raise ValueError(msg)

        try:
            # Check if all columns can be treated as integers
            years = in_df.columns.astype(int)
        except (ValueError, TypeError):
            msg_tuple = (
                "Input columns must be integer years. ",
                f"Found: {in_df.columns.tolist()}",
            )
            raise ValueError(msg_tuple)

        # Ensure the time axis covers the required assessment periods
        required_years = set(self.gsat_assessment_time_period) | set(
            self.gsat_assessment_pre_industrial_period
        )
        missing_years = required_years - set(years)
        if missing_years:
            msg_years = (
                "Input data is missing years required for assessment:",
                f"{sorted(list(missing_years))}",
            )
            raise ValueError(msg_years)

        # Check if metadata is appropriate/usable
        # Check for required index levels that are used in grouping/processing
        required_levels = ["model", "scenario", "climate_model", "run_id", "unit"]
        missing_levels = [
            level for level in required_levels if level not in in_df.index.names
        ]
        if missing_levels:
            msg_level = (
                f"Input index is missing required metadata levels: {missing_levels}"
            )
            raise ValueError(msg_level)

        # Ensure there are no NaNs in the essential grouping metadata
        for level in ["model", "scenario", "run_id"]:
            if in_df.index.get_level_values(level).isnull().any():
                msg_level = f"Found NaN values in required metadata level: '{level}'"
                raise ValueError(msg_level)
