"""
Post-processing configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import multiprocessing

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
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in the output
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

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
                ).quantile(self.percentiles_to_calculate),  # type: ignore # pandas-stubs confused
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

        peak_warming = set_new_single_value_levels(
            temperatures_in_line_with_assessment.max(axis="columns"), {"metric": "max"}
        )
        # Moving to frame for typing
        peak_warming_f = peak_warming.to_frame().copy()

        # Rebuild proper MultiIndex columns BEFORE your helper
        peak_warming_f.columns = pd.MultiIndex.from_tuples(
            [("max",)],  # placeholder, will be replaced
            names=temperatures_in_line_with_assessment.columns.names,
        )

        peak_warming_f = set_new_single_value_levels(
            peak_warming_f,
            {"metric": "max"},
        )
        peak_warming_quantiles = fix_index_name_after_groupby_quantile(
            groupby_except(peak_warming, "run_id").quantile(
                self.percentiles_to_calculate
            ),
            new_name="quantile",
        )

        eoc_warming = set_new_single_value_levels(
            temperatures_in_line_with_assessment[2100], {"metric": 2100}
        )
        eoc_warming_quantiles = fix_index_name_after_groupby_quantile(
            groupby_except(eoc_warming, "run_id").quantile(
                list(self.percentiles_to_calculate)
            ),
            new_name="quantile",
        )
        peak_warming_year = set_new_single_value_levels(
            update_index_levels_func(
                temperatures_in_line_with_assessment.idxmax(axis="columns"),
                {"unit": lambda x: "yr"},
            ),
            {"metric": "max_year"},
        )
        # Moving to frame for typing
        peak_warming_f = peak_warming_year.to_frame().copy()

        # Rebuild proper MultiIndex columns BEFORE your helper
        peak_warming_f.columns = pd.MultiIndex.from_tuples(
            [("max",)],  # placeholder, will be replaced
            names=temperatures_in_line_with_assessment.columns.names,
        )

        peak_warming_f = set_new_single_value_levels(
            peak_warming_f,
            {"metric": "max"},
        )
        peak_warming_year_quantiles = fix_index_name_after_groupby_quantile(
            groupby_except(peak_warming_f, "run_id").quantile(
                self.percentiles_to_calculate
            ),
            new_name="quantile",
        )

        exceedance_probabilities = get_exceedance_probabilities(
            temperatures_in_line_with_assessment,
            exceedance_thresholds_of_interest=self.exceedance_global_warming_levels,
            group_cols=["model", "scenario", "climate_model"],
            unit_col="unit",
            groupby_except_levels="run_id",
        )

        categories = categorise_scenarios(
            peak_warming_quantiles=peak_warming_quantiles,
            eoc_warming_quantiles=eoc_warming_quantiles,
            group_levels=["climate_model", "model", "scenario"],
            quantile_level="quantile",
        )

        # Compile climate output result
        timeseries_run_id = pd.concat([temperatures_in_line_with_assessment])
        timeseries_quantile = pd.concat(
            [temperatures_in_line_with_assessment_quantiles]
        )
        timeseries_exceedance_probabilities = pd.concat(
            [exceedance_probabilities_over_time]
        )

        metadata_run_id = pd.concat([peak_warming, eoc_warming, peak_warming_year])
        metadata_quantile = pd.concat(
            [
                peak_warming_quantiles,
                eoc_warming_quantiles,
                peak_warming_year_quantiles,
            ]
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

        Parameters
        ----------
        exceedance_thresholds_of_interest
            The thresholds for which we are interested in exceedance probabilities

        quantiles_of_interest
            The quantiles we want to include in the results

        raw_gsat_variable_in
            Name of the variable that contains raw temperature output in the input

            The temperature output should be global-mean surface air temperature (GSAT).

        assessed_gsat_variable
            Name of the output variable that will contain temperature output

            This temperature output is in line with the
            CMIP7 ScenarioMIP assessed historical warming.

        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        Returns
        -------
        :
            Initialised post-processor
        """
        return cls(
            gsat_variable_name="Surface Air Temperature Change",
            gsat_in_line_with_assessment_variable_name="Surface Temperature (GSAT)",
            gsat_assessment_median=0.85,
            gsat_assessment_time_period=range(1995, 2014 + 1),
            gsat_assessment_pre_industrial_period=range(1850, 1900 + 1),
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
            # TODO: implement and activate
            run_checks=False,
        )
