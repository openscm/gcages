"""
Post-processing part of the AR6 workflow
"""

from __future__ import annotations

import multiprocessing
from typing import TypeVar

import numpy as np
import pandas as pd
from attrs import define
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_data_for_times,
    assert_has_index_levels,
    assert_index_is_multiindex,
)
from gcages.index_manipulation import set_new_single_value_levels
from gcages.post_processing import PostProcessingResult

T = TypeVar("T")


def get_temperatures_in_line_with_assessment(
    raw_temperatures: pd.DataFrame,
    assessment_median: float,
    assessment_time_period: tuple[int, ...],
    assessment_pre_industrial_period: tuple[int, ...],
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Get temperatures in line with the historical assessment

    Parameters
    ----------
    raw_temperatures
        Raw temperatures

    assessment_median
        Median of the assessment to match

    assessment_time_period
        Time period over which the assessment applies

    assessment_pre_industrial_period
        Pre-industrial period used for the assessment

    group_cols
        Columns to use when grouping `raw_temperatures`

    Returns
    -------
    :
        Temperatures,
        adjusted so their medians are in line with the historical assessment.
    """
    # TODO: move to pandas-openscm
    pre_industrial_mean = raw_temperatures.loc[
        :, list(assessment_pre_industrial_period)
    ].mean(axis="columns")
    rel_pi_temperatures = raw_temperatures.subtract(pre_industrial_mean, axis="rows")  # type: ignore # pandas-stubs confused

    assessment_period_median = (
        rel_pi_temperatures.loc[:, list(assessment_time_period)]
        .mean(axis="columns")
        .groupby(group_cols)
        .median()
    )
    res = (
        rel_pi_temperatures.subtract(assessment_period_median, axis="rows")  # type: ignore # pandas-stubs confused
        + assessment_median
    )
    # Checker:
    # res.loc[:, list(assessment_time_period)].mean(axis="columns").groupby( ["model", "scenario"]).median()  # noqa: E501

    return res


def get_exceedance_probabilities_over_time(  # noqa: D103
    temperatures: pd.DataFrame,
    exceedance_thresholds_of_interest: tuple[float, ...],
    group_cols: list[str],
    unit_col: str,
    groupby_except_levels: list[str] | str,
) -> pd.DataFrame:
    # TODO: Move to pandas-openscm
    n_runs_per_group = temperatures.groupby(group_cols).count()

    # TODO: add get_index_value to pandas-openscm with keyword `expect_singular`
    temperature_unit_l = temperatures.index.get_level_values(unit_col).unique().tolist()
    if len(temperature_unit_l) > 1:
        raise AssertionError(temperature_unit_l)
    temperature_unit = temperature_unit_l[0]

    exceedance_probs_l = []
    for thresh in exceedance_thresholds_of_interest:
        exceedance_prob_transient = update_index_levels_func(
            groupby_except(temperatures > thresh, groupby_except_levels)
            .sum()
            .divide(n_runs_per_group)
            * 100,
            dict(
                variable=lambda x: "Exceedance probability",
                unit=lambda x: "%",
            ),
        )
        exceedance_prob_transient = set_new_single_value_levels(
            exceedance_prob_transient,
            {"threshold": thresh, "threshold_unit": temperature_unit},
            copy=False,
        )

        exceedance_probs_l.append(exceedance_prob_transient)

    exceedance_probs = pd.concat(exceedance_probs_l)

    return exceedance_probs


def get_exceedance_probabilities(  # noqa: D103
    temperatures: pd.DataFrame,
    exceedance_thresholds_of_interest: tuple[float, ...],
    group_cols: list[str],
    unit_col: str,
    groupby_except_levels: list[str] | str,
) -> pd.Series[float]:
    # TODO: Move to pandas-openscm
    # Have to do it this way because we don't want the columns
    n_runs_per_group = temperatures.index.droplevel(
        temperatures.index.names.difference(group_cols)  # type: ignore # pandas-stubs confused
    ).value_counts()

    # TODO: add get_index_value to pandas-openscm with keyword `expect_singular`
    temperature_unit_l = temperatures.index.get_level_values(unit_col).unique().tolist()
    if len(temperature_unit_l) > 1:
        raise AssertionError(temperature_unit_l)
    temperature_unit = temperature_unit_l[0]

    peak_warming = temperatures.max(axis="columns")

    exceedance_probs_l = []
    for thresh in exceedance_thresholds_of_interest:
        exceedance_prob = update_index_levels_func(
            groupby_except(peak_warming > thresh, groupby_except_levels)  # type: ignore # error in pandas-openscm
            .sum()
            .divide(n_runs_per_group)
            * 100,
            dict(
                variable=lambda x: "Exceedance probability",
                unit=lambda x: "%",
            ),
        )
        exceedance_prob = set_new_single_value_levels(
            exceedance_prob,
            {"threshold": thresh, "threshold_unit": temperature_unit},
            copy=False,
        )

        exceedance_probs_l.append(exceedance_prob)

    exceedance_probs: pd.Series[float] = pd.concat(exceedance_probs_l)  # type: ignore # pandas-stubs out of date

    return exceedance_probs


def categorise_scenarios(
    peak_warming_quantiles: pd.DataFrame,
    eoc_warming_quantiles: pd.DataFrame,
    group_levels: list[str],
    quantile_level: str,
) -> pd.Series[str]:
    """
    Categorise scenarios

    Parameters
    ----------
    peak_warming_quantiles
        Peak warming quantiles

    eoc_warming_quantiles
        End of century warming quantiles

    group_levels
        Levels of the input indexes to group the results by

        In other words, each unique combination of values in `group_levels`
        will get its own category.

        Typically, this is something like `["model", "scenario", "climate_model"]`

    quantile_level
        The level in `peak_warming_quantiles` and `eoc_warming_quantiles`
        that holds information about the quantile of each value.

    Returns
    -------
    :
        Scenario categorisation
    """
    index = peak_warming_quantiles.index.droplevel(
        peak_warming_quantiles.index.names.difference(group_levels)  # type: ignore # pandas-stubs confused
    ).unique()

    peak_warming_quantiles_use = peak_warming_quantiles.reset_index(
        peak_warming_quantiles.index.names.difference([*group_levels, quantile_level]),  # type: ignore # pandas-stubs confused
        drop=True,
    ).unstack(quantile_level)
    eoc_warming_quantiles_use = eoc_warming_quantiles.reset_index(
        eoc_warming_quantiles.index.names.difference([*group_levels, quantile_level]),  # type: ignore # pandas-stubs confused
        drop=True,
    ).unstack(quantile_level)

    category_names = pd.Series("C8: exceed warming of 4°C (>=50%)", index=index)
    category_names[peak_warming_quantiles_use[0.5] < 4.0] = (  # noqa: PLR2004
        "C7: limit warming to 4°C (>50%)"
    )
    category_names[peak_warming_quantiles_use[0.5] < 3.0] = (  # noqa: PLR2004
        "C6: limit warming to 3°C (>50%)"
    )
    category_names[peak_warming_quantiles_use[0.5] < 2.5] = (  # noqa: PLR2004
        "C5: limit warming to 2.5°C (>50%)"
    )
    category_names[peak_warming_quantiles_use[0.5] < 2.0] = (  # noqa: PLR2004
        "C4: limit warming to 2°C (>50%)"
    )
    category_names[peak_warming_quantiles_use[0.67] < 2.0] = (  # noqa: PLR2004
        "C3: limit warming to 2°C (>67%)"
    )
    category_names[
        (peak_warming_quantiles_use[0.33] > 1.5)  # noqa: PLR2004
        & (eoc_warming_quantiles_use[0.5] < 1.5)  # noqa: PLR2004
    ] = "C2: return warming to 1.5°C (>50%) after a high overshoot"
    category_names[
        (peak_warming_quantiles_use[0.33] <= 1.5)  # noqa: PLR2004
        & (eoc_warming_quantiles_use[0.5] < 1.5)  # noqa: PLR2004
    ] = "C1: limit warming to 1.5°C (>50%) with no or limited overshoot"

    category_names = set_new_single_value_levels(
        category_names, {"metric": "category_name"}, copy=False
    )
    categories = update_index_levels_func(
        category_names.apply(lambda x: x.split(":")[0]),
        {"metric": lambda x: "category"},
    )
    out: pd.Series[str] = pd.concat([category_names, categories])  # type: ignore # pandas-stubs out of date

    return out


@define
class AR6PostProcessor:
    """
    Post-processor that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_config`][(c)]
    """

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

    quantiles_of_interest: tuple[float, ...]
    """
    Quantiles to include in output
    """

    exceedance_thresholds_of_interest: tuple[float, ...]
    """
    Thresholds of interest for calculating exceedance probabilities
    """

    raw_gsat_variable_in: str
    """
    Name of the variable that contains raw temperature output in the input

    The temperature output should be global-mean surface air temperature (GSAT).
    """

    assessed_gsat_variable: str
    """
    Name of the output variable that will contain temperature output

    This temperature output is in line with the (AR6) assessed historical warming.
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    progress: bool = True
    """
    Should progress bars be shown for each operation where they make sense?
    """

    n_processes: int | None = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to `None` to process in serial.
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
        timeseries, metadata :
            Post-processed results

            These are both timeseries as well as scenario-level metadata.
        """
        if self.run_checks:
            assert_index_is_multiindex(in_df)
            assert_has_index_levels(
                in_df, ["variable", "unit", "model", "scenario", "climate_model"]
            )
            assert_data_is_all_numeric(in_df)
            assert_has_data_for_times(
                in_df, name="in_df", times=[2100], allow_nan=False
            )

            if self.raw_gsat_variable_in not in in_df.index.get_level_values(
                "variable"
            ):
                msg = (
                    f"{self.raw_gsat_variable_in} must be provided. "
                    f"Received: {in_df.index.get_level_values('variable')=}"
                )
                raise AssertionError(msg)

        temperatures_in_line_with_assessment = update_index_levels_func(
            get_temperatures_in_line_with_assessment(
                in_df.loc[
                    in_df.index.get_level_values("variable").isin(
                        [self.raw_gsat_variable_in]
                    )
                ],
                assessment_median=self.gsat_assessment_median,
                assessment_time_period=self.gsat_assessment_time_period,
                assessment_pre_industrial_period=self.gsat_assessment_pre_industrial_period,
                group_cols=["climate_model", "model", "scenario"],
            ),
            {"variable": lambda x: self.assessed_gsat_variable},
        )
        temperatures_in_line_with_assessment_quantiles = (
            fix_index_name_after_groupby_quantile(
                groupby_except(
                    temperatures_in_line_with_assessment,
                    "run_id",
                ).quantile(self.quantiles_of_interest),  # type: ignore # pandas-stubs confused
                new_name="quantile",
            )
        )
        exceedance_probabilities_over_time = get_exceedance_probabilities_over_time(
            temperatures_in_line_with_assessment,
            exceedance_thresholds_of_interest=self.exceedance_thresholds_of_interest,
            group_cols=["model", "scenario", "climate_model"],
            unit_col="unit",
            groupby_except_levels="run_id",
        )

        # TODO: move pandas-openscm.max to pandas-openscm
        peak_warming = set_new_single_value_levels(
            temperatures_in_line_with_assessment.max(axis="columns"), {"metric": "max"}
        )
        peak_warming_quantiles = fix_index_name_after_groupby_quantile(
            groupby_except(peak_warming, "run_id").quantile(self.quantiles_of_interest),  # type: ignore # pandas-stubs confused
            new_name="quantile",
        )

        eoc_warming = set_new_single_value_levels(
            temperatures_in_line_with_assessment[2100], {"metric": 2100}
        )
        eoc_warming_quantiles = fix_index_name_after_groupby_quantile(
            groupby_except(eoc_warming, "run_id").quantile(self.quantiles_of_interest),  # type: ignore # pandas-stubs confused
            new_name="quantile",
        )
        peak_warming_year = set_new_single_value_levels(
            update_index_levels_func(
                temperatures_in_line_with_assessment.idxmax(axis="columns"),  # type: ignore # error in pandas-openscm
                {"unit": lambda x: "yr"},
            ),
            {"metric": "max_year"},
        )
        peak_warming_year_quantiles = fix_index_name_after_groupby_quantile(
            groupby_except(peak_warming_year, "run_id").quantile(
                self.quantiles_of_interest  # type: ignore # pandas-stubs out of date
            ),
            new_name="quantile",
        )

        exceedance_probabilities = get_exceedance_probabilities(
            temperatures_in_line_with_assessment,
            exceedance_thresholds_of_interest=self.exceedance_thresholds_of_interest,
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

        timeseries_run_id = pd.concat([temperatures_in_line_with_assessment])
        timeseries_quantile = pd.concat(
            [temperatures_in_line_with_assessment_quantiles]
        )
        timeseries_exceedance_probabilities = pd.concat(
            [exceedance_probabilities_over_time]
        )

        metadata_run_id: pd.Series[float] = pd.concat(  # type: ignore # pandas-stubs out of date
            [peak_warming, eoc_warming, peak_warming_year]
        )
        metadata_quantile: pd.Series[float] = pd.concat(  # type: ignore # pandas-stubs out of date
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

        if self.run_checks:
            comparison_levels = ["model", "scenario", "climate_model"]
            for attr in [
                "timeseries_run_id",
                "timeseries_quantile",
                "timeseries_exceedance_probabilities",
                "metadata_run_id",
                "metadata_quantile",
                "metadata_exceedance_probabilities",
                "metadata_categories",
            ]:
                pd.testing.assert_index_equal(  # type: ignore # pandas-stubs out of date
                    getattr(res, attr)
                    .index.droplevel(
                        getattr(res, attr).index.names.difference(comparison_levels)
                    )
                    .drop_duplicates()
                    .reorder_levels(comparison_levels),
                    in_df.index.droplevel(
                        in_df.index.names.difference(comparison_levels)  # type: ignore # pandas-stubs out of date
                    )
                    .drop_duplicates()
                    .reorder_levels(comparison_levels),
                    check_order=False,
                )

        return res

    @classmethod
    def from_ar6_config(  # noqa: PLR0913
        cls,
        exceedance_thresholds_of_interest: tuple[float, ...] = tuple(
            np.arange(1.0, 4.01, 0.5)
        ),
        quantiles_of_interest: tuple[float, ...] = (
            0.05,
            0.10,
            1.0 / 6.0,
            0.33,
            0.50,
            0.67,
            5.0 / 6.0,
            0.90,
            0.95,
        ),
        raw_gsat_variable_in: str = "Surface Air Temperature Change",
        assessed_gsat_variable: str = "Surface Temperature (GSAT)",
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> AR6PostProcessor:
        """
        Initialise from the config used in AR6

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
            (AR6) assessed historical warming.

        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        progress
            Should progress bars be shown for each operation?

        n_processes
            Number of processes to use for parallel processing.

            Set to 1 to process in serial.

        Returns
        -------
        :
            Initialised post-processor
        """
        if not all(q in quantiles_of_interest for q in [0.50, 0.33]):
            msg = (
                "quantiles_of_interest must contain 0.50 and 0.33 "
                "for the categorisation to work, "
                f"received {quantiles_of_interest=}"
            )
            raise AssertionError(msg)

        return cls(
            raw_gsat_variable_in=raw_gsat_variable_in,
            assessed_gsat_variable=assessed_gsat_variable,
            gsat_assessment_median=0.85,
            gsat_assessment_time_period=tuple(range(1995, 2014 + 1)),
            gsat_assessment_pre_industrial_period=tuple(range(1850, 1900 + 1)),
            quantiles_of_interest=quantiles_of_interest,
            exceedance_thresholds_of_interest=exceedance_thresholds_of_interest,
            run_checks=run_checks,
            n_processes=n_processes,
        )
