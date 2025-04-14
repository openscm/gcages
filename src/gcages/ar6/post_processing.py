"""
Post-processing part of the AR6 workflow
"""

from __future__ import annotations

import multiprocessing

import numpy as np
import pandas as pd
import pandas_indexing as pix
from attrs import define
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)
from pandas_openscm.index_manipulation import update_index_levels_func


def get_temperatures_in_line_with_assessment(
    raw_temperatures: pd.DataFrame,
    assessment_median: float,
    assessment_time_period: tuple[int, ...],
    assessment_pre_industrial_period: tuple[int, ...],
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


    Returns
    -------
    :
        Temperatures,
        adjusted so their medians are in line with the historical assessment.
    """
    assert False, "Move to pandas-openscm"
    pre_industrial_mean = raw_temperatures.loc[
        :, list(assessment_pre_industrial_period)
    ].mean(axis="columns")
    rel_pi_temperatures = raw_temperatures.subtract(pre_industrial_mean, axis="rows")

    assessment_period_median = (
        rel_pi_temperatures.loc[:, list(assessment_time_period)]
        .mean(axis="columns")
        .groupby(["climate_model", "model", "scenario"])
        .median()
    )
    res = (
        rel_pi_temperatures.subtract(assessment_period_median, axis="rows")
        + assessment_median
    )
    # Checker:
    # res.loc[:, list(assessment_time_period)].mean(axis="columns").groupby( ["model", "scenario"]).median()  # noqa: E501

    return res


def get_temperatures_in_line_with_assessment_percentiles(
    temperatures_in_line_with_assessment: pd.DataFrame,
    percentiles_of_interest: tuple[float, ...],
) -> pd.DataFrame:
    quantiles_of_interest = [v / 100.0 for v in percentiles_of_interest]

    temperatures_in_line_with_assessment_percentiles = (
        fix_index_name_after_groupby_quantile(
            groupby_except(
                temperatures_in_line_with_assessment,
                "run_id",
            ).quantile(quantiles_of_interest),
            new_name="quantile",
        )
    )
    assert False, "add convert_quantile_to_percentile to pandas-openscm"
    temperatures_in_line_with_assessment_percentiles = (
        temperatures_in_line_with_assessment_percentiles.reset_index("quantile")
    )
    temperatures_in_line_with_assessment_percentiles["percentile"] = (
        (100 * temperatures_in_line_with_assessment_percentiles["quantile"])
        .round(1)
        .astype(str)
    )
    temperatures_in_line_with_assessment_percentiles = (
        temperatures_in_line_with_assessment_percentiles.drop(
            "quantile", axis="columns"
        ).set_index("percentile", append=True)
    )

    return temperatures_in_line_with_assessment_percentiles


def get_exceedance_probabilities_over_time(
    temperatures_in_line_with_assessment: pd.DataFrame,
    exceedance_thresholds_of_interest: tuple[float, ...],
) -> pd.DataFrame:
    assert False, "Move to pandas-openscm"
    n_runs_per_scenario = temperatures_in_line_with_assessment.groupby(
        ["model", "scenario"]
    ).count()

    exceedance_probs_l = []
    for thresh in exceedance_thresholds_of_interest:
        exceedance_prob_transient = (
            groupby_except(temperatures_in_line_with_assessment > thresh, "run_id")
            .sum()
            .divide(n_runs_per_scenario)
            * 100
        ).pix.assign(
            variable="Exceedance probability",
            threshold=thresh,
            unit="%",
        )

        exceedance_probs_l.append(exceedance_prob_transient)

    exceedance_probs = pd.concat(exceedance_probs_l)

    return exceedance_probs


def get_peak_warming(
    temperatures_in_line_with_assessment: pd.DataFrame,
    quantiles_of_interest: tuple[float, ...],
) -> pd.DataFrame:
    """
    Get peak warming

    Parameters
    ----------
    temperatures_in_line_with_assessment
        Temperatures in line with the historical assessment

    quantiles_of_interest
        Peak warming quantiles of interest

    Returns
    -------
    :
        Peak warming quantiles for each model-scenario
    """
    assert False, "Move to pandas-openscm"
    if len(temperatures_in_line_with_assessment.pix.unique("climate_model")) > 1:
        raise NotImplementedError

    peak_warming_quantiles = fix_index_name_after_groupby_quantile(
        groupby_except(
            # TODO: make run_id a parameter
            temperatures_in_line_with_assessment.max(axis="columns"),
            "run_id",
        ).quantile(quantiles_of_interest),
        new_name="quantile",
    ).pix.format(variable="Peak {variable}")

    return peak_warming_quantiles


def get_time_warming(
    temperatures_in_line_with_assessment: pd.DataFrame,
    time: float | int,
    quantiles_of_interest: tuple[float, ...],
) -> pd.DataFrame:
    """
    Get warming at a given time

    Parameters
    ----------
    temperatures_in_line_with_assessment
        Temperatures in line with the historical assessment

    time
        Time at which to get the warming

    quantiles_of_interest
        Peak warming quantiles of interest

    Returns
    -------
    :
        Peak warming quantiles for each model-scenario
    """
    assert False, "Move to pandas-openscm"
    if len(temperatures_in_line_with_assessment.pix.unique("climate_model")) > 1:
        raise NotImplementedError

    time_warming_quantiles = (
        fix_index_name_after_groupby_quantile(
            groupby_except(
                # TODO: make run_id a parameter
                temperatures_in_line_with_assessment[time],
                "run_id",
            ).quantile(quantiles_of_interest),
            new_name="quantile",
        )
        .pix.assign(time=time)
        .pix.format(variable="{time} {variable}", drop=True)
    )

    return time_warming_quantiles


def get_exceedance_probabilities(
    temperatures_in_line_with_assessment: pd.DataFrame,
    exceedance_thresholds_of_interest: tuple[float, ...],
) -> pd.DataFrame:
    """
    Get exceedance probabilities

    Parameters
    ----------
    temperatures_in_line_with_assessment
        Temperatures in line with the historical assessment

    time
        Time at which to get the warming

    quantiles_of_interest
        Peak warming quantiles of interest

    Returns
    -------
    :
        Peak warming quantiles for each model-scenario
    """
    assert False, "Move to pandas-openscm"
    if len(temperatures_in_line_with_assessment.pix.unique("climate_model")) > 1:
        raise NotImplementedError

    peak_warming = temperatures_in_line_with_assessment.max(axis="columns")
    # This is the better way to do this
    n_runs_per_scenario = temperatures_in_line_with_assessment.index.droplevel(
        temperatures_in_line_with_assessment.index.names.difference(
            ["model", "scenario"]
        )
    ).value_counts()

    exceedance_probs_l = []
    for thresh in exceedance_thresholds_of_interest:
        exceedance_prob = (
            # TODO: make run_id injectable
            groupby_except(peak_warming > thresh, "run_id")
            .sum()
            .divide(n_runs_per_scenario)
            * 100
        ).pix.assign(
            variable="Exceedance probability",
            threshold=thresh,
            unit="%",
        )

        exceedance_probs_l.append(exceedance_prob)

    exceedance_probs = pd.concat(exceedance_probs_l)

    return exceedance_probs


def categorise_scenarios(
    peak_warming_quantiles: pd.DataFrame,
    eoc_warming_quantiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Categorise scenarios

    Parameters
    ----------
    temperatures_in_line_with_assessment
        Temperatures in line with the historical assessment

    Returns
    -------
    :
        Scenario categorisation
    """
    assert False, "Think about the levels more carefully here rather than hard-coding"
    index = peak_warming_quantiles.index.droplevel(
        peak_warming_quantiles.index.names.difference(["model", "scenario"])
    ).unique()

    peak_warming_quantiles_use = peak_warming_quantiles.reset_index(
        peak_warming_quantiles.index.names.difference(
            ["model", "scenario", "quantile"]
        ),
        drop=True,
    ).unstack("quantile")
    eoc_warming_quantiles_use = eoc_warming_quantiles.reset_index(
        eoc_warming_quantiles.index.names.difference(["model", "scenario", "quantile"]),
        drop=True,
    ).unstack("quantile")

    categories = pd.Series(
        "C8: exceed warming of 4°C (>=50%)",
        index=index,
        name="category_name",
    )
    categories[peak_warming_quantiles_use[0.5] < 4.0] = (  # noqa: PLR2004
        "C7: limit warming to 4°C (>50%)"
    )
    categories[peak_warming_quantiles_use[0.5] < 3.0] = (  # noqa: PLR2004
        "C6: limit warming to 3°C (>50%)"
    )
    categories[peak_warming_quantiles_use[0.5] < 2.5] = (  # noqa: PLR2004
        "C5: limit warming to 2.5°C (>50%)"
    )
    categories[peak_warming_quantiles_use[0.5] < 2.0] = (  # noqa: PLR2004
        "C4: limit warming to 2°C (>50%)"
    )
    categories[peak_warming_quantiles_use[0.67] < 2.0] = (  # noqa: PLR2004
        "C3: limit warming to 2°C (>67%)"
    )
    categories[
        (peak_warming_quantiles_use[0.33] > 1.5)  # noqa: PLR2004
        & (eoc_warming_quantiles_use[0.5] < 1.5)  # noqa: PLR2004
    ] = "C2: return warming to 1.5°C (>50%) after a high overshoot"
    categories[
        (peak_warming_quantiles_use[0.33] <= 1.5)  # noqa: PLR2004
        & (eoc_warming_quantiles_use[0.5] < 1.5)  # noqa: PLR2004
    ] = "C1: limit warming to 1.5°C (>50%) with no or limited overshoot"

    out = categories.to_frame()
    out["category"] = out["category_name"].apply(lambda x: x.split(":")[0])

    return out


# TODO: move this to gcages.post_processing
@define
class PostProcessingResult:
    """
    Results of post-processing
    """


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

    raw_gsat_variable: str
    """
    Name of the output variable that contains raw temperature output

    The temperature output should be global-mean surface air temperature (GSAT).
    """

    assessed_gsat_variable: str
    """
    Name of the output variable that will contain temperature output

    This temperature output is in line with the (AR6) assessment.
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
            breakpoint()
            # TODO:
            #   - enable optional checks for:
            #       - only known variable names are in the output
            #       - only data with a useable time axis is in there
            #       - metadata is appropriate/usable
            #       - raw variable we expect is in there

        temperatures_in_line_with_assessment = update_index_levels_func(
            get_temperatures_in_line_with_assessment(
                in_df.loc[pix.isin(variable=[self.raw_gsat_variable])],
                assessment_median=self.gsat_assessment_median,
                assessment_time_period=self.gsat_assessment_time_period,
                assessment_pre_industrial_period=self.gsat_assessment_pre_industrial_period,
            ),
            {"variable": lambda x: self.assessed_gsat_variable},
        )
        temperatures_in_line_with_assessment_percentiles = (
            get_temperatures_in_line_with_assessment_percentiles(
                temperatures_in_line_with_assessment,
                percentiles_of_interest=tuple(
                    v * 100 for v in self.quantiles_of_interest
                ),
            )
        )

        exceedance_probabilities_over_time = get_exceedance_probabilities_over_time(
            temperatures_in_line_with_assessment,
            exceedance_thresholds_of_interest=self.exceedance_thresholds_of_interest,
        )

        timeseries_l = [
            temperatures_in_line_with_assessment_percentiles,
            exceedance_probabilities_over_time,
        ]

        timeseries = pix.concat(timeseries_l)
        timeseries.columns = timeseries.columns.astype(int)

        peak_warming = get_peak_warming(
            temperatures_in_line_with_assessment,
            quantiles_of_interest=self.quantiles_of_interest,
        )
        eoc_warming = get_time_warming(
            temperatures_in_line_with_assessment,
            time=2100,
            quantiles_of_interest=self.quantiles_of_interest,
        )
        exceedance_probabilities = get_exceedance_probabilities(
            temperatures_in_line_with_assessment,
            exceedance_thresholds_of_interest=self.exceedance_thresholds_of_interest,
        )
        categories = categorise_scenarios(
            peak_warming_quantiles=peak_warming,
            eoc_warming_quantiles=eoc_warming,
        )

        metadata_l = [peak_warming, eoc_warming, exceedance_probabilities, categories]
        metadata = pd.concat(metadata_l, axis="columns")

        if self.run_checks:
            breakpoint()
            # TODO:
            #   - enable optional checks for:
            #       - input and output scenarios are the same

        return PostProcessingResult(timeseries=timeseries, metadata=metadata)

    @classmethod
    def from_ar6_config(  # noqa: PLR0913
        cls,
        exceedance_thresholds_of_interest: tuple[float, ...] = tuple(
            np.arange(1.0, 4.01, 0.5)
        ),
        percentiles_of_interest: tuple[float, ...] = (
            5.0,
            10.0,
            100.0 / 6,
            33.0,
            50.0,
            67.0,
            100.0 * 5.0 / 6,
            90.0,
            95.0,
        ),
        raw_gsat_variable: str = (
            "AR6 climate diagnostics|Raw Surface Temperature (GSAT)"
        ),
        assessed_gsat_variable: str = (
            "AR6 climate diagnostics|Surface Temperature (GSAT)"
        ),
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

        percentiles_of_interest
            The quantiles for which we are interested in percentiles

        raw_gsat_variable
            Name of the output variable that contains raw temperature output

            The temperature output should be global-mean surface air temperature (GSAT).

        assessed_gsat_variable
            Name of the output variable that will contain temperature output

            This temperature output is in line with the (AR6) assessment.

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
        assert False, "Change percentiles_of_interest to quantiles_of_interest"
        if not all(p in percentiles_of_interest for p in [50.0, 33.0]):
            msg = (
                "percentiles_of_interest must contain 50.0 and 33.0, "
                f"received {percentiles_of_interest=}"
            )
            raise AssertionError(msg)

        return cls(
            raw_gsat_variable=raw_gsat_variable,
            assessed_gsat_variable=assessed_gsat_variable,
            gsat_assessment_median=0.85,
            gsat_assessment_time_period=tuple(range(1995, 2014 + 1)),
            gsat_assessment_pre_industrial_period=tuple(range(1850, 1900 + 1)),
            percentiles_of_interest=percentiles_of_interest,
            exceedance_thresholds_of_interest=exceedance_thresholds_of_interest,
            run_checks=run_checks,
            n_processes=n_processes,
        )
