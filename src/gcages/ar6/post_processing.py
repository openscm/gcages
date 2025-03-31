"""
Post-processing part of the AR6 workflow
"""

from __future__ import annotations

import multiprocessing

import pandas as pd
import pandas_indexing as pix
from attrs import define


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
    rel_pi_temperatures = raw_temperatures.subtract(
        raw_temperatures.loc[:, list(assessment_pre_industrial_period)].mean(
            axis="columns"
        ),
        axis="rows",
    )
    mod_scen_medians = (
        rel_pi_temperatures.loc[:, list(assessment_time_period)]
        .mean(axis="columns")
        .groupby(["climate_model", "model", "scenario"])
        .median()
    )
    res = (
        rel_pi_temperatures.subtract(mod_scen_medians, axis="rows") + assessment_median
    )
    # Checker:
    # res.loc[:, list(assessment_time_period)].mean(axis="columns").groupby( ["model", "scenario"]).median()  # noqa: E501

    return res


def categorise_scenarios(
    temperatures_in_line_with_assessment: pd.DataFrame,
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
    if len(temperatures_in_line_with_assessment.pix.unique("climate_model")) > 1:
        raise NotImplementedError

    peak_warming_quantiles = (
        temperatures_in_line_with_assessment.max(axis="columns")
        .groupby(["model", "scenario"])
        .quantile([0.33, 0.5, 0.67])
        .unstack()
    )
    eoc_warming_quantiles = (
        temperatures_in_line_with_assessment[2100]
        .groupby(["model", "scenario"])
        .quantile([0.5])
        .unstack()
    )
    categories = pd.Series(
        "C8: exceed warming of 4°C (>=50%)",
        index=peak_warming_quantiles.index,
        name="category_name",
    )
    categories[peak_warming_quantiles[0.5] < 4.0] = "C7: limit warming to 4°C (>50%)"  # noqa: PLR2004
    categories[peak_warming_quantiles[0.5] < 3.0] = "C6: limit warming to 3°C (>50%)"  # noqa: PLR2004
    categories[peak_warming_quantiles[0.5] < 2.5] = "C5: limit warming to 2.5°C (>50%)"  # noqa: PLR2004
    categories[peak_warming_quantiles[0.5] < 2.0] = "C4: limit warming to 2°C (>50%)"  # noqa: PLR2004
    categories[peak_warming_quantiles[0.67] < 2.0] = "C3: limit warming to 2°C (>67%)"  # noqa: PLR2004
    categories[
        (peak_warming_quantiles[0.33] > 1.5) & (eoc_warming_quantiles[0.5] < 1.5)  # noqa: PLR2004
    ] = "C2: return warming to 1.5°C (>50%) after a high overshoot"
    categories[
        (peak_warming_quantiles[0.33] <= 1.5) & (eoc_warming_quantiles[0.5] < 1.5)  # noqa: PLR2004
    ] = "C1: limit warming to 1.5°C (>50%) with no or limited overshoot"

    out = categories.to_frame()
    out["category"] = out["category_name"].apply(lambda x: x.split(":")[0])

    return out


@define
class AR6PostProcessor:
    """
    Post-processor that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_like_config`][(c)]
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

    def __call__(self, in_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        # TODO: check the return type rendering
        if self.run_checks:
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in the output
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        temperatures_in_line_with_assessment = get_temperatures_in_line_with_assessment(
            in_df.loc[
                pix.isin(
                    variable=["AR6 climate diagnostics|Raw Surface Temperature (GSAT)"]
                )
            ],
            assessment_median=self.gsat_assessment_median,
            assessment_time_period=self.gsat_assessment_time_period,
            assessment_pre_industrial_period=self.gsat_assessment_pre_industrial_period,
        ).pix.assign(variable="AR6 climate diagnostics|Surface Temperature (GSAT)")

        categories = categorise_scenarios(temperatures_in_line_with_assessment)

        temperatures_in_line_with_assessment_percentiles = (
            temperatures_in_line_with_assessment.groupby(
                ["climate_model", "model", "scenario", "variable", "region", "unit"]
            ).quantile([0.05, 0.1, 1 / 6, 0.33, 0.5, 0.67, 5 / 6, 0.9, 0.95])
        )
        temperatures_in_line_with_assessment_percentiles.index.names = [
            *temperatures_in_line_with_assessment_percentiles.index.names[:-1],
            "quantile",
        ]
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
            )
            .set_index("percentile", append=True)
            .pix.format(
                variable="{variable}|{climate_model}|{percentile}th Percentile",
                drop=True,
            )
        )

        timeseries_l = [
            temperatures_in_line_with_assessment_percentiles,
        ]

        timeseries = pix.concat(timeseries_l)
        timeseries.columns = timeseries.columns.astype(int)

        metadata_l = [
            categories,
        ]
        metadata = pix.concat(metadata_l)

        # TODO:
        #   - enable optional checks for:
        #       - input and output scenarios are the same

        return timeseries, metadata

    @classmethod
    def from_ar6_config(
        cls,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> AR6PostProcessor:
        """
        Initialise from the config used in AR6

        Parameters
        ----------
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
        return cls(
            gsat_assessment_median=0.85,
            gsat_assessment_time_period=tuple(range(1995, 2014 + 1)),
            gsat_assessment_pre_industrial_period=tuple(range(1850, 1900 + 1)),
            run_checks=run_checks,
            n_processes=n_processes,
        )
