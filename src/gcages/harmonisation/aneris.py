"""
Harmonisation using [aneris](https://aneris.readthedocs.io/)
"""

from __future__ import annotations

import multiprocessing
from typing import Any

import attr
import pandas as pd
from attrs import define, field
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

import gcages.aneris_helpers
from gcages.assertions import (
    NotAllowedMetadataValuesError,
    assert_data_is_all_numeric,
    assert_has_data_for_times,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_metadata_values_all_allowed,
)
from gcages.harmonisation.common import assert_harmonised


@define
class AnerisHarmoniser:
    """
    Harmoniser that uses [aneris](https://aneris.readthedocs.io/)
    """

    historical_emissions: pd.DataFrame = field()
    """
    Historical emissions to use for harmonisation
    """

    harmonisation_year: int
    """
    Year in which to harmonise
    """

    aneris_overrides: pd.Series[str] | None = field(default=None)
    """
    Overrides to supply to `aneris.convenience.harmonise_all`

    For source code and docs,
    see e.g. [https://github.com/iiasa/aneris/blob/v0.4.2/src/aneris/convenience.py]().
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    variable_level: str = "variable"
    """
    Level in data indexes that represents the variable of the timeseries
    """

    region_level: str = "region"
    """
    Level in data indexes that represents the region of the timeseries
    """

    unit_level: str = "unit"
    """
    Level in data indexes that represents the unit of the timeseries
    """

    scenario_group_levels: list[str] = field(factory=lambda: ["model", "scenario"])
    """
    Levels in data indexes to use to group data into scenarios

    Here, 'scenarios' means groups of timeseries
    that will be run through a climate model.
    """

    progress: bool = True
    """
    Should progress bars be shown for each operation?
    """

    n_processes: int | None = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to `None` to process in serial.
    """

    @aneris_overrides.validator
    def validate_aneris_overrides(
        self, attribute: attr.Attribute[Any], value: pd.Series[str] | None
    ) -> None:
        """
        Validate the aneris overrides value

        If `self.run_checks` is `False`, then this is a no-op
        """
        if value is None:
            return

        if not self.run_checks:
            return

    @historical_emissions.validator
    def validate_historical_emissions(
        self, attribute: attr.Attribute[Any], value: pd.DataFrame
    ) -> None:
        """
        Validate the historical emissions value

        If `self.run_checks` is `False`, then this is a no-op
        """
        if not self.run_checks:
            return

        assert_index_is_multiindex(value)
        assert_data_is_all_numeric(value)
        assert_has_index_levels(
            value, [self.variable_level, self.region_level, self.unit_level]
        )
        assert_has_data_for_times(
            value,
            name="historical_emissions",
            times=[self.harmonisation_year],
            allow_nan=False,
        )

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonise

        Parameters
        ----------
        in_emissions
            Emissions to harmonise

        Returns
        -------
        :
            Harmonised emissions
        """
        if self.run_checks:
            assert_index_is_multiindex(in_emissions)
            assert_data_is_all_numeric(in_emissions)
            assert_has_index_levels(
                in_emissions,
                [
                    self.variable_level,
                    self.region_level,
                    self.unit_level,
                    # Needed for parallelisation
                    *self.scenario_group_levels,
                ],
            )
            assert_has_data_for_times(
                in_emissions,
                name="in_emissions",
                times=[self.harmonisation_year],
                allow_nan=False,
            )

            try:
                assert_metadata_values_all_allowed(
                    in_emissions,
                    metadata_key=self.variable_level,
                    allowed_values=self.historical_emissions.index.get_level_values(
                        self.variable_level
                    ).unique(),
                )
            except NotAllowedMetadataValuesError as exc:
                msg = "The input emissions contains values that aren't in history"
                raise ValueError(msg) from exc

        harmonised_df = pd.concat(
            apply_op_parallel_progress(
                func_to_call=gcages.aneris_helpers.harmonise_scenario,
                iterable_input=(
                    gdf for _, gdf in in_emissions.groupby(self.scenario_group_levels)
                ),
                parallel_op_config=ParallelOpConfig.from_user_facing(
                    progress=self.progress,
                    max_workers=self.n_processes,
                    progress_results_kwargs=dict(desc="Scenarios to harmonise"),
                ),
                history=self.historical_emissions,
                year=self.harmonisation_year,
                overrides=self.aneris_overrides,
            )
        )

        if self.run_checks:
            assert_harmonised(
                harmonised_df,
                history=self.historical_emissions,
                harmonisation_time=self.harmonisation_year,
            )

            pd.testing.assert_index_equal(
                harmonised_df.index,
                in_emissions.index,
                check_order=False,  # type: ignore # pandas-stubs out of date
            )
            if harmonised_df.columns.dtype != in_emissions.columns.dtype:
                msg = (
                    "Column type has changed: "
                    f"{harmonised_df.columns.dtype=} {in_emissions.columns.dtype=}"
                )
                raise AssertionError(msg)

        return harmonised_df
