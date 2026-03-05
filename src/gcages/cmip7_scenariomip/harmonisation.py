"""
Harmonisation part of the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import attr
import pandas as pd
from attrs import define, field
from pandas_openscm.io import load_timeseries_csv
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_data_for_times,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_metadata_values_all_allowed,
    assert_only_working_on_variable_unit_region_variations,
)
from gcages.harmonisation import assert_harmonised
from gcages.harmonisation.aneris import AnerisHarmoniser


def _get_remind_overrides(
    in_emissions: pd.DataFrame, harmonisation_year: int
) -> pd.Series[str] | None:
    """
    Get overrides required for REMIND scenarios.

    This follows the CMIP7 rule:
    use ``reduce_ratio_2050`` for all timeseries
    that are non-zero in the harmonisation year.
    """
    model_values = in_emissions.index.get_level_values("model").unique()
    if model_values.shape[0] != 1:
        msg = (
            "Expected to process one model at a time when creating "
            f"REMIND overrides. {model_values=}"
        )
        raise AssertionError(msg)

    model = model_values[0]
    if not model.startswith("REMIND"):
        return None

    non_zero_in_harm_year = in_emissions[harmonisation_year] != 0.0
    if not non_zero_in_harm_year.any():
        return None

    override_index = (
        in_emissions.loc[non_zero_in_harm_year]
        .reset_index()[["variable", "region"]]
        .drop_duplicates()
        .set_index(["variable", "region"])
        .index
    )
    out = pd.Series(
        "reduce_ratio_2050",
        index=override_index,
        name="method",
    )

    return out


def _combine_overrides(
    base: pd.Series[str] | None, addition: pd.Series[str] | None
) -> pd.Series[str] | None:
    """
    Combine two overrides Series.

    If entries overlap, ``addition`` takes precedence.
    """
    if addition is None:
        return base

    if base is None:
        return addition

    if base.index.names != addition.index.names:
        msg = (
            "Combining overrides is only supported when both use the same index "
            f"structure. {base.index.names=} {addition.index.names=}"
        )
        raise NotImplementedError(msg)

    out = pd.concat([base, addition])
    out = out[~out.index.duplicated(keep="last")]

    return out


def _harmonise_single_scenario(
    in_emissions: pd.DataFrame,
    historical_emissions: pd.DataFrame,
    harmonisation_year: int,
    aneris_overrides: pd.Series[str] | None,
    run_checks: bool,
) -> pd.DataFrame:
    """
    Harmonise a single model-scenario.
    """
    assert_only_working_on_variable_unit_region_variations(in_emissions)

    if harmonisation_year < in_emissions.columns.min():
        msg = (
            "Harmonisation year is before the first available year in the scenario. "
            f"{harmonisation_year=} {in_emissions.columns.min()=}"
        )
        raise KeyError(msg)

    if harmonisation_year > in_emissions.columns.max():
        msg = (
            "Harmonisation year is after the last available year in the scenario. "
            f"{harmonisation_year=} {in_emissions.columns.max()=}"
        )
        raise KeyError(msg)

    # CMIP7 inputs can be reported at 5-year intervals.
    # Interpolate to annual values so we always have the harmonisation year.
    out_years = list(range(in_emissions.columns.min(), in_emissions.columns.max() + 1))
    emissions_to_harmonise = in_emissions.reindex(columns=out_years).interpolate(
        method="linear", axis="columns"
    )

    remind_overrides = _get_remind_overrides(
        in_emissions=emissions_to_harmonise, harmonisation_year=harmonisation_year
    )
    effective_overrides = _combine_overrides(aneris_overrides, remind_overrides)

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=harmonisation_year,
        aneris_overrides=effective_overrides,
        run_checks=run_checks,
        progress=False,
        n_processes=None,
    )
    out = harmoniser(emissions_to_harmonise)

    return out


def load_cmip7_scenariomip_historical_emissions(filepath: Path) -> pd.DataFrame:
    """
    Load historical emissions for CMIP7 ScenarioMIP harmonisation.
    """
    res = load_timeseries_csv(
        filepath,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )

    # Drop out all metadata except region, variable and unit.
    res = res.reset_index(
        res.index.names.difference(["variable", "region", "unit"]),  # type: ignore # pandas-stubs out of date
        drop=True,
    )

    if res.index.duplicated().any():
        duplicated = res.index[res.index.duplicated()]
        msg = (
            "Historical emissions contains duplicate index rows "
            "after dropping model/scenario metadata. "
            f"{duplicated=}"
        )
        raise ValueError(msg)

    # Drop rows that have no data at any time.
    res = res.dropna(how="all")

    return res


@define
class CMIP7ScenarioMIPHarmoniser:
    """
    Harmoniser for CMIP7 ScenarioMIP.
    """

    historical_emissions: pd.DataFrame = field()
    """
    Historical emissions to use for harmonisation.
    """

    harmonisation_year: int = 2023
    """
    Year in which to harmonise.
    """

    aneris_overrides: pd.Series[str] | None = field(default=None)
    """
    Overrides to supply to aneris.
    """

    run_checks: bool = True
    """
    If ``True``, run checks on input and output data.
    """

    progress: bool = True
    """
    Should progress bars be shown for each operation?
    """

    n_processes: int | None = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.
    """

    @aneris_overrides.validator
    def validate_aneris_overrides(
        self, attribute: attr.Attribute[Any], value: pd.Series[str] | None
    ) -> None:
        """
        Validate ``aneris_overrides``.
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
        Validate ``historical_emissions``.
        """
        if not self.run_checks:
            return

        assert_index_is_multiindex(value)
        assert_data_is_all_numeric(value)
        assert_has_index_levels(value, ["variable", "region", "unit"])
        assert_has_data_for_times(
            value,
            name="historical_emissions",
            times=[self.harmonisation_year],
            allow_nan=False,
        )

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonise.
        """
        if self.run_checks:
            assert_index_is_multiindex(in_emissions)
            assert_data_is_all_numeric(in_emissions)
            assert_has_index_levels(
                in_emissions, ["model", "scenario", "variable", "region", "unit"]
            )

            assert_metadata_values_all_allowed(
                in_emissions,
                metadata_key="variable",
                allowed_values=self.historical_emissions.index.get_level_values(
                    "variable"
                ).unique(),
            )

        harmonised_df = pd.concat(
            apply_op_parallel_progress(
                func_to_call=_harmonise_single_scenario,
                iterable_input=(
                    gdf for _, gdf in in_emissions.groupby(["model", "scenario"])
                ),
                parallel_op_config=ParallelOpConfig.from_user_facing(
                    progress=self.progress,
                    max_workers=self.n_processes,
                ),
                historical_emissions=self.historical_emissions,
                harmonisation_year=self.harmonisation_year,
                aneris_overrides=self.aneris_overrides,
                run_checks=self.run_checks,
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

    @classmethod
    def from_cmip7_scenariomip_config(
        cls,
        cmip7_scenariomip_historical_emissions_file: Path,
        harmonisation_year: int = 2023,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> CMIP7ScenarioMIPHarmoniser:
        """
        Initialise from CMIP7 ScenarioMIP configuration.
        """
        historical_emissions = load_cmip7_scenariomip_historical_emissions(
            cmip7_scenariomip_historical_emissions_file
        )

        return cls(
            historical_emissions=historical_emissions,
            harmonisation_year=harmonisation_year,
            run_checks=run_checks,
            n_processes=n_processes,
            progress=progress,
        )
