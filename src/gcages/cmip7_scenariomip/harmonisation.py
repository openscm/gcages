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
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

from gcages.assertions import (
    MissingDataForTimesError,
    assert_data_is_all_numeric,
    assert_has_data_for_times,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_metadata_values_all_allowed,
    assert_only_working_on_variable_unit_region_variations,
)
from gcages.harmonisation import assert_harmonised
from gcages.harmonisation.aneris import AnerisHarmoniser
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name


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


def harmonise_single_scenario(
    indf: pd.DataFrame,
    history: pd.DataFrame,
    harmonisation_year: int,
    overrides: pd.Series[str] | None,
    run_checks: bool,
) -> pd.DataFrame:
    """
    Harmonise a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    history
        History to harmonise to

    harmonisation_year
        Year to use for harmonisation

    overrides
        Overrides to pass to aneris

    Returns
    -------
    :
        Harmonised scenario
    """
    assert_only_working_on_variable_unit_region_variations(indf)

    harmoniser = AnerisHarmoniser(
        historical_emissions=history,
        harmonisation_year=harmonisation_year,
        aneris_overrides=overrides,
        run_checks=run_checks,
        progress=False,
        n_processes=None,
    )
    out = harmoniser(indf)

    return out


def load_cmip7_scenariomip_historical_emissions(filepath: Path) -> pd.DataFrame:
    """
    Load historical emissions for CMIP7 ScenarioMIP harmonisation.

    Parameters
    ----------
    filepath
        Path from which to load the file

    Returns
    -------
    :
        Historical emissions

    Raises
    ------
    AssertionError
        `filepath` does not have the expected hash.

        We expect to be reading the file from
        https://zenodo.org/records/17845154/files/global-workflow-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d_202511040855_202512071232_202511040855_202511040855_0002_0002.csv?download=1
    """
    fp_hash = get_file_hash(filepath, algorithm="md5")
    fp_hash_exp = "19482df604f1dc746fb354ef66ef9047"

    if fp_hash != fp_hash_exp:
        msg = (
            f"The md5 hash of {filepath} is {fp_hash}. "
            f"This does not match what we expect {fp_hash_exp=}."
        )
        raise AssertionError(msg)

    res = load_timeseries_csv(
        filepath,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )

    return res


def load_aneris_overrides_file(filepath: Path) -> pd.Series[str]:
    """
    Load historical emissions for CMIP7 ScenarioMIP harmonisation.

    Parameters
    ----------
    filepath
        Path from which to load the file

    Returns
    -------
    :
        Aneris overrides
    """
    raw = pd.read_csv(filepath)
    res = raw.set_index(list(raw.columns.difference(["method"])))["method"]

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

        # TODO: implement a `assert_aneris_overrides_align_with_historical` function

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
            # Needed for parallelisation
            assert_has_index_levels(
                in_emissions, ["variable", "unit", "model", "scenario"]
            )
            try:
                assert_has_data_for_times(
                    in_emissions,
                    name="in_emissions",
                    times=[self.harmonisation_year],
                    allow_nan=False,
                )
            except MissingDataForTimesError as exc:
                msg = f"We require data for {self.harmonisation_year}."
                raise KeyError(msg) from exc

            assert_metadata_values_all_allowed(
                in_emissions,
                metadata_key="variable",
                allowed_values=self.historical_emissions.index.get_level_values(
                    "variable"
                ).unique(),
            )

        harmonised_df = pd.concat(
            apply_op_parallel_progress(
                func_to_call=harmonise_single_scenario,
                iterable_input=(
                    gdf for _, gdf in in_emissions.groupby(["model", "scenario"])
                ),
                parallel_op_config=ParallelOpConfig.from_user_facing(
                    progress=self.progress,
                    max_workers=self.n_processes,
                ),
                history=self.historical_emissions,
                harmonisation_year=self.harmonisation_year,
                overrides=self.aneris_overrides,
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
    def from_cmip7_scenariomip_global_config(
        cls,
        cmip7_scenariomip_global_historical_emissions_file: Path,
        aneris_global_overrides_file: Path,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> CMIP7ScenarioMIPHarmoniser:
        """
        Initialise from CMIP7 ScenarioMIP global-level workflow configuration

        Parameters
        ----------
        cmip7_scenariomip_global_historical_emissions_file
            File containing the CMIP7 ScenarioMIP historical emissions

            Can be retrieved from:
            https://zenodo.org/records/17845154/files/global-workflow-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d_202511040855_202512071232_202511040855_202511040855_0002_0002.csv?download=1

        aneris_global_overrides_file
            File from which to load the aneris overrides

            TODO: figure out where to store this

        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        progress
            Should a progress bar be shown for each operation?

        n_processes
            Number of processes to use for parallel processing.

            Set to 1 to process in serial.

        Returns
        -------
        :
            Initialised harmoniser
        """
        historical_emissions = load_cmip7_scenariomip_historical_emissions(
            cmip7_scenariomip_global_historical_emissions_file
        )

        # Drop out the model and scenario levels
        historical_emissions = historical_emissions.reset_index(
            historical_emissions.index.names.difference(["variable", "region", "unit"]),  # type: ignore # pandas-stubs out of date
            drop=True,
        )

        # Convert names to gcages naming before continuing
        historical_emissions = update_index_levels_func(
            historical_emissions,
            {
                "variable": lambda x: convert_variable_name(
                    x,
                    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            },
            copy=False,
        )

        # Grab all the overrides we've saved.
        # Load them here.
        # The overrides are currently applied at the model level.
        # That's probably a sensible default,
        # but sensible defaults can be very hard to define...
        aneris_overrides = load_aneris_overrides_file(aneris_global_overrides_file)

        return cls(
            historical_emissions=historical_emissions,
            harmonisation_year=2023,
            aneris_overrides=aneris_overrides,
            run_checks=run_checks,
            progress=progress,
            n_processes=n_processes,
        )
