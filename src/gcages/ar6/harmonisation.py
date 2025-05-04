"""
Harmonisation part of the AR6 workflow
"""

from __future__ import annotations

import importlib
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Any

import attr
import pandas as pd
from attrs import define, field
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

from gcages.aneris_helpers import harmonise_all
from gcages.assertions import (
    MissingDataForTimesError,
    assert_data_is_all_numeric,
    assert_has_data_for_times,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_metadata_values_all_allowed,
    assert_only_working_on_variable_unit_variations,
)
from gcages.exceptions import MissingOptionalDependencyError
from gcages.harmonisation import assert_harmonised
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


def add_historical_year_based_on_scaling(
    year_to_add: int,
    year_calc_scaling: int,
    emissions: pd.DataFrame,
    emissions_history: pd.DataFrame,
    ms: tuple[str, ...] = ("model", "scenario"),
) -> pd.DataFrame:
    """
    Add a historical emissions year based on scaling

    Parameters
    ----------
    year_to_add
        Year to add

    year_calc_scaling
        Year to use to calculate the scaling

    emissions
        Emissions to which to add data for `year_to_add`

    emissions_history
        Emissions history to use to calculate
        the fill values based on scaling

    ms
        Name of the model and scenario columns.

        These have to be dropped from `emissions_historical`
        before everything will line up.

    Returns
    -------
    :
        `emissions` with data for `year_to_add`
        based on the scaling between `emissions`
        and `emissions_historical` in `year_calc_scaling`.
    """
    mod_scen_unique = emissions.index.droplevel(
        emissions.index.names.difference(["model", "scenario"])  # type: ignore
    ).unique()
    if mod_scen_unique.shape[0] > 1:
        # Processing is much trickier with multiple scenarios
        raise NotImplementedError(mod_scen_unique)

    emissions_historical_common_vars = emissions_history.loc[
        emissions_history.index.get_level_values("variable").isin(
            emissions.index.get_level_values("variable")
        )
    ]

    emissions_historical_no_ms = emissions_historical_common_vars.reset_index(
        emissions_historical_common_vars.index.names.difference(  # type: ignore # pandas-stubs not up to date
            ["region", "variable", "unit"]
        ),
        drop=True,
    )

    scale_factor = emissions[year_calc_scaling].divide(
        emissions_historical_no_ms[year_calc_scaling]
    )
    fill_value = scale_factor.multiply(emissions_historical_no_ms[year_to_add])
    fill_value.name = year_to_add

    out = pd.concat([emissions, fill_value], axis="columns").sort_index(axis="columns")

    return out


def load_ar6_historical_emissions(filepath: Path) -> pd.DataFrame:
    """
    Load the historical emissions that were used in AR6

    The data is massaged to what is expected by our harmonisation,
    it isn't the raw data (at least not raw metadata).

    Parameters
    ----------
    filepath
        Filepath from which to load the historical emissions

    Returns
    -------
    :
        Historical emissions used in AR6

    Raises
    ------
    AssertionError
        `filepath` points to a file that does not have the expected hash
    """
    fp_hash = get_file_hash(filepath, algorithm="sha256")
    if fp_hash != "b0538b63aca8e0846a4bb55da50529e72f83cb0c7373f26eac4c2a80ca6e3ac1":
        msg = (
            f"The sha256 hash of {filepath} is {fp_hash}. "
            "This does not match what we expect."
        )
        raise AssertionError(msg)

    res = load_timeseries_csv(
        filepath,
        lower_column_names=True,
        index_columns=["model", "scenario", "variable", "unit", "region"],
        out_column_type=int,
    )

    return res


def harmonise_scenario(
    indf: pd.DataFrame,
    history: pd.DataFrame,
    year: int,
    overrides: pd.Series[str] | None,
    calc_scaling_year: int,
) -> pd.DataFrame:
    """
    Harmonise a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    history
        History to harmonise to

    year
        Year to use for harmonisation

    overrides
        Overrides to pass to aneris

    calc_scaling_year
        Year to use for calculating scaling if `year` is not in `indf`

    Returns
    -------
    :
        Harmonised scenario
    """
    if importlib.util.find_spec("scipy") is None:
        raise MissingOptionalDependencyError("harmonise_scenario", requirement="scipy")

    assert_only_working_on_variable_unit_variations(indf)

    # In AR6, if the year we needed wasn't there, we tried some workarounds
    if year not in indf:
        emissions_to_harmonise = add_historical_year_based_on_scaling(
            year_to_add=year,
            year_calc_scaling=calc_scaling_year,
            emissions=indf,
            emissions_history=history,
        )

    elif indf[year].isnull().any():
        null_emms_in_harm_year = indf[year].isnull()

        dont_change = indf[~null_emms_in_harm_year]

        updated = add_historical_year_based_on_scaling(
            year_to_add=year,
            year_calc_scaling=calc_scaling_year,
            emissions=indf[null_emms_in_harm_year].drop(year, axis="columns"),
            emissions_history=history,
        )

        emissions_to_harmonise = pd.concat([dont_change, updated])

    else:
        emissions_to_harmonise = indf

    # In AR6, any emissions with zero in the harmonisation year were dropped
    emissions_to_harmonise = emissions_to_harmonise[
        ~(emissions_to_harmonise[year] == 0.0)
    ]

    ### In AR6, we interpolated before harmonising

    # First, check that there are no nans in the max year.
    # I don't know what happens in that case.
    if (  # pragma: no cover
        emissions_to_harmonise[emissions_to_harmonise.columns.max()].isnull().any()
    ):
        raise NotImplementedError

    # Then, interpolate
    out_interp_years = list(range(year, emissions_to_harmonise.columns.max() + 1))
    emissions_to_harmonise = emissions_to_harmonise.reindex(
        columns=out_interp_years
    ).interpolate(method="slinear", axis="columns")

    harmonised = harmonise_all(
        emissions_to_harmonise,
        history=history,
        year=year,
        overrides=overrides,
    )

    return harmonised


@define
class AR6Harmoniser:
    """
    Harmoniser that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_like_config`][(c)]
    """

    historical_emissions: pd.DataFrame = field()
    """
    Historical emissions to use for harmonisation
    """

    harmonisation_year: int
    """
    Year in which to harmonise
    """

    calc_scaling_year: int
    """
    Year to use for calculating a scaling factor from historical

    This is only needed if `self.harmonisation_year`
    is not in the emissions to be harmonised.

    For example, if `self.harmonisation_year` is 2015
    and `self.calc_scaling_year` is 2010
    and we have a scenario without 2015 data,
    then we will use the difference from historical in 2010
    to infer a value for 2015.

    This logic was perculiar to AR6, it may not be repeated.
    """

    aneris_overrides: pd.Series[str] | None = field()
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

    progress: bool = True
    """
    Should progress bars be shown for each operation?
    """

    n_processes: int = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to 1 to process in serial.
    """

    @aneris_overrides.validator
    def validate_aneris_overrides(
        self, attribute: attr.Attribute[Any], value: pd.DataFrame | None
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
        assert_has_index_levels(value, ["variable", "unit"])
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
            except MissingDataForTimesError as exc_hy:
                try:
                    assert_has_data_for_times(
                        in_emissions,
                        name="in_emissions",
                        times=[self.calc_scaling_year],
                        allow_nan=False,
                    )
                except MissingDataForTimesError as exc_csy:
                    msg = (
                        f"We require data for either {self.harmonisation_year} "
                        f"or {self.calc_scaling_year} "
                        "but neither had the required data. "
                        f"Error from checking for {self.harmonisation_year} data: "
                        f"{exc_hy}. "
                        f"Error from checking for {self.calc_scaling_year} data: "
                        f"{exc_csy}. "
                    )
                    raise KeyError(msg)

            assert_metadata_values_all_allowed(
                in_emissions,
                metadata_key="variable",
                allowed_values=self.historical_emissions.index.get_level_values(
                    "variable"
                ).unique(),
            )

        harmonised_df = pd.concat(
            apply_op_parallel_progress(
                func_to_call=harmonise_scenario,
                iterable_input=(
                    gdf for _, gdf in in_emissions.groupby(["model", "scenario"])
                ),
                parallel_op_config=ParallelOpConfig.from_user_facing(
                    progress=self.progress,
                    max_workers=self.n_processes,
                ),
                history=self.historical_emissions,
                year=self.harmonisation_year,
                overrides=self.aneris_overrides,
                calc_scaling_year=self.calc_scaling_year,
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
    def from_ar6_config(
        cls,
        ar6_historical_emissions_file: Path,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int = multiprocessing.cpu_count(),
    ) -> AR6Harmoniser:
        """
        Initialise from the config used in AR6

        Parameters
        ----------
        ar6_historical_emissions_file
            File containing the AR6 historical emissions

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
        historical_emissions = load_ar6_historical_emissions(
            ar6_historical_emissions_file
        )

        # Drop out all metadata except region, variable and unit
        historical_emissions = historical_emissions.reset_index(
            historical_emissions.index.names.difference(["variable", "region", "unit"]),  # type: ignore # pandas-stubs out of date
            drop=True,
        )

        # Strip off prefix
        historical_emissions = update_index_levels_func(
            historical_emissions,
            {
                "variable": lambda x: x.replace("AR6 climate diagnostics|", "").replace(
                    "|Unharmonized", ""
                )
            },
            copy=False,
        )

        # Drop down to only the variables we care about
        historical_emissions = historical_emissions.loc[
            historical_emissions.index.get_level_values("variable").isin(
                [
                    "Emissions|BC",
                    "Emissions|PFC|C2F6",
                    "Emissions|PFC|C6F14",
                    "Emissions|PFC|CF4",
                    "Emissions|CO",
                    "Emissions|CO2",
                    "Emissions|CO2|AFOLU",
                    "Emissions|CO2|Energy and Industrial Processes",
                    "Emissions|CH4",
                    # "Emissions|F-Gases",  # Not used
                    # "Emissions|HFC",  # Not used
                    "Emissions|HFC|HFC125",
                    "Emissions|HFC|HFC134a",
                    "Emissions|HFC|HFC143a",
                    "Emissions|HFC|HFC227ea",
                    "Emissions|HFC|HFC23",
                    # 'Emissions|HFC|HFC245ca',  # all nan in historical dataset (RCMIP)
                    # "Emissions|HFC|HFC245fa",  # not in historical dataset (RCMIP)
                    "Emissions|HFC|HFC32",
                    "Emissions|HFC|HFC43-10",
                    "Emissions|N2O",
                    "Emissions|NH3",
                    "Emissions|NOx",
                    "Emissions|OC",
                    # "Emissions|PFC",  # Not used
                    "Emissions|SF6",
                    "Emissions|Sulfur",
                    "Emissions|VOC",
                ]
            )
        ]

        # Update variable names
        historical_emissions = update_index_levels_func(
            historical_emissions,
            {
                "variable": partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.IAMC,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            },
            copy=False,
        )

        # Strip out any units that won't play nice with pint
        historical_emissions = strip_pint_incompatible_characters_from_units(
            historical_emissions, units_index_level="unit"
        )

        # Drop out rows with all NaNs
        historical_emissions = historical_emissions.dropna(how="all")

        # We don't need historical emissions after 1990
        # (probably even later, but this is fine).
        historical_emissions = historical_emissions.loc[:, 1990:]

        # All variables not mentioned here use aneris' default decision tree
        aneris_overrides_ar6_df = pd.DataFrame(
            [
                # Not used
                # {
                #     # high historical variance (cov=16.2)
                #     "method": "reduce_ratio_2150_cov",
                #     "variable": "Emissions|PFC",
                # },
                {
                    # high historical variance (cov=16.2)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|C2F6",
                },
                {
                    # high historical variance (cov=15.4)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|C6F14",
                },
                {
                    # high historical variance (cov=11.2)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|CF4",
                },
                {
                    # high historical variance (cov=15.4)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|CO",
                },
                {
                    # always ratio method by choice
                    "method": "reduce_ratio_2080",
                    "variable": "Emissions|CO2",
                },
                {
                    # high historical variance,
                    # but using offset method to prevent diff
                    # from increasing when going negative rapidly (cov=23.2)
                    "method": "reduce_offset_2150_cov",
                    "variable": "Emissions|CO2|AFOLU",
                },
                {
                    # always ratio method by choice
                    "method": "reduce_ratio_2080",
                    "variable": "Emissions|CO2|Energy and Industrial Processes",
                },
                # Not used
                # {
                #     # basket not used in infilling
                #     # (sum of f-gases with low model reporting confidence)
                #     "method": "constant_ratio",
                #     "variable": "Emissions|F-Gases",
                # },
                # Not used
                # {
                #     # basket not used in infilling
                #     # (sum of subset of f-gases with low model reporting confidence)
                #     "method": "constant_ratio",
                #     "variable": "Emissions|HFC",
                # },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC125",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC134a",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC143a",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC227ea",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC23",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC32",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC43-10",
                },
                {
                    # high historical variance (cov=18.5)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|OC",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|SF6",
                },
                {
                    # high historical variance (cov=12.0)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|VOC",
                },
            ]
        )
        aneris_overrides_ar6_df["variable"] = aneris_overrides_ar6_df["variable"].map(
            partial(
                convert_variable_name,
                from_convention=SupportedNamingConventions.IAMC,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        )
        aneris_overrides_ar6 = aneris_overrides_ar6_df.set_index("variable")["method"]

        return cls(
            historical_emissions=historical_emissions,
            harmonisation_year=2015,
            calc_scaling_year=2010,
            aneris_overrides=aneris_overrides_ar6,
            run_checks=run_checks,
            n_processes=n_processes,
            progress=progress,
        )
