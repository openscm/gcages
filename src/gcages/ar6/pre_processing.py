"""
Pre-processing part of the workflow
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Callable

import pandas as pd
from attrs import define
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_data_for_times,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_only_working_on_variable_unit_variations,
)
from gcages.exceptions import MissingOptionalDependencyError
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec

    P = ParamSpec("P")


# As a note, if you wanted to re-use these helpers somewhere else,
# you would want to test them carefully first,
# it's really not clear to me that they are well-written or obvious.
def add_conditional_sums(
    indf: pd.DataFrame,
    conditional_sums: tuple[tuple[str, tuple[str, ...]], ...],
    copy_on_entry: bool = True,
) -> pd.DataFrame:
    """
    Add sums to a [pd.DataFrame][pandas.DataFrame] if all components are present

    Parameters
    ----------
    indf
        Data to add sums to

    conditional_sums
        Definition of the conditional sums.

        The first element of each sub-tuple is the name of the variable to add.
        The second element are its components.
        If the variable is added, all the sub-components are dropped.
        All components must be present for the variable to be added.
        If the variable is already there, the sum is not re-calculated or checked.

    copy_on_entry
        Should the data be copied on entry?

    Returns
    -------
    :
        `indf` with conditional sums added if all enabling conditions were fulfilled.
    """
    try:
        from pandas_indexing.core import concat
        from pandas_indexing.selectors import isin
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "add_conditional_sums", requirement="pandas_indexing"
        ) from exc

    assert_only_working_on_variable_unit_variations(indf)

    if copy_on_entry:
        out = indf.copy()

    else:
        out = indf

    for v_target, v_sources in conditional_sums:
        existing_vars: pd.MultiIndex = out.pix.unique("variable")  # type: ignore
        if v_target not in existing_vars:
            if all(v in existing_vars for v in v_sources):
                locator_sources = isin(variable=v_sources)
                to_add = out.loc[locator_sources]

                tmp = to_add.groupby(list(set(to_add.index.names) - {"variable"})).sum(
                    min_count=len(v_sources)
                )
                tmp = tmp.pix.assign(variable=v_target)
                out = concat([out.loc[~locator_sources], tmp], axis="index")

    return out


def reclassify_variables(
    indf: pd.DataFrame,
    reclassifications: Mapping[str, tuple[str, ...]],
    copy_on_entry: bool = True,
) -> pd.DataFrame:
    """
    Reclassify variables

    Parameters
    ----------
    indf
        Data to add sums to

    reclassifications
        Definition of the reclassifications.

        For each variable (key) in `reclassifications`, the variables in its value
        will be reclassified as part of its total.

        For example, if `reclassifications` is

        ```python
        {"var_a": ("var_b", "var_c")}
        ```

        then if "var_b" or "var_c" (or both) is in `indf`,
        they will be removed and their contents will be added to the total of `var_a`.

    copy_on_entry
        Should the data be copied on entry?

    Returns
    -------
    :
        `indf`, reclassified as needed.
    """
    try:
        from pandas_indexing.selectors import isin
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "reclassify_variables", requirement="pandas_indexing"
        ) from exc

    assert_only_working_on_variable_unit_variations(indf)

    if copy_on_entry:
        out = indf.copy()

    else:
        out = indf

    for v_target, v_sources in reclassifications.items():
        locator_sources = isin(variable=v_sources)
        to_add = out.loc[locator_sources]
        if not to_add.empty:
            out.loc[isin(variable=v_target)] += to_add.sum()  # type: ignore
            out = out.loc[~locator_sources]

    return out


def condtionally_remove_variables(
    indf: pd.DataFrame,
    conditional_removals: tuple[tuple[str, tuple[str, ...]], ...],
    copy_on_entry: bool = True,
) -> pd.DataFrame:
    """
    Conditionally remove variables

    Parameters
    ----------
    indf
        Data to add sums to

    conditional_removals
        Definition of the conditional removals.

        For each tuple, the first element defines the variable that can be removed.
        This variable will be removed if all variables in the tuple's second element
        are present in `indf`.

    copy_on_entry
        Should the data be copied on entry?

    Returns
    -------
    :
        `indf` with variables removed according to this function's logic.
    """
    try:
        from pandas_indexing.selectors import isin
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "condtionally_remove_variables", requirement="pandas_indexing"
        ) from exc

    assert_only_working_on_variable_unit_variations(indf)

    if copy_on_entry:
        out = indf.copy()

    else:
        out = indf

    for v_drop, v_sub_components in conditional_removals:
        existing_vars: pd.MultiIndex = out.pix.unique("variable")  # type: ignore
        if v_drop in existing_vars and all(
            v in existing_vars for v in v_sub_components
        ):
            out = out.loc[~isin(variable=v_drop)]

    return out


def drop_variables_if_identical(
    indf: pd.DataFrame,
    drop_if_identical: tuple[tuple[str, str], ...],
    copy_on_entry: bool = True,
) -> pd.DataFrame:
    """
    Drop variables if they are identical to another variable

    Parameters
    ----------
    indf
        Data to add sums to

    drop_if_identical
        Definition of the variables that can be dropped.

        For each tuple, the first element defines the variable that can be removed
        and the second element defines the variable to compare it to.
        If the variable to drop has the same values as the variable to compare to,
        it is dropped.

    copy_on_entry
        Should the data be copied on entry?

    Returns
    -------
    :
        `indf` with variables removed according to this function's logic.
    """
    try:
        from pandas_indexing.selectors import isin
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "drop_variables_if_identical", requirement="pandas_indexing"
        ) from exc

    assert_only_working_on_variable_unit_variations(indf)

    if copy_on_entry:
        out = indf.copy()

    else:
        out = indf

    for v_drop, v_check in drop_if_identical:
        existing_vars: pd.MultiIndex = out.pix.unique("variable")  # type: ignore
        if all(v in existing_vars for v in (v_drop, v_check)):
            # Should really use isclose here, but we didn't in AR6
            # so we get some funny reporting for weird scenarios
            # e.g. C3IAM 2.0 2C-hybrid
            if (
                (
                    out.loc[isin(variable=v_drop)]
                    .reset_index("variable", drop=True)
                    .dropna(axis="columns")
                    == out.loc[isin(variable=v_check)]
                    .reset_index("variable", drop=True)
                    .dropna(axis="columns")
                )
                .all()
                .all()
            ):
                out = out.loc[~isin(variable=v_drop)]

    return out


def run_parallel_pre_processing(  # noqa: PLR0913
    indf: pd.DataFrame,
    func_to_call: Callable[Concatenate[pd.DataFrame, P], pd.DataFrame],
    groups: tuple[str, ...] = ("model", "scenario"),
    progress: bool = True,
    progress_bar_desc: str | None = None,
    n_processes: int | None = multiprocessing.cpu_count(),
    *args: P.args,
    **kwargs: P.kwargs,
) -> pd.DataFrame:
    """
    Run a pre-processing step in parallel

    Parameters
    ----------
    indf
        Input data to process

    func_to_call
        Function to apply to each group in `indf`

    groups
        Columns to use to group the data in `indf`

    progress
        Should a progress bar be displayed?

    progress_bar_desc
        If `progress`, the description of the progress bar.

        If not supplied, we use a default description.

    n_processes
        Number of parallel processes to use

    **kwargs
        Passed to `run_parallel`

    Returns
    -------
    :
        Result of calling `func_to_call` on each group in `indf`.
    """
    if progress and progress_bar_desc is None:
        progress_bar_desc = f"{', '.join(groups)} combinations"

    res = pd.concat(
        apply_op_parallel_progress(
            func_to_call,
            (gdf for _, gdf in indf.groupby(list(groups))),
            ParallelOpConfig.from_user_facing(
                progress=progress,
                progress_results_kwargs=dict(desc=progress_bar_desc),
                max_workers=n_processes,
            ),
            *args,
            **kwargs,
        )
    )

    return res


@define
class AR6PreProcessor:
    """
    Pre-processor that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_config`][(c)]
    """

    emissions_out: tuple[str, ...]
    """
    Names of emissions that can be included in the result of pre-processing

    Not all these emissions need to be there,
    but any names which are not in this list will be removed as part of pre-processing.
    """

    negative_value_not_small_threshold: float
    """
    Threshold which defines when a negative value is not small

    Non-CO2 emissions less than this that are negative
    are not automatically set to zero.
    """

    conditional_sums: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    """
    Specification for variables that can be created from other variables

    Form:

    ```python
    (
        (variable_that_can_be_created, (component_1, component_2)),
        ...
    )
    ```

    The variable that can be created is only created
    if all the variables it depends on are present.
    """

    reclassifications: Mapping[str, tuple[str, ...]] | None = None
    """
    Variables that should be reclassified as being part of another variable

    Form:

    ```python
    {
        variable_to_add_to: (variable_to_rename_1, variable_to_rename_2),
        ...
    }
    ```

    For example
    ```python
    {
        "Emissions|CO2|Energy and Industrial Processes": (
            "Emissions|CO2|Other",
            "Emissions|CO2|Waste",
        )
    }
    ```
    """

    conditional_removals: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    """
    Specification for variables that can be removed if other variables are present

    Form:

    ```python
    (
        (variable_that_can_be_removed, (component_1, component_2)),
        ...
    )
    ```

    The variable that can be removed is only removed
    if all the variables it depends on are present.
    """

    drop_if_identical: tuple[tuple[str, str], ...] | None = None
    """
    Variables that can be dropped if they are idential to another variable

    Form:

    ```python
    (
        (variable_that_can_be_removed, variable_to_compare_to),
        ...
    )
    ```

    The variable that can be removed is only removed
    if its values are identical to the variable it is compared to.
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

    n_processes: int | None = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to `None` to process in serial.
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process

        Parameters
        ----------
        in_emissions
            Emissions to pre-process

        Returns
        -------
        :
            Pre-processed emissions
        """
        try:
            from pandas_indexing.selectors import isin, ismatch
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "AR6PreProcessor.__call__", requirement="pandas_indexing"
            ) from exc

        if self.run_checks:
            assert_index_is_multiindex(in_emissions)
            assert_data_is_all_numeric(in_emissions)
            assert_has_index_levels(in_emissions, ["variable", "unit"])

        # Remove any rows with only zero (custom AR6 thing)
        in_emissions = in_emissions[
            ~(((in_emissions == 0.0) | in_emissions.isnull()).all(axis="columns"))
        ]

        rp = partial(
            run_parallel_pre_processing,
            progress=self.progress,
            n_processes=self.n_processes,
        )
        if self.conditional_sums is not None:
            in_emissions = rp(  # type: ignore
                in_emissions,
                func_to_call=add_conditional_sums,
                progress_bar_desc=(
                    "For each model-scenario, calculating conditional sums"
                ),
                conditional_sums=self.conditional_sums,
            )

        if self.reclassifications is not None:
            in_emissions = rp(  # type: ignore
                in_emissions,
                func_to_call=reclassify_variables,
                progress_bar_desc="For each model-scenario, reclassifying variables",
                reclassifications=self.reclassifications,
            )

        if self.conditional_removals is not None:
            in_emissions = rp(  # type: ignore
                in_emissions,
                func_to_call=condtionally_remove_variables,
                progress_bar_desc=(
                    "For each model-scenario, conditionally removing variables"
                ),
                conditional_removals=self.conditional_removals,
            )

        if self.drop_if_identical is not None:
            in_emissions = rp(  # type: ignore
                in_emissions,
                func_to_call=drop_variables_if_identical,
                progress_bar_desc=(
                    "For each model-scenario, dropping variables if they are identical"
                ),
                drop_if_identical=self.drop_if_identical,
            )

        # Negative value handling
        co2_locator = ismatch(variable="**CO2**")
        in_emissions.loc[~co2_locator] = in_emissions.loc[~co2_locator].where(
            # Where these conditions are true, keep the original data.
            (in_emissions.loc[~co2_locator] > 0)
            | (in_emissions.loc[~co2_locator] < self.negative_value_not_small_threshold)
            | in_emissions.loc[~co2_locator].isnull(),
            # Otherwise, set to zero
            other=0.0,
        )

        res: pd.DataFrame = in_emissions.loc[isin(variable=self.emissions_out)]

        # Strip out any units that won't play nice with pint
        res = strip_pint_incompatible_characters_from_units(
            res, units_index_level="unit"
        )

        # Convert to gcages naming conventions
        res = update_index_levels_func(
            res,
            {
                "variable": partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.IAMC,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            },
        )

        if self.run_checks:
            # AR6 required emissions for these years after pre-processing,
            # for some reason
            required_years = list(range(2020, 2100 + 1, 10))
            assert_has_data_for_times(
                res, name="res", times=required_years, allow_nan=False
            )

        return res

    @classmethod
    def from_ar6_config(
        cls,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> AR6PreProcessor:
        """
        Initialise from config that was used in AR6

        Parameters
        ----------
        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        progress
            Should a progress bar be shown for each operation?

        n_processes
            Number of processes to use for parallel processing.

            Set to `None` to process in serial.

        Returns
        -------
        :
            Initialised Pre-processor
        """
        ar6_emissions_for_harmonisation_iamc = tuple(
            v
            for v in (
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
            )
        )
        conditional_sums = (
            (
                "Emissions|CO2|Energy and Industrial Processes",
                (
                    "Emissions|CO2|Industrial Processes",
                    "Emissions|CO2|Energy",
                ),
            ),
        )
        reclassifications = {
            "Emissions|CO2|Energy and Industrial Processes": (
                "Emissions|CO2|Other",
                "Emissions|CO2|Waste",
            )
        }
        conditional_removals = (
            (
                "Emissions|CO2",
                (
                    "Emissions|CO2|Energy and Industrial Processes",
                    "Emissions|CO2|AFOLU",
                ),
            ),
        )
        drop_if_identical = (
            ("Emissions|CO2", "Emissions|CO2|Energy and Industrial Processes"),
            ("Emissions|CO2", "Emissions|CO2|AFOLU"),
        )

        return cls(
            emissions_out=ar6_emissions_for_harmonisation_iamc,
            negative_value_not_small_threshold=-0.1,
            conditional_sums=conditional_sums,
            reclassifications=reclassifications,
            conditional_removals=conditional_removals,
            drop_if_identical=drop_if_identical,
            run_checks=run_checks,
            n_processes=n_processes,
            progress=progress,
        )
