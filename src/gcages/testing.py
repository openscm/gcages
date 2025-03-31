"""
Code to support our tests

This is here, rather than in our `tests` directory
because of the issues that come
when you turn your tests into a package using `__init__.py` files
(for details, see https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#choosing-an-import-mode).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas_openscm.io import load_timeseries_csv

from gcages.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import pytest

AR6_IPS = (
    ("AIM/CGE 2.2", "EN_NPi2020_900f"),
    ("COFFEE 1.1", "EN_NPi2020_400f_lowBECCS"),
    ("GCAM 5.3", "NGFS2_Current Policies"),
    ("IMAGE 3.0", "EN_INDCi2030_3000f"),
    ("MESSAGEix-GLOBIOM 1.0", "LowEnergyDemand_1.3_IPCC"),
    ("MESSAGEix-GLOBIOM_GEI 1.0", "SSP2_openres_lc_50"),
    ("REMIND-MAgPIE 2.1-4.2", "SusDev_SDP-PkBudg1000"),
    ("REMIND-MAgPIE 2.1-4.3", "DeepElec_SSP2_ HighRE_Budg900"),
    ("WITCH 5.0", "CO_Bridge"),
)

KEY_TESTING_MODEL_SCENARIOS = tuple(
    [
        *AR6_IPS,
        # Other special cases
        ("C3IAM 2.0", "2C-hybrid"),
        ("DNE21+ V.14E1", "EMF30_BCOC-EndU"),
    ]
)


def get_key_testing_model_scenario_parameters() -> pytest.MarkDecorator:
    try:
        import pytest
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_key_testing_model_scenario_parameters", requirement="pytest"
        ) from exc

    return pytest.mark.parametrize(
        "model, scenario",
        [(model, scenario) for model, scenario in KEY_TESTING_MODEL_SCENARIOS],
    )


@functools.cache
def get_ar6_all_emissions(
    model: str, scenario: str, processed_ar6_output_data_dir: Path
) -> pd.DataFrame:
    """
    Get all emissions from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    processed_ar6_output_data_dir
        Directory in which the AR6 was processed into individual model-scenario files

        (In the repo, see `tests/regression/ar6/convert_ar6_res_to_checking_csvs.py`.)

    Returns
    -------
    :
        All emissions from AR6 for `model`-`scenario`
    """
    filename_emissions = f"ar6_scenarios__{model}__{scenario}__emissions.csv"
    filename_emissions = filename_emissions.replace("/", "_").replace(" ", "_")
    emissions_file = processed_ar6_output_data_dir / filename_emissions

    res = load_timeseries_csv(
        emissions_file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_column_type=int,
    )

    return res


@functools.cache
def get_ar6_raw_emissions(
    model: str, scenario: str, processed_ar6_output_data_dir: Path
) -> pd.DataFrame:
    """
    Get all raw emissions from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    processed_ar6_output_data_dir
        Directory in which the AR6 was processed into individual model-scenario files

        (In the repo, see `tests/regression/ar6/convert_ar6_res_to_checking_csvs.py`.)

    Returns
    -------
    :
        All raw emissions from AR6 for `model`-`scenario`
    """
    try:
        from pandas_indexing.selectors import ismatch
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_ar6_raw_emissions", requirement="pandas_indexing"
        ) from exc

    all_emissions = get_ar6_all_emissions(
        model=model,
        scenario=scenario,
        processed_ar6_output_data_dir=processed_ar6_output_data_dir,
    )
    res: pd.DataFrame = all_emissions.loc[ismatch(variable="Emissions**")].dropna(
        how="all", axis="columns"
    )

    return res


@functools.cache
def get_ar6_harmonised_emissions(
    model: str, scenario: str, processed_ar6_output_data_dir: Path
) -> pd.DataFrame:
    """
    Get all harmonised emissions from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    processed_ar6_output_data_dir
        Directory in which the AR6 was processed into individual model-scenario files

        (In the repo, see `tests/regression/ar6/convert_ar6_res_to_checking_csvs.py`.)

    Returns
    -------
    :
        All harmonised emissions from AR6 for `model`-`scenario`
    """
    try:
        from pandas_indexing.selectors import ismatch
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_ar6_harmonised_emissions", requirement="pandas_indexing"
        ) from exc

    all_emissions = get_ar6_all_emissions(
        model=model,
        scenario=scenario,
        processed_ar6_output_data_dir=processed_ar6_output_data_dir,
    )
    res: pd.DataFrame = all_emissions.loc[ismatch(variable="**Harmonized**")].dropna(
        how="all", axis="columns"
    )

    return res


@functools.cache
def get_ar6_infilled_emissions(
    model: str, scenario: str, processed_ar6_output_data_dir: Path
) -> pd.DataFrame:
    """
    Get all infilled emissions from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    processed_ar6_output_data_dir
        Directory in which the AR6 was processed into individual model-scenario files

        (In the repo, see `tests/regression/ar6/convert_ar6_res_to_checking_csvs.py`.)

    Returns
    -------
    :
        All infilled emissions from AR6 for `model`-`scenario`
    """
    try:
        from pandas_indexing.selectors import ismatch
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_ar6_harmonised_emissions", requirement="pandas_indexing"
        ) from exc

    all_emissions = get_ar6_all_emissions(
        model=model,
        scenario=scenario,
        processed_ar6_output_data_dir=processed_ar6_output_data_dir,
    )
    res: pd.DataFrame = all_emissions.loc[ismatch(variable="**Infilled**")].dropna(
        how="all", axis="columns"
    )

    return res


def assert_frame_equal(
    res: pd.DataFrame, exp: pd.DataFrame, rtol: float = 1e-8, **kwargs: Any
) -> None:
    """
    Assert two [pd.DataFrame][pandas.DataFrame]'s are equal.

    This is a very thin wrapper around
    [pd.testing.assert_frame_equal][pandas.testing.assert_frame_equal]
    that makes some use of [pandas_indexing][]
    to give slightly nicer and clearer errors.

    Parameters
    ----------
    res
        Result

    exp
        Expected value

    rtol
        Relative tolerance

    **kwargs
        Passed to [pd.testing.assert_frame_equal][pandas.testing.assert_frame_equal]

    Raises
    ------
    AssertionError
        The frames aren't equal
    """
    for idx_name in res.index.names:
        idx_diffs = res.pix.unique(idx_name).symmetric_difference(  # type: ignore
            exp.pix.unique(idx_name)  # type: ignore
        )
        if not idx_diffs.empty:
            msg = f"Differences in the {idx_name} (res on the left): {idx_diffs=}"
            raise AssertionError(msg)

    pd.testing.assert_frame_equal(
        res.T, exp.T, check_like=True, check_exact=False, rtol=rtol, **kwargs
    )
