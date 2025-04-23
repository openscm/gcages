"""
Code to support our tests

This is here, rather than in our `tests` directory
because of the issues that come
when you turn your tests into a package using `__init__.py` files
(for details, see https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#choosing-an-import-mode).
"""

from __future__ import annotations

import functools
import itertools
import os
import platform
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
from pandas_openscm.grouping import groupby_except
from pandas_openscm.io import load_timeseries_csv

from gcages.exceptions import MissingOptionalDependencyError
from gcages.typing import NP_ARRAY_OF_FLOAT_OR_INT

if TYPE_CHECKING:
    import pytest

RNG = np.random.default_rng()

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
        Directory in which the AR6 output was processed into model-scenario files

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
            "get_ar6_infilled_emissions", requirement="pandas_indexing"
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


@functools.cache
def get_ar6_temperature_outputs(
    model: str, scenario: str, processed_ar6_output_data_dir: Path, dropna: bool = True
) -> pd.DataFrame:
    """
    Get temperature outputs we've downloaded from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    processed_ar6_output_data_dir
        Directory in which the AR6 output was processed into model-scenario files

        (In the repo, see `tests/regression/ar6/convert_ar6_res_to_checking_csvs.py`.)

    dropna
        Drop time columns that only contain NaN

    Returns
    -------
    :
        All temperature outputs we've downloaded from AR6 for `model`-`scenario`
    """
    filename_temperatures = f"ar6_scenarios__{model}__{scenario}__temperatures.csv"
    filename_temperatures = filename_temperatures.replace("/", "_").replace(" ", "_")
    temperatures_file = processed_ar6_output_data_dir / filename_temperatures

    res = load_timeseries_csv(
        temperatures_file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_column_type=int,
    )
    if dropna:
        res = res.dropna(axis="columns", how="all")

    return res


@functools.cache
def get_ar6_metadata_outputs(
    model: str,
    scenario: str,
    ar6_output_data_dir: Path,
    filename: str = "AR6_Scenarios_Database_metadata_indicators_v1.1_meta.csv",
) -> pd.DataFrame:
    """
    Get metadata from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    ar6_output_data_dir
        Directory in which the AR6 output was saved

    Returns
    -------
    :
        Metadata from AR6 for `model`-`scenario`
    """
    res = load_timeseries_csv(
        ar6_output_data_dir / filename,
        lower_column_names=False,
        index_columns=["Model", "Scenario"],
    ).loc[[(model, scenario)]]

    res.index = res.index.rename({"Model": "model", "Scenario": "scenario"})

    return res


def guess_magicc_exe_path() -> Path:
    """
    Guess the path to the MAGICC executable

    Uses the `MAGICC_EXECUTABLE_7` environment variable.
    If that isn't set, it guesses.

    Returns
    -------
    :
        Path to the MAGICC executable

    Raises
    ------
    FileNotFoundError
        The guessed path to the MAGICC executable does not exist
    """
    env_var = os.environ.get("MAGICC_EXECUTABLE_7", None)
    if env_var is not None:
        return Path(env_var)

    guess = None
    guess_path = (
        Path(__file__).parents[2]
        / "tests"
        / "regression"
        / "ar6"
        / "ar6-workflow-inputs"
        / "magicc-v7.5.3"
        / "bin"
    )
    if platform.system() == "Darwin":
        if platform.processor() == "arm":
            guess = guess_path / "magicc-darwin-arm64"

    elif platform.system() == "Linux":
        guess = guess_path / "magicc"

    elif platform.system() == "Windows":
        guess = guess_path / "magicc.exe"

    if guess is not None:
        if guess.exists():
            return guess

        msg = f"Guessed that the MAGICC executable was in: {guess}"
        raise FileNotFoundError(msg)

    msg = "No guess about where the MAGICC executable is for your system"
    raise FileNotFoundError(msg)


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
    try:
        from pandas_indexing.core import uniquelevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "assert_frame_equal", requirement="pandas_indexing"
        ) from exc

    for idx_name in res.index.names:
        idx_diffs = uniquelevel(res, idx_name).symmetric_difference(  # type: ignore
            uniquelevel(exp, idx_name)  # type: ignore
        )
        if not idx_diffs.empty:
            msg = f"Differences in the {idx_name} (res on the left): {idx_diffs=}"
            raise AssertionError(msg)

    pd.testing.assert_frame_equal(
        res.reorder_levels(exp.index.names).T,
        exp.T,
        check_like=True,
        check_exact=False,
        rtol=rtol,
        **kwargs,
    )


def compare_close(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    rtol: float = 1e-8,
    **kwargs: Any,
) -> pd.DataFrame:
    left_stacked = left.stack()
    left_stacked.name = left_name

    right_stacked = right.stack()
    right_stacked.name = right_name

    left_stacked_aligned, right_stacked_aligned = left_stacked.align(right_stacked)
    differences_locator = ~np.isclose(
        left_stacked_aligned, right_stacked_aligned, rtol=rtol, **kwargs
    )

    res = pd.concat(
        [
            left_stacked_aligned[differences_locator],
            right_stacked_aligned[differences_locator],
        ],
        axis="columns",
    )

    return res


# TODO: split this out into separate module
def unstack_sector(indf: pd.DataFrame, time_name: str = "year") -> pd.DataFrame:
    try:
        from pandas_indexing.core import extractlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "unstack_sector", requirement="pandas_indexing"
        ) from exc

    res = (
        extractlevel(indf, variable="{table}|{species}|{sectors}")
        .unstack("sectors")
        .stack(time_name, future_stack=True)
    )

    return res


def aggregate_up_sectors(indf, copy=False):
    # TODO: doc string
    res = indf
    if copy:
        res = res.copy()

    aggregations = set(["|".join(c.split("|")[:-1]) for c in res if "|" in c])
    # Have to aggregate lowest level sectors first
    aggregations_sorted = sorted(aggregations, key=lambda x: x.count("|"))[::-1]
    for aggregation in aggregations_sorted:
        if aggregation in res:
            msg = f"{aggregation} already in indf?!"
            raise KeyError(msg)

        contributing = []
        for c in res:
            if not c.startswith(f"{aggregation}|"):
                continue

            split = c.split(f"{aggregation}|")
            if "|" in split[-1]:
                # going too many levels deep
                continue

            contributing.append(c)

        # Sum with any data we have,
        # don't require complete levels for all sectors
        res[aggregation] = res[contributing].sum(axis="columns", min_count=1)

    return res


def get_cmip7_scenariomip_like_input_get_species_bottom_sectors_full_reporting() -> (
    tuple[str, ...]
):
    species = (
        "CO2",
        "CH4",
        "N2O",
        "BC",
        "CO",
        "NH3",
        "OC",
        "NOx",
        "Sulfur",
        "VOC",
    )
    bottom_level_sectors = (
        # Energy sector
        "Energy|Supply",
        # Industrial sector stuff
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
        # Residential commercial and other
        "Energy|Demand|Residential and Commercial and AFOFI",
        # Solvents production and application
        "Product Use",
        # Aviation stuff
        "Energy|Demand|Transportation|Domestic Aviation",
        # Other components of "Energy|Demand|Transportation"
        # (doesn't actually matter what they are,
        # as long as they can be added to domestic aviation above)
        "Energy|Demand|Transportation|Rail",
        "Energy|Demand|Bunkers|International Aviation",
        # International shipping
        "Energy|Demand|Bunkers|International Shipping",
        # Waste
        "Waste",
        # Agriculture
        "AFOLU|Agriculture",
        # Burning sectors
        "AFOLU|Agricultural Waste Burning",
        "AFOLU|Land|Fires|Forest Burning",
        "AFOLU|Land|Fires|Grassland Burning",
        "AFOLU|Land|Fires|Peat Burning",
        # Imperfect but put these in to test agriculture aggregation too for now
        # (we don't have a better source to harmonise too,
        # except for CO2 but they don't use CO2 LULUCF
        # emissions as input anyway, they use land-use change patterns).
        # (For SCMs, this also doesn't matter as it all gets rolled up to
        # "AFOLU" anyway because SCMs aren't able to handle the difference,
        # except maybe OSCAR but that isn't in OpenSCM-Runner
        # so we can't guess anything to do with OSCAR for now anyway.)
        "AFOLU|Land|Land Use and Land-Use Change",
        "AFOLU|Land|Harvested Wood Products",
        "AFOLU|Land|Other",
        "AFOLU|Land|Wetlands",
    )
    res = tuple(
        (
            (species, sectors)
            for species, sectors in itertools.product(species, bottom_level_sectors)
        )
    )

    return res


def get_cmip7_scenariomip_like_input(
    timesteps: NP_ARRAY_OF_FLOAT_OR_INT = np.arange(2015, 2100 + 1, 5),
    model: str = "model_a",
    scenario: str = "scenario_a",
    regions: Iterable[str] = ("China", "Pacific OECD"),
    get_species_bottom_level_sectors: Callable[
        [], Iterable[str]
    ] = get_cmip7_scenariomip_like_input_get_species_bottom_sectors_full_reporting,
) -> pd.DataFrame:
    try:
        from pandas_indexing.core import assignlevel, concat, formatlevel
        from pandas_indexing.selectors import ismatch
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_cmip7_scenariomip_like_input", requirement="pandas_indexing"
        ) from exc

    start_index = pd.MultiIndex.from_tuples(
        [
            (
                model,
                scenario,
                "Emissions",
                species,
                sectors,
                f"{model}|{region}",
                timestep,
            )
            for model, scenario, (
                species,
                sectors,
            ), region, timestep in itertools.product(
                [model],
                [scenario],
                get_species_bottom_level_sectors(),
                regions,
                timesteps,
            )
        ],
        names=["model", "scenario", "table", "species", "sectors", "region", "year"],
    )
    pds = pd.Series(
        # Use random so conservation is not easily done by accident
        RNG.random(start_index.shape[0]),
        index=start_index,
    )

    hierarchy = (
        aggregate_up_sectors(pds.unstack("sectors")).stack("sectors").unstack("year")
    )
    top_level = hierarchy.loc[ismatch(sectors="*")]
    totals = groupby_except(top_level, "sectors").sum()
    df = concat(
        [
            formatlevel(hierarchy, variable="{table}|{species}|{sectors}", drop=True),
            formatlevel(totals, variable="{table}|{species}", drop=True),
        ]
    )

    def get_unit(v):
        species = v.split("|")[1]
        unit_map = {
            "BC": "Mt BC/yr",
            "CH4": "Mt CH4/yr",
            "CO": "Mt CO/yr",
            "CO2": "Mt CO2/yr",
            "N2O": "kt N2O/yr",
            "NH3": "Mt NH3/yr",
            "NOx": "Mt NO2/yr",
            "OC": "Mt OC/yr",
            "Sulfur": "Mt SO2/yr",
            "VOC": "Mt VOC/yr",
        }
        return unit_map[species]

    df["unit"] = df.index.get_level_values("variable").map(get_unit)
    df = df.set_index("unit", append=True)
    world_sum = assignlevel(
        df.groupby(df.index.names.difference(["region"])).sum(), region="World"
    )
    df = pd.concat(
        [
            world_sum,
            df.reorder_levels(world_sum.index.names),
        ]
    )

    global_only_base = pd.DataFrame(
        RNG.random(timesteps.size)[np.newaxis, :],
        columns=df.columns,
        index=start_index.droplevel(
            ["region", "year", "species", "sectors"]
        ).drop_duplicates(),
    )
    global_only_l = []
    for global_only_gas, unit in [
        ("HFC|HFC23", "kt HFC23/yr"),
        ("HFC", "kt HFC134a-equiv/yr"),
        ("HFC|HFC134a", "kt HFC134a/yr"),
        ("HFC|HFC43-10", "kt HFC43-10/yr"),
        ("PFC", "kt CF4-equiv/yr"),
        ("F-Gases", "Mt CO2-equiv/yr"),
        ("SF6", "kt SF6/yr"),
        ("CF4", "kt CF4/yr"),
        ("C2F6", "kt C2F6/yr"),
        ("C6F14", "kt C6F14/yr"),
    ]:
        tmp = global_only_base.copy()
        tmp.loc[:, :] = RNG.random(tmp.shape)
        global_only_l.append(
            formatlevel(
                assignlevel(tmp, gas=global_only_gas, unit=unit, region="World"),
                variable="{table}|{gas}",
                drop=True,
            ).reorder_levels(df.index.names)
        )
    global_only = pd.concat(global_only_l)
    df = pd.concat([df, global_only.reorder_levels(df.index.names)])
    df.columns = df.columns.astype(int)

    return df
