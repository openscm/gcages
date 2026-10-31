"""
Extension of the CMIP7 ScenarioMIP marker scenarios beyond 2100

This is a port of `notebooks/5191_extension.py` from
https://github.com/iiasa/emissions_harmonization_historical
(commit 376987dbe3b66fcd23537519a16771e21cba6d1e),
which is based on FLEX (https://github.com/benmsanderson/FLEX, MIT licence).
The logic is unchanged, so that we reproduce the CMIP7 ScenarioMIP extensions exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from attrs import define, field
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.pandas_indexing_helpers import import_pandas_indexing
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.workflows.cmip7_scenariomip.extension.afolu import (
    get_cumulative_afolu,
    get_cumulative_afolu_fill_from_hist,
)
from gcages.workflows.cmip7_scenariomip.extension.cdr import (
    add_removals_and_positive_fossil_emissions_to_historical,
    extend_cdr_components_vectorized,
    get_2100_compound_composition_co2,
)
from gcages.workflows.cmip7_scenariomip.extension.fossil_co2 import (
    extend_co2_for_scen_storyline,
)
from gcages.workflows.cmip7_scenariomip.extension.functionality import (
    extend_linear_rampdown,
    sigmoid_function,
)
from gcages.workflows.cmip7_scenariomip.extension.non_co2 import (
    do_single_component_for_scenario_model_regionally,
)
from gcages.workflows.cmip7_scenariomip.extension.regional import (
    extend_regional_for_missing,
    merge_historical_future_timeseries,
)
from gcages.workflows.cmip7_scenariomip.extension.utils import (
    fix_up_and_concatenate_extensions,
    glue_with_historical,
    interpolate_to_annual,
)

if TYPE_CHECKING:
    from gcages.workflows.cmip7_scenariomip.harmonisation import (
        CMIP7ScenarioMIPWorkflowHarmonisedResult,
    )

FUTURE_START_YEAR = 2023
"""First year of the scenarios (i.e. the harmonisation year)"""

SCENARIO_END_YEAR = 2100
"""Last year of the scenarios as reported by the models"""

EXTENSION_END_YEAR = 2500
"""Last year of the extensions"""

INTERNAL_VARIABLES = (
    "Emissions|CO2|Gross Positive Emissions",
    "Emissions|CO2|Gross Removals",
)
"""Variables that are only used internally while extending"""

INDEX_LEVELS = ["model", "scenario", "region", "variable", "unit"]
"""Index levels of the data we extend"""


@define(frozen=True)
class CMIP7ScenarioMIPExtensionConfig:
    """
    Configuration for the extension of a marker scenario
    """

    fossil_co2_storyline: tuple[Any, ...]
    """
    Storyline for the extension of (total) CO2 emissions

    The first element is the storyline type,
    the rest are its parameters
    (see [extend_co2_for_scen_storyline][gcages.workflows.cmip7_scenariomip.extension.fossil_co2.extend_co2_for_scen_storyline]).
    """

    removal_strategy: tuple[Any, ...]
    """
    Strategy for splitting fossil CO2 into gross positive emissions and removals

    `("POS",)`: removals stay constant at their 2100 level.
    `("NEG", decay_timescale, offset)`: gross positive emissions decay
    following a sigmoid, removals are the residual.
    """

    ch4_target: float | None
    """Long-term target for global CH4 emissions"""

    sulfur_target: float | None
    """Long-term target for global Sulfur emissions"""


CMIP7_SCENARIOMIP_EXTENSION_CONFIGS: dict[
    tuple[str, str], CMIP7ScenarioMIPExtensionConfig
] = {
    # The order matters, it is the order in which the scenarios are extended
    # VL
    ("REMIND-MAgPIE 3.5-4.11", "SSP1 - Very Low Emissions"): (
        CMIP7ScenarioMIPExtensionConfig(
            fossil_co2_storyline=("ECS", 2200, -3.5e3, 2450, 2500, 20, 20),
            removal_strategy=("NEG", 100, 60),
            ch4_target=95.0,
            sulfur_target=20.0,
        )
    ),
    # LN
    ("AIM 3.0", "SSP2 - Low Overshoot_a"): CMIP7ScenarioMIPExtensionConfig(
        fossil_co2_storyline=("ECS", 2120, -24e3, 2200, 2300, 20, 20),
        removal_strategy=("NEG", 50, 100),
        ch4_target=150.0,
        sulfur_target=10.0,
    ),
    # L
    ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Emissions"): (
        CMIP7ScenarioMIPExtensionConfig(
            fossil_co2_storyline=("ECS", 2160, None, 2160, 2260, 40, 20),
            removal_strategy=("NEG", 50, 50),
            ch4_target=95.0,
            sulfur_target=None,
        )
    ),
    # ML
    ("COFFEE 1.6", "SSP2 - Medium-Low Emissions"): CMIP7ScenarioMIPExtensionConfig(
        fossil_co2_storyline=("ECS", 2150, -13e3, 2230, 2300, 20, 20),
        removal_strategy=("NEG", 100, 0),
        ch4_target=120.0,
        sulfur_target=20.0,
    ),
    # M
    ("IMAGE 3.4", "SSP2 - Medium Emissions"): CMIP7ScenarioMIPExtensionConfig(
        fossil_co2_storyline=("CS", 2100, 2240, 20, 20),
        removal_strategy=("POS",),
        ch4_target=450.0,
        sulfur_target=20.0,
    ),
    # H
    ("GCAM 8s", "SSP3 - High Emissions"): CMIP7ScenarioMIPExtensionConfig(
        fossil_co2_storyline=("ECS", 2150, None, 2175, 2300, 20, 20),
        removal_strategy=("POS",),
        ch4_target=520.0,
        sulfur_target=50.0,
    ),
    # HL
    ("WITCH 6.0", "SSP5 - Medium-Low Emissions_a"): CMIP7ScenarioMIPExtensionConfig(
        fossil_co2_storyline=("ECS", 2150, -22e3, 2200, 2300, 20, 20),
        removal_strategy=("NEG", 80, 20),
        ch4_target=110.0,
        sulfur_target=10.0,
    ),
}
"""
Configuration of the extension of each marker scenario

Keys are (model, scenario) using the scenario names as submitted.
"""


def convert_variable_names(
    df: pd.DataFrame,
    from_convention: SupportedNamingConventions,
    to_convention: SupportedNamingConventions,
) -> pd.DataFrame:
    """Convert variable names between naming conventions"""
    return update_index_levels_func(
        df,
        {
            "variable": lambda x: convert_variable_name(
                x, from_convention=from_convention, to_convention=to_convention
            )
        },
    )


def to_extension_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data to the format expected by the extension functions

    i.e. our index levels, integer years as columns and no columns name.
    """
    res = df.reorder_levels(INDEX_LEVELS).copy()
    res.columns = res.columns.astype(int)
    res.columns.name = None

    return res


@define
class CMIP7ScenarioMIPExtender:
    """
    Extender of emissions beyond 2100 following the logic used in CMIP7 ScenarioMIP

    Only the scenarios in [configs][(c).] are supported.
    If you want exactly the same behaviour as in CMIP7 ScenarioMIP,
    use the default [configs][(c).] (the configurations of the marker scenarios).
    """

    historical_emissions: pd.DataFrame
    """
    Historical emissions

    These should use CMIP7 ScenarioMIP's naming convention
    and have model and scenario levels
    (the CO2 AFOLU history is identified using these).
    """

    configs: Mapping[tuple[str, str], CMIP7ScenarioMIPExtensionConfig] = field(
        factory=lambda: dict(CMIP7_SCENARIOMIP_EXTENSION_CONFIGS)
    )
    """Configuration of the extension of each (model, scenario)"""

    co2_afolu_history_model: str = "GCB-extended"
    """Model of the CO2 AFOLU history in `historical_emissions`"""

    def __call__(
        self,
        infilled: pd.DataFrame,
        harmonised: CMIP7ScenarioMIPWorkflowHarmonisedResult,
    ) -> pd.DataFrame:
        """
        Extend the emissions

        Parameters
        ----------
        infilled
            Infilled emissions (using gcages' naming convention)

        harmonised
            Harmonised emissions

            The region-sector emissions are used to split the extensions
            between sectors and regions.

        Returns
        -------
        :
            Extended emissions from 2023 to 2500 (using gcages' naming convention)

        Raises
        ------
        ValueError
            There is no configuration for some of the scenarios in `infilled`
        """
        pix = import_pandas_indexing("CMIP7ScenarioMIPExtender")

        model_scenarios = (
            infilled.index.droplevel(
                infilled.index.names.difference(["model", "scenario"])  # type: ignore # pandas-stubs confused
            )
            .drop_duplicates()
            .tolist()
        )
        missing = [ms for ms in model_scenarios if ms not in self.configs]
        if missing:
            msg = (
                f"No extension configuration for {missing}. "
                f"Available configurations: {list(self.configs)}"
            )
            raise ValueError(msg)

        # Keep the order of the configurations
        configs = {k: v for k, v in self.configs.items() if k in model_scenarios}

        scenarios_complete_global = to_extension_format(
            convert_variable_names(
                infilled,
                from_convention=SupportedNamingConventions.GCAGES,
                to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            )
        )
        scenarios_regional = pix.concat(
            [
                pix.assignlevel(
                    to_extension_format(harmonised.region_sector), workflow="gridding"
                ),
                pix.assignlevel(
                    to_extension_format(harmonised.global_only), workflow="global"
                ),
                pix.assignlevel(
                    to_extension_format(
                        convert_variable_names(
                            harmonised.for_infilling,
                            from_convention=SupportedNamingConventions.GCAGES,
                            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                        )
                    ),
                    workflow="for_scms",
                ),
            ]
        )
        history = to_extension_format(self.historical_emissions)

        extended = extend(
            scenarios_complete_global=scenarios_complete_global,
            scenarios_regional=scenarios_regional,
            history=history,
            configs=configs,
            co2_afolu_history_model=self.co2_afolu_history_model,
        )

        # Only keep the scenarios (not the history we glued on)
        # in the same format as the input
        extended = extended.loc[:, float(FUTURE_START_YEAR) :]  # type: ignore # pandas-stubs confused
        extended.columns = extended.columns.astype(int)
        extended.columns.name = "year"

        res = convert_variable_names(
            extended.reset_index("workflow", drop=True).reorder_levels(
                infilled.index.names
            ),
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        )

        return res


def extend(  # noqa: PLR0912, PLR0915
    scenarios_complete_global: pd.DataFrame,
    scenarios_regional: pd.DataFrame,
    history: pd.DataFrame,
    configs: Mapping[tuple[str, str], CMIP7ScenarioMIPExtensionConfig],
    co2_afolu_history_model: str,
) -> pd.DataFrame:
    """
    Extend emissions

    Port of the main logic of `notebooks/5191_extension.py`
    from emissions_harmonization_historical.

    Parameters
    ----------
    scenarios_complete_global
        Complete (i.e. infilled) global emissions

    scenarios_regional
        Harmonised emissions, with a "workflow" level
        ("gridding" for the region-sector emissions,
        "global" for the global-only emissions
        and "for_scms" for the emissions that were infilled)

    history
        Historical emissions

    configs
        Configuration of the extension of each (model, scenario)

    co2_afolu_history_model
        Model of the CO2 AFOLU history in `history`

    Returns
    -------
    :
        Continuous World emissions from history to 2500
        (using CMIP7 ScenarioMIP's naming convention)
    """
    pix = import_pandas_indexing("extend")

    unique_model_scenario_pairs = scenarios_complete_global.index.droplevel(
        ["region", "variable", "unit"]
    ).drop_duplicates()

    scenarios_regional = (
        scenarios_regional.sort_index(axis="columns").T.interpolate("index").T
    )

    fractions_fossil_total = {}
    for model, scen in unique_model_scenario_pairs.to_list():
        scen_here = scenarios_regional.loc[
            pix.ismatch(scenario=scen, model=model, variable="Emissions|CO2**")
        ]
        fractions_list = get_2100_compound_composition_co2(scen_here[SCENARIO_END_YEAR])
        fractions_fossil_total[(model, scen)] = {
            "fractions_tot_fossil": fractions_list[0],
            "fractions_cdr": fractions_list[1],
            "fractions_fossil_nocdr": fractions_list[2],
        }

    # AFOLU
    cumulative_history_afolu = get_cumulative_afolu(
        history, co2_afolu_history_model, "historical"
    )
    afolu_l = []
    for model, scen in configs:
        scen_afolu = scenarios_complete_global.loc[
            pix.ismatch(variable="**CO2|AFOLU", model=model, scenario=scen)
        ]
        scen_full = glue_with_historical(
            scen_afolu, history.loc[pix.ismatch(variable="Emissions|CO2|AFOLU")]
        )
        cumulative_2100 = get_cumulative_afolu_fill_from_hist(
            scen_afolu, model, scen, cumulative_history_afolu
        )
        afolu_years: np.ndarray[Any, Any] = np.arange(
            cumulative_2100.columns[0], EXTENSION_END_YEAR + 1
        )
        em_ext_linear_ramp_down = extend_linear_rampdown(
            scen_full.values[0, :], afolu_years
        )
        afolu_l.append(
            pd.DataFrame(
                data=[em_ext_linear_ramp_down],
                columns=afolu_years,
                index=scen_afolu.index,
            )
        )

    df_afolu = pd.concat(afolu_l)

    # Non-CO2
    non_co2_l = []
    for variable in pix.uniquelevel(scenarios_complete_global, "variable").values:
        if variable.startswith("Emissions|CO2"):
            continue

        if history.loc[pix.ismatch(variable=f"{variable}")].shape[0] < 1:
            continue

        for (model, scen), config in configs.items():
            if variable == "Emissions|CH4":
                global_target = config.ch4_target
            elif variable == "Emissions|Sulfur":
                global_target = config.sulfur_target
            else:
                global_target = None

            df_comp_scen_model = do_single_component_for_scenario_model_regionally(
                scen,
                model,
                variable,
                scenarios_regional,
                scenarios_complete_global,
                history,
                global_target=global_target,
            )
            if "workflow" not in df_comp_scen_model.index.names:
                msg = f"No workflow level for {variable=}, {model=}, {scen=}"
                raise AssertionError(msg)

            non_co2_l.append(df_comp_scen_model)

    df_all = pix.concat(non_co2_l)

    # Fossil CO2
    fossil_l = []
    for (model, scen), config in configs.items():
        co2_fossil = interpolate_to_annual(
            scenarios_complete_global.loc[
                pix.ismatch(
                    variable="Emissions|CO2|Energy and Industrial Processes",
                    model=model,
                    scenario=scen,
                )
            ]
        )
        co2_fossil = co2_fossil.loc[:, FUTURE_START_YEAR:]
        co2_afolu = df_afolu.loc[pix.ismatch(model=model, scenario=scen)]
        _, co2_fossil_extend, extend_years = extend_co2_for_scen_storyline(
            co2_afolu, co2_fossil, list(config.fossil_co2_storyline)
        )
        fossil_l.append(
            pd.DataFrame(
                data=[co2_fossil_extend], columns=extend_years, index=co2_fossil.index
            )
        )

    fossil_extension_df = pd.concat(fossil_l)

    # Removal disaggregation
    def get_regional(variable: str, **kwargs: str) -> pd.DataFrame:
        res: pd.DataFrame = interpolate_to_annual(
            scenarios_regional.loc[pix.ismatch(variable=variable, **kwargs)]
        )
        return res

    co2_beccs = get_regional("Emissions|CO2|BECCS")
    co2_dacc = get_regional("Emissions|CO2|Direct Air Capture")
    co2_ocean = get_regional("Emissions|CO2|Ocean")
    co2_ew = get_regional("Emissions|CO2|Enhanced Weathering")
    co2_biochar = get_regional("Emissions|CO2|Biochar")
    co2_soil = get_regional("Emissions|CO2|Soil Carbon Management")
    co2_othercdr = get_regional("Emissions|CO2|Other CDR")
    co2_ffi = get_regional(
        "Emissions|CO2|Energy and Industrial Processes", workflow="for_scms"
    )
    co2_cdr = (
        co2_dacc
        + co2_ocean.values
        + co2_ew.values
        + co2_beccs.values
        + co2_biochar.values
        + co2_soil.values
        + co2_othercdr.values
    )
    co2_cdr = pix.assignlevel(co2_cdr, variable="Emissions|CO2|Gross Removals")
    global_cdr = co2_cdr.groupby(["model", "scenario", "variable", "unit"]).sum()

    # Gross positive emissions
    co2_ffi_grouped = co2_ffi.groupby(["model", "scenario", "variable", "unit"]).sum()
    gross_positive_l = []
    for model, scenario, var_cdr, unit in global_cdr.index:
        matching_ffi = co2_ffi_grouped.loc[
            (co2_ffi_grouped.index.get_level_values("model") == model)
            & (co2_ffi_grouped.index.get_level_values("scenario") == scenario)
            & (co2_ffi_grouped.index.get_level_values("unit") == unit)
        ]
        if len(matching_ffi) > 0:
            cdr_data = global_cdr.loc[(model, scenario, var_cdr, unit)]  # type: ignore # pandas-stubs confused
            ffi_data = matching_ffi.iloc[0]
            gross_positive_l.append(
                (
                    (model, scenario, "Emissions|CO2|Gross Positive Emissions", unit),
                    -cdr_data + ffi_data,
                )
            )

    indices, data_rows = zip(*gross_positive_l)
    co2_gross_positive = pd.DataFrame(
        data=list(data_rows),
        index=pd.MultiIndex.from_tuples(indices, names=global_cdr.index.names),
    )

    # Extension of gross positive emissions and removals
    years_extension = np.arange(SCENARIO_END_YEAR + 1, EXTENSION_END_YEAR + 1)
    co2_gross_positive_ext = pd.concat(
        [
            co2_gross_positive,
            pd.DataFrame(
                np.nan, index=co2_gross_positive.index, columns=years_extension
            ),
        ],
        axis=1,
    )
    global_cdr_ext = pd.concat(
        [
            global_cdr,
            pd.DataFrame(np.nan, index=global_cdr.index, columns=years_extension),
        ],
        axis=1,
    )

    removal_strategy_map = {
        scen: config.removal_strategy for (_, scen), config in configs.items()
    }
    for idx in co2_gross_positive.index:
        model, scenario, _, unit = idx
        if scenario not in removal_strategy_map:
            continue

        strategy_info = removal_strategy_map[scenario]
        strategy = strategy_info[0]

        try:
            fossil_row = fossil_extension_df.loc[
                (  # type: ignore # pandas-stubs confused
                    model,
                    scenario,
                    "World",
                    "Emissions|CO2|Energy and Industrial Processes",
                    unit,
                )
            ]
        except KeyError:
            continue

        cdr_idx = (model, scenario, "Emissions|CO2|Gross Removals", unit)
        cdr_2100 = global_cdr.loc[cdr_idx, float(SCENARIO_END_YEAR)]
        gross_pos_2100 = co2_gross_positive.loc[idx, float(SCENARIO_END_YEAR)]

        if strategy == "POS":
            # Removals remain constant,
            # gross positive emissions adjust to match the fossil trajectory
            cdr_extension = np.full(len(years_extension), cdr_2100)
            fossil_vals = fossil_row[years_extension].values
            gross_pos_extension = fossil_vals - cdr_2100

        elif strategy == "NEG":
            # Gross positive emissions follow a sigmoid decay,
            # removals are the residual to match the fossil trajectory
            decay_timescale = strategy_info[1]
            offset = strategy_info[2]
            offset_shift = (1.25 * decay_timescale) / 2
            gross_pos_extension = sigmoid_function(
                0,
                gross_pos_2100,  # type: ignore # pandas-stubs confused
                years_extension[0] + offset - offset_shift,
                years_extension[0] + offset + offset_shift,
                years_extension,
                adjust_from=True,
            )
            fossil_vals = fossil_row[years_extension].values
            cdr_extension = fossil_vals - gross_pos_extension

        else:
            continue

        co2_gross_positive_ext.loc[idx, years_extension] = gross_pos_extension  # type: ignore # pandas-stubs confused
        global_cdr_ext.loc[cdr_idx, years_extension] = cdr_extension  # type: ignore # pandas-stubs confused

    # Regional CDR components, keeping the 2100 ratios
    extended_cdr_components = extend_cdr_components_vectorized(
        {
            "BECCS": co2_beccs,
            "DACCS": co2_dacc,
            "Ocean": co2_ocean,
            "Enhanced_Weathering": co2_ew,
            "Biochar": co2_biochar,
            "Soil_Management": co2_soil,
            "Other_CDR": co2_othercdr,
        },
        global_cdr_ext,
    )

    # Combine everything
    df_everything = fix_up_and_concatenate_extensions(
        {
            "fossil_extension": fossil_extension_df,
            "afolu_extensions": df_afolu,
            "non_co2_extensions": df_all,
            "gross_positive_extensions": co2_gross_positive_ext,
            "cdr_extensions": global_cdr_ext,
            "beccs_extensions": extended_cdr_components["BECCS"],
            "dacc_extensions": extended_cdr_components["DACCS"],
            "ocean_extensions": extended_cdr_components["Ocean"],
            "ew_extensions": extended_cdr_components["Enhanced_Weathering"],
            "biochar_extensions": extended_cdr_components["Biochar"],
            "soil_extensions": extended_cdr_components["Soil_Management"],
            "other_cdr_extensions": extended_cdr_components["Other_CDR"],
        }
    )
    df_everything = extend_regional_for_missing(
        df_everything, scenarios_regional, fractions_fossil_total
    )
    df_everything = df_everything[~df_everything.index.duplicated(keep="first")]
    df_everything = df_everything.rename(
        columns={col: float(col) for col in df_everything.columns if str(col).isdigit()}
    )

    history = add_removals_and_positive_fossil_emissions_to_historical(history)
    continuous: pd.DataFrame = merge_historical_future_timeseries(
        history, df_everything
    )
    continuous = continuous.loc[
        ~continuous.index.get_level_values("variable").isin(INTERNAL_VARIABLES)
    ]

    return continuous
