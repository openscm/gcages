"""
Storyline-driven extension of fossil CO2 emissions

Ported from https://github.com/iiasa/emissions_harmonization_historical
(commit 376987dbe3b66fcd23537519a16771e21cba6d1e,
`src/emissions_harmonization_historical/extensions/fossil_co2_storyline_functions.py`),
which is based on FLEX (https://github.com/benmsanderson/FLEX, MIT licence).
The logic is unchanged, so that we reproduce the CMIP7 ScenarioMIP extensions exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .functionality import (
    get_exp_targ_from_current_data,
    make_linear_function_with_smooth_transition,
)


def extend_from_start_to_stop_with_value(
    value,
    start: int,
    stop: int,
    co2_fossil_extend: np.ndarray,
    co2_total_extend: np.ndarray,
) -> list[np.ndarray, np.ndarray]:
    """
    Set total CO2 emissions to a constant value over a specified time period and adjust fossil emissions accordingly.

    This function modifies the total CO2 emissions array to have a constant value over a specified range,
    then adjusts the fossil CO2 emissions to maintain consistency (since total = fossil + AFOLU).

    Parameters
    ----------
    value : float or array-like
        The target total CO2 emissions value(s) to set for the specified period.
        Can be a scalar for constant emissions or an array for varying emissions.
    start : int
        Start index in the emissions arrays (inclusive).
    stop : int
        Stop index in the emissions arrays (exclusive).
    co2_fossil_extend : numpy.ndarray
        Array of fossil CO2 emissions to be modified in-place.
    co2_total_extend : numpy.ndarray
        Array of total CO2 emissions to be modified in-place.

    Returns
    -------
    co2_fossil_extend : numpy.ndarray
        Modified fossil CO2 emissions array.
    co2_total_extend : numpy.ndarray
        Modified total CO2 emissions array.

    Notes
    -----
    This function assumes that total CO2 = fossil CO2 + AFOLU CO2, and that the AFOLU
    component remains unchanged during the modification.
    """
    co2_fossil_extend[start:stop] = value - co2_total_extend[start:stop]
    co2_total_extend[start:stop] = value
    return co2_fossil_extend, co2_total_extend


def linear_to_target(target: float, from_val: float, start: int, stop: int, time):
    """
    Calculate linear interpolation between two values over a specified time period.

    This function computes a linear trajectory from an initial value to a target value
    over a specified time range, useful for creating transitions in emissions scenarios.

    Parameters
    ----------
    target : float
        The target value to reach at the end of the interpolation period.
    from_val : float
        The initial value at the start of the interpolation period.
    start : int or float
        The start time/year of the interpolation period.
    stop : int or float
        The end time/year of the interpolation period.
    time : int, float, or array-like
        The time point(s) at which to evaluate the linear function.
        Can be a single value or an array of time points.

    Returns
    -------
    float or numpy.ndarray
        The linearly interpolated value(s) at the specified time point(s).
        Returns a scalar if `time` is scalar, or an array if `time` is array-like.

    Examples
    --------
    >>> # Linear transition from 100 to 0 between years 2020 and 2050
    >>> linear_to_target(0, 100, 2020, 2050, 2035)
    50.0
    >>>
    >>> # Get values for multiple years
    >>> years = np.array([2020, 2030, 2040, 2050])
    >>> linear_to_target(0, 100, 2020, 2050, years)
    array([100.,  66.67,  33.33,   0.])
    """
    slope = (target - from_val) / (stop - start)
    return from_val + slope * (time - start)


def _apply_cs_storyline(  # noqa: PLR0913
    co2_fossil_extend: np.ndarray,
    co2_total_extend: np.ndarray,
    storyline: list,
    start: int,
    end: int,
    scenario_end=2100,
) -> list[np.ndarray, np.ndarray]:
    """
    Apply Constant-then-Sigmoid storyline.

    This function applies a storyline where emissions remain constant after the scenario end,
    then transition to zero using a sigmoid function.

    Parameters
    ----------
    co2_fossil_extend : np.ndarray
        Array of fossil CO2 emissions to extend.
    co2_total_extend : np.ndarray
        Array of total CO2 emissions to extend.
    storyline : list
        List of storyline parameters: [type, stop_const, end_sig, roll_in, roll_out].
    start : int
        Start year of the extension.
    end : int
        End year of the extension.
    scenario_end : int, optional
        End year of the scenario, default is 2100.

    Returns
    -------
    co2_fossil_extend : np.ndarray
        Extended fossil CO2 emissions.
    co2_total_extend : np.ndarray
        Extended total CO2 emissions.
    """
    stop_const = storyline[1]
    end_sig = storyline[2]
    roll_in = storyline[3]
    roll_out = storyline[4]
    co2_constant = co2_total_extend[scenario_end - start]
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        co2_constant,
        scenario_end + 1 - start,
        stop_const - start + 1,
        co2_fossil_extend,
        co2_total_extend,
    )
    linear_roll_extension = make_linear_function_with_smooth_transition(
        co2_total_extend[: stop_const + 1 - start],
        0,
        roll_start_length=roll_in,
        roll_end_length=roll_out,
        t_vals=np.arange(start, end_sig + 1),
        t_extend=stop_const - start,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        linear_roll_extension,
        stop_const + 1 - start,
        end_sig + 1 - start,
        co2_fossil_extend,
        co2_total_extend,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        0, end_sig + 1 - start, end + 1 - start, co2_fossil_extend, co2_total_extend
    )
    return co2_fossil_extend, co2_total_extend


def _apply_ecs_storyline(  # noqa: PLR0913
    co2_fossil_extend: np.ndarray,
    co2_total_extend: np.ndarray,
    storyline: list,
    start: int,
    end: int,
    scenario_end=2100,
):
    """
    Apply Exponential-then-Constant-then-Sigmoid storyline.

    This function applies a storyline with exponential decay, then constant, then sigmoid to zero.

    Parameters
    ----------
    co2_fossil_extend : np.ndarray
        Array of fossil CO2 emissions to extend.
    co2_total_extend : np.ndarray
        Array of total CO2 emissions to extend.
    storyline : list
        List of storyline parameters: [type, exp_end, exp_targ, sig_start, sig_end, roll_in, roll_out].
    start : int
        Start year of the extension.
    end : int
        End year of the extension.
    scenario_end : int, optional
        End year of the scenario, default is 2100.

    Returns
    -------
    co2_fossil_extend : np.ndarray
        Extended fossil CO2 emissions.
    co2_total_extend : np.ndarray
        Extended total CO2 emissions.
    """
    exp_end = storyline[1]
    exp_targ = storyline[2]
    sig_start = storyline[3]

    if exp_targ is None:
        exp_targ = get_exp_targ_from_current_data(
            co2_total_extend[: scenario_end + 1 - start], exp_end - start
        )
    sig_end = storyline[4]
    roll_in = storyline[5]
    roll_out = storyline[6]

    linear_decay_total = make_linear_function_with_smooth_transition(
        co2_total_extend[: scenario_end + 1 - start],
        exp_targ,
        20,
        20,
        np.arange(start, exp_end + 1),
        scenario_end - start,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        linear_decay_total,
        scenario_end - start + 1,
        exp_end - start + 1,
        co2_fossil_extend,
        co2_total_extend,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        exp_targ,
        exp_end - start + 1,
        sig_start - start + 1,
        co2_fossil_extend,
        co2_total_extend,
    )
    linear_total_extension = make_linear_function_with_smooth_transition(
        co2_total_extend[: sig_start - start + 1],
        0,
        roll_in,
        roll_out,
        np.arange(start, sig_end + 1),
        sig_start - start,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        linear_total_extension,
        sig_start - start + 1,
        sig_end + 1 - start,
        co2_fossil_extend,
        co2_total_extend,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        0, sig_end + 1 - start, end + 1 - start, co2_fossil_extend, co2_total_extend
    )
    return co2_fossil_extend, co2_total_extend


def _apply_cscs_storyline(  # noqa: PLR0913
    co2_fossil_extend: np.ndarray,
    co2_total_extend: np.ndarray,
    storyline: list,
    start: int,
    end: int,
    scenario_end=2100,
) -> list[np.ndarray, np.ndarray]:
    """
    Apply Constant-Sigmoid-Constant-Sigmoid storyline.

    This function applies a storyline with constant, then sigmoid to a target, constant again, then sigmoid to zero.

    Parameters
    ----------
    co2_fossil_extend : np.ndarray
        Array of fossil CO2 emissions to extend.
    co2_total_extend : np.ndarray
        Array of total CO2 emissions to extend.
    storyline : list
        List of storyline parameters: [type, stop_const, sig_targ, end_sig1, start_sig2, end_sig2, roll_in, roll_out].
    start : int
        Start year of the extension.
    end : int
        End year of the extension.
    scenario_end : int, optional
        End year of the scenario, default is 2100.

    Returns
    -------
    co2_fossil_extend : np.ndarray
        Extended fossil CO2 emissions.
    co2_total_extend : np.ndarray
        Extended total CO2 emissions.
    """
    stop_const = storyline[1]
    sig_targ = storyline[2]
    end_sig1 = storyline[3]
    start_sig2 = storyline[4]
    end_sig2 = storyline[5]
    roll_in = storyline[6]
    roll_out = storyline[7]
    co2_constant = co2_total_extend[scenario_end - start]
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        co2_constant,
        scenario_end + 1 - start,
        stop_const - start + 1,
        co2_fossil_extend,
        co2_total_extend,
    )
    linear_total_extension = make_linear_function_with_smooth_transition(
        co2_total_extend[: stop_const + 1 - start],
        sig_targ,
        roll_in,
        roll_out,
        np.arange(start, end_sig1 + 1),
        stop_const - start,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        linear_total_extension,
        stop_const + 1 - start,
        end_sig1 + 1 - start,
        co2_fossil_extend,
        co2_total_extend,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        sig_targ,
        end_sig1 + 1 - start,
        start_sig2 - start + 1,
        co2_fossil_extend,
        co2_total_extend,
    )
    linear_total_extension = make_linear_function_with_smooth_transition(
        co2_total_extend[: start_sig2 - start + 1],
        0,
        roll_in,
        roll_out,
        np.arange(start, end_sig2 + 1),
        start_sig2 - start,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        linear_total_extension,
        start_sig2 - start + 1,
        end_sig2 + 1 - start,
        co2_fossil_extend,
        co2_total_extend,
    )
    co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        0,
        end_sig2 + 1 - start,
        end + 1 - start,
        co2_fossil_extend,
        co2_total_extend,
    )
    return co2_fossil_extend, co2_total_extend


def extend_co2_for_scen_storyline(  # noqa: PLR0913
    df_extended_afolu: pd.DataFrame,
    df_fossil: pd.DataFrame,
    storyline: list,
    start=2023,
    end=2500,
    scenario_end=2100,
) -> list[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extend CO2 emissions time series for a given scenario storyline up to a specified end year.

    This function combines fossil and AFOLU (Agriculture, Forestry, and Other Land Use) CO2 emissions,
    and applies scenario-specific extensions (constant, linear, or sigmoid transitions) based on the
    provided storyline parameters. The extension logic supports multiple storyline types, including
    constant-then-sigmoid (CS), exponential-then-sigmoid (ECS), and double constant-sigmoid (CSCS).

    Parameters
    ----------
    df_extended_afolu : pandas.DataFrame
        DataFrame containing extended AFOLU CO2 emissions, indexed or columned by year.
    df_fossil : pandas.DataFrame
        DataFrame containing fossil CO2 emissions, indexed or columned by year.
    storyline : tuple
        Scenario storyline parameters. The first element specifies the storyline type ("CS", "ECS", "CSCS"),
        followed by integers specifying transition years and target values for the extension.
    start : int, optional
        Start year for the extension (default is 2023).
    end : int, optional
        End year for the extension (default is 2500).

    Returns
    -------
    co2_total_extend : numpy.ndarray
        Extended total CO2 emissions (fossil + AFOLU) for each year in the range [start, end].
    co2_fossil_extend : numpy.ndarray
        Extended fossil CO2 emissions for each year in the range [start, end].
    extended_years : numpy.ndarray
        Array of years corresponding to the extended emissions time series.

    Raises
    ------
    SystemExit
        If the provided storyline type is not recognized.

    Notes
    -----
    The function relies on helper functions such as `extend_from_start_to_stop_with_value` and
    `make_linear_function_with_smooth_transition` for constructing the extended emissions profiles.
    """
    extended_years = np.arange(start, end + 1)
    co2_fossil_extend = np.zeros(len(extended_years))
    co2_fossil_extend[: scenario_end + 1 - start] = (
        df_fossil.loc[f"{start}" :, :].to_numpy().flatten()
    )
    co2_total_extend = (
        co2_fossil_extend + df_extended_afolu.loc[:, f"{start}" :].to_numpy().flatten()
    )

    storyline_type = storyline[0]
    if storyline_type == "CS":
        co2_fossil_extend, co2_total_extend = _apply_cs_storyline(
            co2_fossil_extend,
            co2_total_extend,
            storyline,
            start,
            end,
            scenario_end=scenario_end,
        )
    elif storyline_type == "ECS":
        co2_fossil_extend, co2_total_extend = _apply_ecs_storyline(
            co2_fossil_extend,
            co2_total_extend,
            storyline,
            start,
            end,
            scenario_end=scenario_end,
        )
    elif storyline_type == "CSCS":
        co2_fossil_extend, co2_total_extend = _apply_cscs_storyline(
            co2_fossil_extend,
            co2_total_extend,
            storyline,
            start,
            end,
            scenario_end=scenario_end,
        )
    else:
        msg = "Unexpected state (`sys.exit(4)` in the original)"
        raise AssertionError(msg)

    return co2_total_extend, co2_fossil_extend, extended_years
