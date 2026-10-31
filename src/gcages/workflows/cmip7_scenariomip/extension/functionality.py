"""
Low-level helpers for constructing extension profiles (sigmoids, decays etc.)

Ported from https://github.com/iiasa/emissions_harmonization_historical
(commit 376987dbe3b66fcd23537519a16771e21cba6d1e,
`src/emissions_harmonization_historical/extensions/extension_functionality.py`),
which is based on FLEX (https://github.com/benmsanderson/FLEX, MIT licence).
The logic is unchanged, so that we reproduce the CMIP7 ScenarioMIP extensions exactly.
"""

from __future__ import annotations

import numpy as np

from gcages.exceptions import MissingOptionalDependencyError


def sigmoid_function(  # noqa: PLR0913
    to_val: float,
    from_val: float,
    tstart: float,
    tend: float,
    z: np.ndarray,
    adjust_from=True,
) -> np.ndarray:
    """
    Calculate sigmoid function

    Calculate sigmoid function from from_val to to_val
    between tstart and tend for over t-values z

    Parameters
    ----------
    to_val : float
        Target value.
    from_val : float
        Starting value.
    tstart : float
        Start time.
    tend : float
        End time.
    z : np.ndarray
        Time values.
    adjust_from : bool, optional
        Whether to adjust from_val, default True.

    Returns
    -------
    np.ndarray
        Sigmoid values.
    """
    z_shift = (z - (tstart + tend) / 2.0) * 5 / 2.0 / (tend - tstart)
    if not adjust_from:
        from_inf = from_val
    else:
        from_inf = (from_val - to_val / (1.0 + np.exp(-z_shift[0]))) / (
            1 - 1 / (1.0 + np.exp(-z_shift[0]))
        )
    scaling = to_val - from_inf
    sigmoid = (1.0 / (1.0 + np.exp(-z_shift))) * scaling + from_inf

    return sigmoid


def get_sigmoid_derivative(
    to_val: float, from_val: float, sigmoid_shift: float, sigmoid_len=50
) -> float:
    """
    Calculate the derivative of the sigmoid function

    Parameters
    ----------
    to_val : float
        Target value.
    from_val : float
        Starting value.
    sigmoid_shift : float
        Shift parameter.
    sigmoid_len : int, optional
        Length parameter, default 50.

    Returns
    -------
    float
        Derivative value.
    """
    scaling = to_val - from_val
    t_eval = sigmoid_shift * 5 / 2 / sigmoid_len
    derivative = (
        scaling * (np.exp(-t_eval) / (1 + np.exp(t_eval))) * 5 / 2 / sigmoid_len
    )
    return derivative


def exp_decay(
    end: float, scale: float, decay_time: float, time_0: float, time: np.ndarray
) -> np.ndarray:
    """
    Calculate exponential decay

    Calculate exponential decay to end value, with scale, decay_time,
    time_0 and over the values of time

    Parameters
    ----------
    end : float
        End value.
    scale : float
        Scale parameter.
    decay_time : float
        Decay time.
    time_0 : float
        Initial time.
    time : np.ndarray
        Time array.

    Returns
    -------
    np.ndarray
        Decayed values.
    """
    return end - scale * np.exp((time_0 - time) / decay_time)


def solve_LU_constants(
    cle_0: float, lu_0: float, dlu_0: float, cle_inf_0=False, min_tau=20
) -> tuple[float, float, float]:
    """
    Solve for Land use exponential decay values

    Parameters
    ----------
    cle_0 : float
        Initial CLE value.
    lu_0 : float
        Initial LU value.
    dlu_0 : float
        Initial DLU derivative.
    cle_inf_0 : bool, optional
        Whether CLE infinity is 0, default False.
    min_tau : int, optional
        Minimum tau, default 20.

    Returns
    -------
    tuple[float, float, float]
        CLE infinity, k multiplier, tau.
    """
    if cle_inf_0:
        cle_inf = 0
        k_mult = -cle_0
    else:
        k_mult = -lu_0 / dlu_0
        cle_inf = cle_0 + k_mult
    tau = k_mult / lu_0

    if tau < min_tau:
        tau = min_tau
    elif tau > min_tau * 5:
        tau = min_tau * 5
    k_mult = tau * lu_0
    cle_inf = cle_0 + k_mult
    return cle_inf, k_mult, tau


def find_func_form_lu_extension(
    function: np.ndarray,
    cle_func: np.ndarray,
    t_vals: np.ndarray,
    t_extend: int,
    cle_inf_0=False,
) -> tuple[np.ndarray, float]:
    """
    Calculate extended land use

    Parameters
    ----------
    function : np.ndarray
        Function array.
    cle_func : np.ndarray
        CLE function array.
    t_vals : np.ndarray
        Time values.
    t_extend : int
        Extension point.
    cle_inf_0 : bool, optional
        Whether CLE infinity is 0, default False.

    Returns
    -------
    tuple[np.ndarray, float]
        Extended function and CLE infinity.
    """
    lu_0 = function[t_extend]
    cle_0 = cle_func[t_extend]
    if cle_inf_0:
        cle_inf, k_mult, tau = solve_LU_constants(cle_0, lu_0, 1, cle_inf_0=cle_inf_0)
    else:
        dlu_0 = get_derivative_using_spline(function, t_vals, t_extend)
        cle_inf, k_mult, tau = solve_LU_constants(cle_0, lu_0, dlu_0)
    ext_func = np.zeros(len(t_vals))
    ext_func[:t_extend] = function[:t_extend]
    ext_func[t_extend:] = (
        k_mult / tau * np.exp((t_vals[t_extend] - t_vals[t_extend:]) / tau)
    )
    return ext_func, cle_inf


def extend_flat_evolution(function: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    """
    Extend function flatly

    Parameters
    ----------
    function : np.ndarray
        Function to extend.
    t_vals : np.ndarray
        Time values.

    Returns
    -------
    np.ndarray
        Extended function.
    """
    data_extend = np.zeros_like(len(t_vals))
    data_extend[: len(function)] = function
    data_extend[len(function) :] = function[-1]
    return data_extend


def extend_flat_cumulative(function: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    """
    Extend cumulative function so cumulative stays the same

    Parameters
    ----------
    function : np.ndarray
        Cumulative function to extend.
    t_vals : np.ndarray
        Time values.

    Returns
    -------
    np.ndarray
        Extended cumulative function.
    """
    data_extend = np.zeros_like(len(t_vals))
    data_extend[: len(function)] = function
    return data_extend


def extend_linear_rampdown(
    function: np.ndarray, t_vals: np.ndarray, rampdown_end=2150
) -> np.ndarray:
    """
    Extend cumulative function so cumulative stays the same

    Parameters
    ----------
    function : np.ndarray
        Function to extend.
    t_vals : np.ndarray
        Time values.
    rampdown_end : int, optional
        End of rampdown, default 2150.

    Returns
    -------
    np.ndarray
        Extended function.
    """
    data_extend = np.zeros(len(t_vals))
    data_extend[: len(function)] = function
    data_extend[len(function) : rampdown_end - t_vals[0]] = np.linspace(
        function[-1], 0, rampdown_end - t_vals[len(function)]
    )
    return data_extend


def make_cubic_coefficients_from_end_points_and_derivatives(  # noqa: PLR0913
    x0: float, y0: float, dy0: float, x1: float, y1: float, dy1: float
) -> tuple[float, float, float, float]:
    """
    Make cubic coefficients from end points and derivatives

    Parameters
    ----------
    x0 : float
        Start x.
    y0 : float
        Start y.
    dy0 : float
        Start derivative.
    x1 : float
        End x.
    y1 : float
        End y.
    dy1 : float
        End derivative.

    Returns
    -------
    tuple[float, float, float, float]
        Coefficients A, B, C, D.
    """
    dt = x1 - x0
    A = (dy1 + dy0 - 2 * (y1 - y0) / dt) / (dt**2)
    B = (3 * (y1 - y0) / dt - 2 * dy0 - dy1) / dt
    C = dy0
    D = y0
    return A, B, C, D


def make_combined_quadratic(  # noqa: PLR0913
    x0: float, y0: float, dy0: float, x1: float, y1: float, dy1: float
) -> tuple[float, float, float]:
    """
    Make combined quadratic coefficients from end points and derivatives

    Parameters
    ----------
    x0 : float
        Start x.
    y0 : float
        Start y.
    dy0 : float
        Start derivative.
    x1 : float
        End x.
    y1 : float
        End y.
    dy1 : float
        End derivative.

    Returns
    -------
    tuple[float, float, float]
        Coefficients A, B, C.
    """
    dt = x1 - x0
    A = ((y1 - y0) / dt + (dy1 - dy0) / 2) / (dt)
    B = (y1 - y0) / dt + (dy0 - dy1) / 2
    C = y0
    return A, B, C


def smooth_step(t_vals: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    """
    Make smooth step between two functions

    Parameters
    ----------
    t_vals : np.ndarray
        Time values.
    t_start : float
        Start time.
    t_end : float
        End time.

    Returns
    -------
    np.ndarray
        Smooth step values.
    """
    S = np.zeros(len(t_vals))
    for i, t_val in enumerate(t_vals):
        if t_val <= t_start:
            S[i] = 0
        elif t_val >= t_end:
            S[i] = 1
        else:
            x = (t_val - t_start) / (t_end - t_start)
            S[i] = x**3 * (x * (x * 6 - 15) + 10)
            S[i] = 0.5 * (1 + np.tanh(x))
            # S[i] = 3 * x**2 - 2 * x**3
            # S[i] = x
    return S


def make_linear_function_with_smooth_transition(  # noqa: PLR0913
    function: np.ndarray,
    target_val: float,
    roll_start_length: int,
    roll_end_length: int,
    t_vals: np.ndarray,
    t_extend: int,
) -> np.ndarray:
    """
    Make linear function with smooth transition to target

    Parameters
    ----------
    function : np.ndarray
        Function to extend.
    target_val : float
        Target value.
    roll_start_length : int
        Roll start length.
    roll_end_length : int
        Roll end length.
    t_vals : np.ndarray
        Time values.
    t_extend : int
        Extension point.

    Returns
    -------
    np.ndarray
        Extended function.
    """
    try:
        from scipy.interpolate import PchipInterpolator  # noqa: PLC0415
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "make_linear_function_with_smooth_transition", requirement="scipy"
        ) from exc

    data_extend = np.zeros(len(t_vals))
    data_extend[: len(function)] = function
    slope_at_extension = get_derivative_using_spline(function, t_vals, t_extend)
    vert_dist = target_val - function[-1]
    vert_dist_high = target_val - function[-1] - slope_at_extension * roll_start_length

    slope_max = vert_dist_high / (t_vals[-1] - t_vals[t_extend] - roll_end_length * 0.3)
    slope_min = (vert_dist) / (t_vals[-1] - t_vals[t_extend])
    linear_slope = np.mean([slope_max, slope_min])
    # height_at_start_of_linear = target_val - linear_slope *lin_seg_length
    height_at_start_of_linear = target_val - linear_slope * (
        t_vals[-1] - t_vals[t_extend] - roll_end_length * 0.7
    )
    indices_roll_1 = np.arange(t_extend + 1, t_extend + 1 + roll_start_length)
    indices_roll_2 = np.arange(len(t_vals) - roll_end_length, len(t_vals))
    indices_linear = np.arange(
        t_extend + 1 + roll_start_length, len(t_vals) - roll_end_length
    )
    # data_extend[indices_linear] = function[-1] + linear_slope* (t_vals[indices_linear] - t_vals[t_extend])
    data_extend[indices_linear] = height_at_start_of_linear + linear_slope * (
        t_vals[indices_linear] - t_vals[t_extend]
    )

    interpolator1 = PchipInterpolator(
        np.concatenate((t_vals[: t_extend + 1], t_vals[indices_linear])),
        np.concatenate((function[: t_extend + 1], data_extend[indices_linear])),
    )
    data_extend[indices_roll_1] = interpolator1(t_vals[indices_roll_1])
    interpolator2 = PchipInterpolator(
        np.concatenate(
            (t_vals[indices_linear], np.arange(t_vals[-1], t_vals[-1] + 100))
        ),
        np.concatenate((data_extend[indices_linear], np.ones(100) * target_val)),
    )
    data_extend[indices_roll_2] = interpolator2(t_vals[indices_roll_2])
    return data_extend[t_extend + 1 :]


def make_cubic_roll_to_linear_extension(  # noqa: PLR0913
    function: np.ndarray,
    target_val: float,
    roll_start_length: int,
    roll_end_length: int,
    t_vals: np.ndarray,
    t_extend: int,
    vert_dist_end_frac=0.01,
) -> np.ndarray:
    """
    Make a cubic roll to linear extension

    Parameters
    ----------
    function : np.ndarray
        Function to extend.
    target_val : float
        Target value.
    roll_start_length : int
        Roll start length.
    roll_end_length : int
        Roll end length.
    t_vals : np.ndarray
        Time values.
    t_extend : int
        Extension point.
    vert_dist_end_frac : float, optional
        Vertical distance end fraction, default 0.01.

    Returns
    -------
    np.ndarray
        Extended function.
    """
    data_extend = np.zeros(len(t_vals))
    data_extend[: len(function)] = function
    slope_at_extension = get_derivative_using_spline(function, t_vals, t_extend)
    lin_seg_length = len(t_vals) - len(function) - roll_start_length - roll_end_length
    # Calculate vertical distance to cover in final cubic
    vert_dist = (target_val - function[-1]) * (1 - vert_dist_end_frac)

    slope_max = vert_dist / lin_seg_length
    slope_min = vert_dist / (lin_seg_length + roll_start_length)
    slope = np.mean([slope_max, slope_min])
    height_at_start_of_linear = target_val - slope * lin_seg_length

    a1, b1, c1, d1 = make_cubic_coefficients_from_end_points_and_derivatives(
        t_vals[t_extend],
        function[-1],
        slope_at_extension,
        t_vals[t_extend] + roll_start_length,
        height_at_start_of_linear,
        slope,
    )
    a2, b2, c2, d2 = make_cubic_coefficients_from_end_points_and_derivatives(
        t_vals[t_extend] + roll_start_length + lin_seg_length,
        target_val + vert_dist_end_frac,
        slope,
        t_vals[t_extend] + roll_start_length + lin_seg_length + roll_end_length,
        target_val,
        0,
    )
    # First cubic segment
    for i, t_val in enumerate(t_vals):
        if t_val <= t_vals[t_extend]:
            continue
        dt = t_val - t_vals[t_extend]
        if t_val <= t_vals[t_extend] + roll_start_length:
            data_extend[i] = a1 * dt**3 + b1 * dt**2 + c1 * dt + d1
        elif t_val <= t_vals[t_extend] + roll_start_length + lin_seg_length:
            data_extend[i] = height_at_start_of_linear + slope * (
                t_val - (t_vals[t_extend] + roll_start_length)
            )
        elif (
            t_val
            <= t_vals[t_extend] + roll_start_length + lin_seg_length + roll_end_length
        ):
            dt2 = dt - (roll_start_length + lin_seg_length)
            data_extend[i] = a2 * dt2**3 + b2 * dt2**2 + c2 * dt2 + d2
        else:
            data_extend[i] = target_val
    return data_extend[t_extend + 1 :]


def get_exp_targ_from_current_data(
    function: np.ndarray, exp_end: float, fraction_extend=0.1
) -> float:
    """
    Get exponential target from current data

    Parameters
    ----------
    function : np.ndarray
        Function array.
    exp_end : float
        Exponential end.
    fraction_extend : float, optional
        Fraction to extend, default 0.1.

    Returns
    -------
    float
        Exponential target.
    """
    slope_at_extension = get_derivative_using_spline(
        function, np.arange(len(function)), len(function) - 1
    )
    exp_targ = (
        function[-1]
        + slope_at_extension * (exp_end - (len(function) - 1)) * fraction_extend
    )

    return exp_targ


def make_quadratic_roll_to_linear_extension(  # noqa: PLR0913
    function: np.ndarray,
    target_val: float,
    roll_start_length: int,
    roll_end_length: int,
    t_vals: np.ndarray,
    t_extend: int,
    vert_dist_end_frac=0.01,
) -> np.ndarray:
    """
    Make a cubic roll to linear extension

    Parameters
    ----------
    function : np.ndarray
        Function to extend.
    target_val : float
        Target value.
    roll_start_length : int
        Roll start length.
    roll_end_length : int
        Roll end length.
    t_vals : np.ndarray
        Time values.
    t_extend : int
        Extension point.
    vert_dist_end_frac : float, optional
        Vertical distance end fraction, default 0.01.

    Returns
    -------
    np.ndarray
        Extended function.
    """
    data_extend = np.zeros(len(t_vals))
    data_extend[: len(function)] = function
    slope_at_extension = get_derivative_using_spline(function, t_vals, t_extend)
    lin_seg_length = len(t_vals) - len(function) - roll_start_length - roll_end_length
    # Calculate vertical distance to cover in final cubic
    vert_dist = (target_val - function[-1]) * (1 - vert_dist_end_frac)
    vert_dist_for_end_cubic = (target_val - function[-1]) * vert_dist_end_frac

    slope_max = vert_dist / lin_seg_length
    slope_min = vert_dist / (lin_seg_length + roll_start_length)
    slope = np.mean([slope_max, slope_min])
    height_at_start_of_linear = (
        target_val - slope * lin_seg_length - vert_dist_for_end_cubic
    )

    a1, b1, c1 = make_combined_quadratic(
        t_vals[t_extend],
        function[-1],
        slope_at_extension,
        t_vals[t_extend] + roll_start_length,
        height_at_start_of_linear,
        slope,
    )
    a2, b2, c2 = make_combined_quadratic(
        t_vals[t_extend] + roll_start_length + lin_seg_length,
        target_val + vert_dist_end_frac,
        slope,
        t_vals[t_extend] + roll_start_length + lin_seg_length + roll_end_length,
        target_val,
        0,
    )
    # First cubic segment
    for i, t_val in enumerate(t_vals):
        if t_val <= t_vals[t_extend]:
            continue
        dt = t_val - t_vals[t_extend]
        if t_val <= t_vals[t_extend] + roll_start_length:
            data_extend[i] = a1 * dt**2 + b1 * dt + c1
        elif t_val <= t_vals[t_extend] + roll_start_length + lin_seg_length:
            data_extend[i] = height_at_start_of_linear + slope * (
                t_val - (t_vals[t_extend] + roll_start_length)
            )
        elif (
            t_val
            <= t_vals[t_extend] + roll_start_length + lin_seg_length + roll_end_length
        ):
            dt2 = dt - (roll_start_length + lin_seg_length)
            data_extend[i] = a2 * dt2**2 + b2 * dt2 + c2
        else:
            data_extend[i] = target_val
    return data_extend[t_extend + 1 :]


def get_derivative_using_spline(
    function: np.ndarray, t_vals: np.ndarray, t_extend: int
) -> float:
    """
    Get derivative of function using spline

    Parameters
    ----------
    function : np.ndarray
        Function array.
    t_vals : np.ndarray
        Time values.
    t_extend : int
        Extension point.

    Returns
    -------
    float
        Derivative value.
    """
    spline = np.interp(
        t_vals[t_extend - 10 : t_extend + 10],
        t_vals[t_extend - 50 : t_extend],
        function[t_extend - 50 : t_extend],
    )
    return np.gradient(spline)[8]


def do_simple_sigmoid_or_exponential_extension_to_target(  # noqa: PLR0913
    function: np.ndarray,
    t_vals: np.ndarray,
    t_extend: int,
    target: float,
    sigmoid_shift=40,
    sigmoid_len=50,
) -> np.ndarray:
    """
    Calculate extension function by calling sigmoid functionality to extend

    Parameters
    ----------
    function : np.ndarray
        Function to extend.
    t_vals : np.ndarray
        Time values.
    t_extend : int
        Extension point.
    target : float
        Target value.
    sigmoid_shift : int, optional
        Sigmoid shift, default 40.
    sigmoid_len : int, optional
        Sigmoid length, default 50.

    Returns
    -------
    np.ndarray
        Extended function.
    """
    data_extend = np.zeros(len(t_vals))
    data_extend[: len(function)] = function
    derivative_at_extension = get_derivative_using_spline(function, t_vals, t_extend)
    sigmoid_derivative_at_extension = get_sigmoid_derivative(
        target, function[-1], sigmoid_shift
    )
    if (
        derivative_at_extension < 0
        and sigmoid_derivative_at_extension < 0
        and (np.abs(derivative_at_extension) > np.abs(sigmoid_derivative_at_extension))
    ):
        data_extend[len(function) :] = exp_decay(
            target,
            target - function[-1],
            np.min((sigmoid_len, (target - function[-1]) / derivative_at_extension)),
            time_0=t_vals[t_extend],
            time=t_vals[len(function) :],
        )
    else:
        data_extend[len(function) :] = sigmoid_function(
            target,
            function[-1],
            t_vals[t_extend] + sigmoid_shift,
            t_vals[t_extend + sigmoid_len] + sigmoid_shift,
            t_vals[len(function) :],
        )
    return data_extend
