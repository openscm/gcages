"""
General tools for running MAGICC
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gcages.exceptions import MissingOptionalDependencyError


def load_magicc_probabilistic_config(config_file: Path) -> list[dict[str, Any]]:
    """
    Load MAGICC configuration from a probabilistic config file

    Parameters
    ----------
    config_file
        Config file to load

    Returns
    -------
    :
        MAGICC configurations to use when running MAGICC
    """
    with open(config_file) as fh:
        cfgs_raw = json.load(fh)

    cfgs = [
        {
            "run_id": c["paraset_id"],
            **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
        }
        for c in cfgs_raw["configurations"]
    ]

    return cfgs


def combine_probabilistic_and_common_cfg(
    probabilistic_cfgs: list[dict[str, Any]], common_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Combine probabilistic and common configs to make a full set of run configs

    Parameters
    ----------
    probabilistic_cfgs
        Probabilistic configs, loaded from e.g. [load_magicc_probabilistic_config][(m).]

    common_cfg
        Common configuration to apply to all runs

    Returns
    -------
    :
        Run configuration i.e. the combination of `probabilistic_cfgs` and `common_cfg`
    """
    run_config = [{**common_cfg, **prob_cfg} for prob_cfg in probabilistic_cfgs]

    return run_config


def check_magicc7_version(magicc_exe_path: Path, expected_version: str) -> None:
    """
    Check that the MAGICC7 version is what we expect

    Parameters
    ----------
    magicc_exe_path
        Path to the MAGICC executable to use

    expected_version
        Expected version

    Raises
    ------
    AssertionError
        The MAGICC version is not what we expect

    MissingOptionalDependencyError
        [openscm-runner](https://github.com/openscm/openscm-runner) is not installed
    """
    try:
        import openscm_runner.adapters
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "check_cmip7_scenariomip_magicc7_version", requirement="openscm_runner"
        ) from exc

    with temporary_env_var("MAGICC_EXECUTABLE_7", str(magicc_exe_path)):
        magicc_version = openscm_runner.adapters.MAGICC7.get_version()
        if magicc_version != expected_version:  # type: ignore
            raise AssertionError(openscm_runner.adapters.MAGICC7.get_version())  # type: ignore


@contextlib.contextmanager
def temporary_env_var(env_var: str, value: str) -> Iterator[None]:
    """
    Set a temporary value for an environment variable

    Parameters
    ----------
    env_var
        Environment variable to set

    value
        Value to set
    """
    current_value = os.environ.get(env_var, None)
    os.environ[env_var] = value
    try:
        yield
    finally:
        if current_value is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = current_value
