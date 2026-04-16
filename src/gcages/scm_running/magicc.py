"""
General tools for running MAGICC
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
