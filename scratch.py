# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Running SCI scenarios through CMIP7 ScenarioMIP global-like workflow.

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from __future__ import annotations

import multiprocessing
import os
import platform
from pathlib import Path

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.cmip7_scenariomip.harmonisation import (
    create_cmip7_scenariomip_global_harmoniser,
    load_cmip7_scenariomip_historical_emissions,
)
from gcages.cmip7_scenariomip.infilling import (
    CMIP7ScenarioMIPInfiller,
    load_cmip7_scenariomip_ghg_inversions,
    load_cmip7_scenariomip_historical_emissions,
    load_cmip7_scenariomip_infilling_db,
)
from gcages.cmip7_scenariomip.post_processing import CMIP7ScenarioMIPPostProcessor
from gcages.cmip7_scenariomip.scm_running import CMIP7ScenarioMIPSCMRunner
from gcages.harmonisation.common import assert_harmonised
from gcages.renaming import (
    SupportedNamingConventions,
    convert_variable_name,
    rename_variables,
)
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

# %%
# Setup pint
pint.set_application_registry(openscm_units.unit_registry)

# %%
pandas_openscm.register_pandas_accessors()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Starting point
#
# The starting point is an emissions scenario submission.
# This must be in a format like the below,
# i.e. a pandas `DataFrame` with a `MultiIndex` with levels:
# `["model", "scenario", "region", "variable", "unit"]`
# and years as the columns.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Naming conventions and reporting
#
# The naming convention and reporting rules
# are specific to CMIP7's ScenarioMIP.
# These are quite complex and not a solid target.
# As a result, there aren't super clear docs.
# Our best advice is to use an existing submission as an example,
# or simply try using one of the different reaggregators (see below)
# and let its error messages guide you through what else is needed.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Here we use the 'basic' naming convention.
# An example of the data required when using this convention
# is provided below
# (it is based on some example data we use for testing,
# because it requires reporting thousands of timeseries
# so writing it by hand would be hard work).
# The example data does not contain all possible timeseries
# that can be reported, rather just a selection useful for this example
# (again, a discussion of the complete reporting rules is out of scope here,
# docs specifically on this to come we hope).

# %% editable=true slideshow={"slide_type": ""}
EXAMPLE_INPUT_FILE = Path(
    "tests/regression/cmip7-scenariomip/test-data/salted-202507-scenariomip-input.csv"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
# Some trickery to make sure we pick up files in the right path,
# even when building the docs :)
if not EXAMPLE_INPUT_FILE.exists():
    EXAMPLE_INPUT_FILE = Path("../..") / EXAMPLE_INPUT_FILE
    if not EXAMPLE_INPUT_FILE.exists():
        raise AssertionError

# %% editable=true slideshow={"slide_type": ""}

#
# start = pd.read_excel("SCI-2025_v1.0_pathways_ensemble_global.xlsx", sheet_name="data")
# tmp = start.copy()
# tmp.columns = tmp.columns.str.lower()
# tmp = tmp.set_index(["model", "scenario", "region", "variable", "unit"])
# emissions = tmp.loc[pix.ismatch(variable="Emissions**", region="World")]
# emissions.to_feather("emissions.feather")

emissions = pd.read_feather("emissions.feather")
emissions.columns = emissions.columns.astype(int)


# %% editable=true slideshow={"slide_type": ""}
CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE = Path(
    "tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs/history_cmip7_scenariomip.csv"
)
ANERIS_GLOBAL_OVERRIDES_FILE = Path(
    "tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs/aneris-overrides-global.csv"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
# Some trickery to make sure we pick up files in the right path,
# even when building the docs :)
if not CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE.exists():
    CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE = (
        Path("../..") / CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE
    )
    if not CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE.exists():
        raise AssertionError

if not ANERIS_GLOBAL_OVERRIDES_FILE.exists():
    ANERIS_GLOBAL_OVERRIDES_FILE = Path("../..") / ANERIS_GLOBAL_OVERRIDES_FILE
    if not ANERIS_GLOBAL_OVERRIDES_FILE.exists():
        raise AssertionError

# %% editable=true slideshow={"slide_type": ""}
harmoniser_global = create_cmip7_scenariomip_global_harmoniser(
    cmip7_scenariomip_global_historical_emissions_file=CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE,
    aneris_global_overrides_file=ANERIS_GLOBAL_OVERRIDES_FILE,
    n_processes=None,  # run serially for this demo
)

# %% editable=true slideshow={"slide_type": ""}

for y in range(2010, 2100 + 1):
    if y not in emissions:
        emissions[y] = np.nan

emissions = emissions.sort_index(axis="columns")

unusable_too_late_start = (
    emissions.isnull()
    .idxmin(axis="columns")
    .groupby(["model", "scenario"])
    .max()
    .sort_values()
)
unusable_too_late_start = unusable_too_late_start[unusable_too_late_start > 2023]
emissions_usable = emissions.loc[
    ~pandas_openscm.indexing.multi_index_match(
        emissions.index,
        unusable_too_late_start.index,
    )
]
emissions = emissions_usable.T.interpolate(method="index").T

history_variables_renamed = [
    convert_variable_name(
        v,
        to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        from_convention=SupportedNamingConventions.GCAGES,
    )
    for v in harmoniser_global.historical_emissions.index.unique("variable")
]

emissions_in_history = emissions.loc[pix.isin(variable=history_variables_renamed)]
emissions_in_history = rename_variables(
    emissions_in_history,
    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    to_convention=SupportedNamingConventions.GCAGES,
)
emissions_in_history = strip_pint_incompatible_characters_from_units(
    emissions_in_history, units_index_level="unit"
)

run_all = True
if run_all:
    emissions_run = emissions_in_history
else:
    models_included = []
    new_idx_levels = []
    for model, scenario in emissions_in_history.index.droplevel(
        ["region", "variable", "unit"]
    ).unique():
        if model in models_included:
            continue
    
        models_included.append(model)
        new_idx_levels.append((model, scenario))
    
    emissions_run = emissions_in_history.loc[
        pandas_openscm.indexing.multi_index_match(
            emissions_in_history.index,
            pd.MultiIndex.from_tuples(new_idx_levels, names=["model", "scenario"]),
        )
    ]

harmonised_global = harmoniser_global(emissions_run)
harmonised_global


# %%
BASE_DIR = Path("tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs")
CMIP7_SCENARIOMIP_INFILLING_FILE = BASE_DIR / "infilling_db_cmip7_scenariomip.csv"
# Need to do this better
# (check into chapter repo)
CMIP7_SCENARIOMIP_INFILLING_FILE = Path("infilling_db_cmip7_scenariomip_20566343.csv")
GHG_INVERSION_FILE = BASE_DIR / "cmip7_ghg_inversions.csv"

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
if not CMIP7_SCENARIOMIP_INFILLING_FILE.exists():
    CMIP7_SCENARIOMIP_INFILLING_FILE = Path("../..") / CMIP7_SCENARIOMIP_INFILLING_FILE
    if not CMIP7_SCENARIOMIP_INFILLING_FILE.exists():
        raise AssertionError

if not GHG_INVERSION_FILE.exists():
    GHG_INVERSION_FILE = Path("../..") / GHG_INVERSION_FILE
    if not GHG_INVERSION_FILE.exists():
        raise AssertionError

# %% [markdown]
# With the infilling databases, we can initialise our infiller.

# %%
# Change this to infill everything except CO2 fossil?

# Hardcode as we are matching CMIP7 ScenarioMIP exactly.
# Users can copy and modify themselves if they wish
# (or we can introduce a lower layer if lots of users want it)
PI_YEAR = 1750
HARMONISATION_YEAR = 2023

ur = openscm_units.unit_registry

# Still embargoed
# Have to use https://zenodo.org/records/20566343
# so that all emissions are there
infilling_db = load_cmip7_scenariomip_infilling_db(
    filepath=CMIP7_SCENARIOMIP_INFILLING_FILE,
    check_hash=False,  # TODO: update when available
)

# CMIP7 GHG inversions
cmip7_ghg_inversions = load_cmip7_scenariomip_ghg_inversions(
    filepath=GHG_INVERSION_FILE,
)
# History
historical_emissions = load_cmip7_scenariomip_historical_emissions(
    filepath=CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE,
    check_hash=True,
)

# Use gcages naming convention.
infilling_db = update_index_levels_func(
    infilling_db,
    {
        "variable": lambda x: convert_variable_name(
            x,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        )
    },
    copy=False,
)
cmip7_ghg_inversions = update_index_levels_func(
    cmip7_ghg_inversions,
    {
        "variable": lambda x: convert_variable_name(
            x,
            from_convention=SupportedNamingConventions.OPENSCM_RUNNER,
            to_convention=SupportedNamingConventions.GCAGES,
        )
    },
    copy=False,
)
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

assert_harmonised(
    infilling_db,
    history=historical_emissions.reset_index(
        level=[
            lvl
            for lvl in ["model", "scenario"]
            if lvl in historical_emissions.index.names
        ],
        drop=True,
    ),
    harmonisation_time=HARMONISATION_YEAR,
    history_unit_level="unit",
    ur=ur,
)

# Notes: currently this uses RMSClosest under the hood.
# That's probably not a bad decision, and avoids the OC-BC decoupling from AR6.
# We should check though and make an active, rather than passive decision.
# We likely also want to use an updated infilling DB rather than the ScenarioMIP one,
# which was just whatever scenarios we had at the time.
infiller = CMIP7ScenarioMIPInfiller(
    infilling_db=infilling_db,
    historical_emissions=historical_emissions,
    cmip7_ghg_inversions=cmip7_ghg_inversions,
    harmonisation_year=HARMONISATION_YEAR,
    pre_industrial_year=PI_YEAR,
    run_checks=True,
    ur=ur,
)
# Not usable, we have to add in the extra infilling
# so it's not the same as CMIP7 ScenarioMIP.
# infiller = CMIP7ScenarioMIPInfiller.from_cmip7_scenariomip_config(
#     cmip7_scenariomip_infilling_leader_emissions_file=CMIP7_SCENARIOMIP_INFILLING_FILE,  # noqa: E501
#     cmip7_ghg_inversions_file=GHG_INVERSION_FILE,
#     cmip7_scenariomip_global_historical_emissions_file=CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE,  # noqa: E501
# )

# %% [markdown]
# And infill

# %% editable=true slideshow={"slide_type": ""}
# Be careful, this will change when we fix up the infiller.
complete = infiller(harmonised_global)
complete.to_feather("complete.feather")
complete

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## SCM Running
#
# The next step is running a simple climate model (SCM) or models (SCMs).
# This provides information about the climate implications of the scenario(s).
#
# In CMIP7 ScenarioMIP, MAGICCv7.6.0 and FaIR [version TBD]
# were used to inform this climate assessment.
#
# Under the hood, the AR6 SCM running uses the
# [OpenSCM-Runner](https://github.com/openscm/openscm-runner) package.

# %%
# Get MAGICC from somewhere public
# FaIR should be pre-installed

# %%
# combine to create complete timeseries using sum of gridded
# and the global harmonised timeseries or provide both as input
# complete_scenarios = pd.concat([harmonised, infilled])
# complete_scenarios

# %%
MAGICC_EXE_PATH = Path(
    "tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs/magicc-v7.6.0a3/bin"
)
MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE = Path(
    "tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs/magicc-v7.6.0a3/configs/magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
if not MAGICC_EXE_PATH.exists():
    MAGICC_EXE_PATH = Path("../..") / MAGICC_EXE_PATH
    if not MAGICC_EXE_PATH.exists():
        raise AssertionError

if not MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE.exists():
    MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE = (
        Path("../..")
        / "tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs"
        / "magicc-v7.6.0a3/configs"
        / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
    )
    if not MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE.exists():
        raise AssertionError

# %%
if platform.system() == "Darwin":
    if platform.processor() == "arm":
        MAGICC_EXE = MAGICC_EXE_PATH / "magicc-darwin-arm64"
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/gfortran/lib/gcc/current/"

elif platform.system() == "Linux":
    MAGICC_EXE = MAGICC_EXE_PATH / "magicc"

elif platform.system() == "Windows":
    MAGICC_EXE = MAGICC_EXE_PATH / "magicc.exe"

# %% [markdown]
# With the set up done, we can initialise our SCM runner.

# %%
# mkdir scm-output-db

# %%
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

scm_output_db = OpenSCMDB(
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
    db_dir=Path("scm-output-db"),
)

# %%
scm_runner = CMIP7ScenarioMIPSCMRunner.from_cmip7_scenariomip_config(
    # Generally, you want to run SCMs in parallel
    n_processes=multiprocessing.cpu_count(),
    magicc_exe_path=MAGICC_EXE,
    magicc_prob_distribution_path=MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE,
    historical_emissions_path=CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE,
    output_variables=("Surface Air Temperature Change", "Effective Radiative Forcing"),
    batch_size_scenarios=15,
    db=scm_output_db,
)

# %% [markdown]
# If you're reading this on RtD,
# note that we run a greatly reduced number of ensemble members.
# You will likely want to skip this step if running yourself.

# %% editable=true slideshow={"slide_type": ""}
# if os.environ.get("READTHEDOCS", False):
n_cfgs = 600
scm_runner.climate_models_cfgs["MAGICC7"] = scm_runner.climate_models_cfgs["MAGICC7"][
    :n_cfgs
]

# %% [markdown] editable=true slideshow={"slide_type": ""}
# And then run

# %% editable=true slideshow={"slide_type": ""}
scm_results = scm_runner(complete)
scm_results

# %% [markdown] editable=true slideshow={"slide_type": ""}
# With these outputs, we can look at raw (i.e. before pre-processing) variables.

# %%
scm_results.loc[
    pix.isin(variable=["Effective Radiative Forcing"], model="TIAM-ECN 1.1")
].openscm.plot_plume_after_calculating_quantiles(
    style_var="variable",
    quantile_over="run_id",
    quantiles_plumes=(
        (0.5, 0.8),
        ((0.05, 0.95), 0.3),
    ),
)

# %% [markdown]
# ## Post-processing
#
# The last step is post-processing.
# This handles calculation of key pieces of metadata.

# %%
post_processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()
post_processed_results = post_processor(
    scm_results.loc[pix.isin(variable=["Surface Air Temperature Change"])]
)

# %% [markdown]
# For example, the scenario category.

# %%
post_processed_results.metadata_categories.unstack("metric")

# %% [markdown]
# Exceedance thresholds.

# %%
post_processed_results.metadata_exceedance_probabilities.unstack("threshold")

# %% [markdown]
# Key warming metrics.

# %%
post_processed_results.metadata_quantile.loc[
    pix.isin(quantile=[0.05, 0.5, 0.95])
].unstack(["quantile", "metric"]).round(2).sort_index(axis="columns")

# %% [markdown]
# Assessed surface temperatures.

# %% editable=true slideshow={"slide_type": ""}
post_processed_results.timeseries_quantile.loc[pix.isin(model="TIAM-ECN 1.1"), 2000:].openscm.plot_plume(
    style_var="variable",
    quantiles_plumes=(
        (0.5, 0.8),
        ((0.05, 0.95), 0.3),
    ),
)

# %%
