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
# # How to run the CMIP7 ScenarioMIP workflow
#
# Here we demonstrate how to run the workflow
# that was used in CMIP7's ScenarioMIP.
# <!--
# If you want a simpler interface for doing this,
# please see the
# [climate-processor](https://github.com/iiasa/climate-processor) package,
# which provides a [facade](https://refactoring.guru/design-patterns/facade)
# around gcages.
# -->
#
# Note: this is not yet complete, we will add further steps in future.

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import multiprocessing
import os
import platform
from pathlib import Path

import openscm_units
import pandas_indexing as pix
import pandas_openscm
import pint

from gcages.cmip7_scenariomip.harmonisation import (
    create_cmip7_scenariomip_global_harmoniser,
)
from gcages.cmip7_scenariomip.infilling import CMIP7ScenarioMIPInfiller
from gcages.cmip7_scenariomip.post_processing import CMIP7ScenarioMIPPostProcessor
from gcages.cmip7_scenariomip.scm_running import CMIP7ScenarioMIPSCMRunner

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
import pandas as pd

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
import numpy as np

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

emissions_in_history = emissions.loc[
    pandas_openscm.indexing.multi_index_match(
        emissions.index, harmoniser_global.historical_emissions.index.droplevel("unit")
    )
]
assert False, "cut to limited set of scenarios before trying this"


harmonised_global = harmoniser_global(emissions_in_history)
harmonised_global


# %%
BASE_DIR = Path("tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs")
CMIP7_SCENARIOMIP_INFILLING_FILE = BASE_DIR / "infilling_db_cmip7_scenariomip.csv"
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
infiller = CMIP7ScenarioMIPInfiller.from_cmip7_scenariomip_config(
    cmip7_scenariomip_infilling_leader_emissions_file=CMIP7_SCENARIOMIP_INFILLING_FILE,
    cmip7_ghg_inversions_file=GHG_INVERSION_FILE,
    cmip7_scenariomip_global_historical_emissions_file=CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE,
)

# %% [markdown]
# And infill

# %% editable=true slideshow={"slide_type": ""}
complete = infiller(harmonised_global)
complete

# %% [markdown]
# You can see infilled pathways compared to raw pathways in the below.
#
# This is not a deep analysis.
# If you want to see the full details of how the infilling works,
# see [Lamboll et al., 2020](https://doi.org/10.5194/gmd-13-5259-2020).

# %% editable=true slideshow={"slide_type": ""}
pdf = (
    pix.concat(
        [
            res_pre_processed.global_workflow_emissions.loc[:, 2023:].pix.assign(
                stage="pre_processed"
            ),
            harmonised_global.pix.assign(stage="harmonised"),
            complete.pix.assign(stage="infilled"),
        ]
    )
    .loc[pix.ismatch(variable=["**CO2|Fossil", "**CH4", "**N2O", "**CO"])]
    .melt(ignore_index=False, var_name="year")
    .reset_index()
)

fg = relplot_in_emms(
    data=pdf,
    hue="scenario",
    style="stage",
    dashes={
        "pre_processed": (3, 3),
        "harmonised": "",
        "infilled": (1, 1),
    },
)

fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
fg.axes.flatten()[1].set_ylim(ymin=0.0)

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
scm_runner = CMIP7ScenarioMIPSCMRunner.from_cmip7_scenariomip_config(
    # Generally, you want to run SCMs in parallel
    n_processes=multiprocessing.cpu_count(),
    magicc_exe_path=MAGICC_EXE,
    magicc_prob_distribution_path=MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE,
    historical_emissions_path=CMIP7_SCENARIOMIP_GLOBAL_HISTORICAL_EMISSIONS_FILE,
    output_variables=("Surface Air Temperature Change", "Effective Radiative Forcing"),
    batch_size_scenarios=15,
)

# %% [markdown]
# If you're reading this on RtD,
# note that we run a greatly reduced number of ensemble members.
# You will likely want to skip this step if running yourself.

# %% editable=true slideshow={"slide_type": ""}
if os.environ.get("READTHEDOCS", False):
    scm_runner.climate_models_cfgs["MAGICC7"] = scm_runner.climate_models_cfgs[
        "MAGICC7"
    ][:10]

# %% [markdown] editable=true slideshow={"slide_type": ""}
# And then run

# %% editable=true slideshow={"slide_type": ""}
scm_results = scm_runner(complete)
scm_results

# %% [markdown] editable=true slideshow={"slide_type": ""}
# With these outputs, we can look at raw (i.e. before pre-processing) variables.

# %%
scm_results.loc[
    pix.isin(variable=["Effective Radiative Forcing"])
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
post_processed_results.timeseries_quantile.loc[:, 2000:].openscm.plot_plume(
    style_var="variable",
    quantiles_plumes=(
        (0.5, 0.8),
        ((0.05, 0.95), 0.3),
    ),
)
