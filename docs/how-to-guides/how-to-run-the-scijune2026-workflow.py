# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to run the SCIJune2026 workflow
#
# Here we demonstrate how to run the workflow that was used in SCIJune2026.
# This is intended to demonstrate how to use the package in a familiar context.
# If you want a simpler interface for doing this,
# please see the
# [climate-processor](https://github.com/iiasa/climate-processor) package,
# which is a [facade](https://refactoring.guru/design-patterns/facade)
# around gcages.

# %% [markdown]
# ## Imports

# %%
import multiprocessing
import os
import platform
from functools import partial
from pathlib import Path

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import seaborn as sns

from gcages.sci_june_2026 import (
    SCIJune2026PostProcessor,
    SCIJune2026PreProcessor,
    SCIJune2026SCMRunner,
    create_scijune2026_harmoniser,
    create_scijune2026_infiller,
)

# %%
# Setup pint
pint.set_application_registry(openscm_units.unit_registry)

# %%
pandas_openscm.register_pandas_accessors()

# %% [markdown]
# ## Starting point
#
# The starting point is some emissions scenario.
# This must be in a format like the below,
# i.e. a pandas `DataFrame` with a `MultiIndex` with levels:
# `["model", "scenario", "region", "variable", "unit"]`
# and years as the columns.

# %% [markdown]
# ### Naming conventions
#
# A complete discussion on naming conventions is provided in
# or our docs on
# [naming conventions](../../tutorials/understanding-the-naming-conventions).
# In short, we use the `gcages` naming conventions throughout.
# However, SCI used the IAMC naming convention as its starting point.
# Hence, we start from the IAMC naming convention here.
# However, the pre-processing step alters the naming convention to gcages names.

# %%
# All the code to generate these demo timeseries
# (this is obviously a scrappy demo,
# you would normally use output from somewhere else
# and be much more careful than this :)).
time = np.arange(2015, 2100 + 1)

co2_flatline = np.ones_like(time) * 38.5
co2_flatline[np.logical_and(time > 2040, time < 2075)] = 38.5 * (
    1 - 1 / (1 + np.exp(-(np.arange(2075 - 2041) - 15) / 3.0))
)
co2_flatline[time >= 2075] = 0.0
co2_flatline *= 1000.0

ch4_flatline = np.ones_like(time) * 410
ch4_flatline[np.logical_and(time > 2040, time < 2075)] = (
    (410.0 - 200.0) * (1 - 1 / (1 + np.exp(-(np.arange(2075 - 2041) - 15) / 3.0)))
) + 200.0
ch4_flatline[time >= 2075] = 200.0

co2_decline = np.ones_like(time) * 38.5
co2_decline[np.logical_and(time > 2030, time < 2050)] = 38.5 * (
    1 - 1.2 / (1 + np.exp(-(np.arange(2050 - 2031) - 10) / 3.0))
)
co2_decline[time >= 2050] = np.nan
co2_decline[time == 2100] = 0.0
co2_decline *= 1000.0

ch4_decline = np.ones_like(time) * 410
ch4_decline[np.logical_and(time > 2030, time < 2060)] = (
    (410.0 - 150.0) * (1 - 1 / (1 + np.exp(-(np.arange(2060 - 2031) - 10) / 3.0)))
) + 150.0
ch4_decline[time >= 2050] = 150.0

# BC and OC (aerosol) trajectories for two of the scenarios.
# Aerosol emissions broadly track combustion, so here they follow a shape
# similar to each scenario's fossil CO2 pathway (falling as fossil use falls).
bc_flatline = np.ones_like(time) * 6.5
bc_flatline[np.logical_and(time > 2040, time < 2075)] = 6.5 * (
    1 - 0.75 / (1 + np.exp(-(np.arange(2075 - 2041) - 15) / 3.0))
)
bc_flatline[time >= 2075] = 6.5 * 0.25

oc_flatline = np.ones_like(time) * 34.0
oc_flatline[np.logical_and(time > 2040, time < 2075)] = 34.0 * (
    1 - 0.75 / (1 + np.exp(-(np.arange(2075 - 2041) - 15) / 3.0))
)
oc_flatline[time >= 2075] = 34.0 * 0.25

bc_decline = np.ones_like(time) * 6.5
bc_decline[np.logical_and(time > 2030, time < 2060)] = (
    (6.5 - 1.5) * (1 - 1 / (1 + np.exp(-(np.arange(2060 - 2031) - 10) / 3.0)))
) + 1.5
bc_decline[time >= 2050] = 1.5

oc_decline = np.ones_like(time) * 34.0
oc_decline[np.logical_and(time > 2030, time < 2060)] = (
    (34.0 - 10.0) * (1 - 1 / (1 + np.exp(-(np.arange(2060 - 2031) - 10) / 3.0)))
) + 10.0
oc_decline[time >= 2050] = 10.0

# %%
# Put it altogether into a DataFrame
start = pd.DataFrame(
    np.vstack(
        [
            co2_flatline,
            ch4_flatline,
            bc_flatline,
            oc_flatline,
            co2_flatline,
            co2_decline,
            ch4_decline,
            bc_decline,
            oc_decline,
            co2_decline * 0.85,
            co2_decline * 0.13
            + np.arange(co2_decline.size) * 10.0
            + 500.0 * np.ones_like(co2_decline),
        ]
    ),
    columns=time,
    index=pd.MultiIndex.from_tuples(
        # Note the use of IAMC names here
        [
            (
                "demo",
                "flatline",
                "World",
                "Emissions|CO2|Energy and Industrial Processes",
                "Mt CO2/yr",
            ),
            ("demo", "flatline", "World", "Emissions|CH4", "Mt CH4/yr"),
            ("demo", "flatline", "World", "Emissions|BC", "Mt BC/yr"),
            ("demo", "flatline", "World", "Emissions|OC", "Mt OC/yr"),
            (
                "demo",
                "flatline-co2-only",
                "World",
                "Emissions|CO2|Energy and Industrial Processes",
                "Mt CO2/yr",
            ),
            (
                "demo",
                "decline",
                "World",
                "Emissions|CO2|Energy and Industrial Processes",
                "Mt CO2/yr",
            ),
            ("demo", "decline", "World", "Emissions|CH4", "Mt CH4/yr"),
            ("demo", "decline", "World", "Emissions|BC", "Mt BC/yr"),
            ("demo", "decline", "World", "Emissions|OC", "Mt OC/yr"),
            (
                "demo",
                "decline_co2_fossil_split",
                "World",
                "Emissions|CO2|Energy",
                "Mt CO2/yr",
            ),
            (
                "demo",
                "decline_co2_fossil_split",
                "World",
                "Emissions|CO2|Industrial Processes",
                "Mt CO2/yr",
            ),
        ],
        names=["model", "scenario", "region", "variable", "unit"],
    ),
)
start = start.T.interpolate("index").T
start

# %% editable=true slideshow={"slide_type": ""}
relplot_in_emms = partial(
    sns.relplot,
    kind="line",
    linewidth=2.0,
    alpha=0.7,
    facet_kws=dict(sharey=False),
    x="year",
    y="value",
    col="variable",
    col_wrap=2,
)
fg = relplot_in_emms(
    data=start.melt(ignore_index=False, var_name="year").reset_index(),
    hue="scenario",
)

fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# ## Pre-process
#
# If you want to run in exactly the same way as AR6,
# the first step is pre-processing the emissions scenario(s)
# (if you just want the same logic as AR6 for other steps,
# it is possible to skip this step and just go straight to harmonisation).
# This step does some variable aggregation that was needed in AR6
# and extracts only those variables that are understood by the workflow.

# %%
pre_processor = SCIJune2026PreProcessor.from_sci_june2026_config(
    n_processes=None,  # run serially for this demo
    progress=False,
    run_checks=True,
)

# %%
# These are the variables (using the IAMC naming convention)
pre_processor.emissions_out

# %% [markdown]
# Here we run the pre-processing and get the pre-processed data.
# Note a few things which were done:
#
# 1. renaming of variables
# 2. aggregation of the scenario which provided energy
#    and industrial emissions separately

# %%
pre_processed = pre_processor(start)
pre_processed

# %% editable=true slideshow={"slide_type": ""}
fg = relplot_in_emms(
    data=pre_processed.melt(ignore_index=False, var_name="year").reset_index(),
    hue="scenario",
)

fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# ## Harmonisation
#
# The next step is harmonisation.
# This is the process of aligning the scenarios with historical emissions estimates.
#
# In SCIJune2026, a specific set of historical emissions
# was used as in ScenarioMIP-CMIP7.
#
# Under the hood, the harmonisation uses the
# [aneris](https://github.com/iiasa/aneris) package.
#
# The overrides harmonisation rules follow the rules prescribed in AR6.

# %% editable=true slideshow={"slide_type": ""}
HISTORICAL_EMISSIONS_FILE = Path(
    "tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs/history_cmip7_scenariomip.csv"
)
ANERIS_OVERRIDES_FILE = Path(
    "tests/regression/sci_june_2026/sci_workflow_inputs/sci_overrides.csv"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
# Some trickery to make sure we pick up files in the right path,
# even when building the docs :)
if not HISTORICAL_EMISSIONS_FILE.exists():
    HISTORICAL_EMISSIONS_FILE = Path("../..") / HISTORICAL_EMISSIONS_FILE
    if not HISTORICAL_EMISSIONS_FILE.exists():
        raise AssertionError
if not ANERIS_OVERRIDES_FILE.exists():
    ANERIS_OVERRIDES_FILE = Path("../..") / ANERIS_OVERRIDES_FILE
    if not ANERIS_OVERRIDES_FILE.exists():
        raise AssertionError

# %% [markdown]
# With this file, we can initialise a harmoniser.
#
# `harmonisation_year` is a custom variable, by default is fixed to 2023.
#

# %% editable=true slideshow={"slide_type": ""}
harmoniser = create_scijune2026_harmoniser(
    historical_emissions_file=HISTORICAL_EMISSIONS_FILE,
    aneris_overrides_file=ANERIS_OVERRIDES_FILE,
)
harmonisation_year = harmoniser.harmonisation_year

# %% [markdown]
# And harmonise

# %% editable=true slideshow={"slide_type": ""}
harmonised = harmoniser(pre_processed)
harmonised

# %% [markdown] editable=true slideshow={"slide_type": ""}
# You can see the modification to the pathways
# as a result of the harmonisation in the plot below.
# In scenarios that have more emissions,
# the same idea is applied to all variables.

# %% editable=true slideshow={"slide_type": ""}
pdf = (
    pix.concat(
        [
            pre_processed.pix.assign(stage="pre_processed"),
            harmonised.pix.assign(stage="harmonised"),
            harmoniser.historical_emissions.pix.assign(
                stage="history", scenario="history", model="history"
            ).loc[pix.isin(variable=pre_processed.pix.unique("variable"))],
        ]
    )
    .melt(ignore_index=False, var_name="year")
    .reset_index()
)

fg = relplot_in_emms(
    data=pdf,
    hue="scenario",
    style="stage",
    dashes={
        "history": (1, 1),
        "pre_processed": (3, 3),
        "harmonised": "",
    },
)

fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# ## Infilling
#
# The next step is infilling.
# This is the process of inferring any emissions which are not included
# in the scenarios, but are needed for running climate models.
# In our example, this is things like N2O, black carbon, sulfates.
#
# In SCIJune2026, a specific infilling database was used.
# There is no official home for all of this,
# but a copy is stored in the path below.
#
# Under the hood, the AR6 infilling uses the
# [silicone](https://github.com/GranthamImperial/silicone) package.

# %% editable=true slideshow={"slide_type": ""}
INFILLING_DB_FILE = Path(
    "tests/regression/sci_june_2026/sci_workflow_inputs/infilling_db_sci.csv"
)
INFILLING_DB_CFCS_FILE = Path(
    "tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs/cmip7_ghg_inversions.csv"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
# Some trickery to make sure we pick up files in the right path,
# even when building the docs :)
if not INFILLING_DB_FILE.exists():
    INFILLING_DB_FILE = Path("../..") / INFILLING_DB_FILE
    if not INFILLING_DB_FILE.exists():
        raise AssertionError

if not INFILLING_DB_CFCS_FILE.exists():
    INFILLING_DB_CFCS_FILE = Path("../..") / INFILLING_DB_CFCS_FILE
    if not INFILLING_DB_CFCS_FILE.exists():
        raise AssertionError

# %% [markdown]
# With the infilling databases, we can initialise our infiller.

# %%
infiller = create_scijune2026_infiller(
    infilling_leader_emissions_file=INFILLING_DB_FILE,
    ghg_inversions_file=INFILLING_DB_CFCS_FILE,
    historical_emissions_file=HISTORICAL_EMISSIONS_FILE,
    harmonisation_year=harmonisation_year,
    pre_industrial_year=1750,
    run_checks=True,
)


# %% [markdown]
# And infill. Right now the result is the infilled + harmonised dataframe

# %% editable=true slideshow={"slide_type": ""}
complete = infiller(harmonised)
complete

# %% [markdown]
# You can see infilled pathways compared to raw pathways in the below.
#
# This is not a deep analysis.
# If you want to see the full details of how the infilling worked in AR6,
# see [Lamboll et al., 2020](https://doi.org/10.5194/gmd-13-5259-2020).
#
# The returned data is not just the infilled timeseries but the complete scenarios,
# ready for running our simple climate models.

# %%
pdf = (
    pix.concat(
        [
            pre_processed.pix.assign(stage="pre_processed"),
            harmonised.pix.assign(stage="harmonised"),
            complete.pix.assign(stage="infilled"),
        ]
    )
    .loc[pix.ismatch(variable=["**CO2|Fossil", "**CH4", "**N2O", "**SOx"])]
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

# %% [markdown]
# ## SCM Running
#
# The next step is running a simple climate model (SCM).
# This provides information about the climate implications of the scenario(s).
#
# In AR6, MAGICCv7.5.3 was used for categorisation,
# but other climate models were also available.
# MAGICCv7.5.3 can be downloaded from magicc.org,
# but a copy is also stored in the path below
# so we can run these docs.
# Please go and download from magicc.org
# to help the MAGICC developers see the usage of their model.
#
# Under the hood, the AR6 SCM running uses the
# [OpenSCM-Runner](https://github.com/openscm/openscm-runner) package.

# %%
MAGICC_EXE_PATH = Path(
    "tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs/magicc-v7.6.0a3/bin"
)
MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE = Path(
    "tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs/magicc-v7.6.0a3/configs/magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
# Some trickery to make sure we pick up files in the right path,
# even when building the docs :)
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
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib/gcc/current/"

elif platform.system() == "Linux":
    MAGICC_EXE = MAGICC_EXE_PATH / "magicc"

elif platform.system() == "Windows":
    MAGICC_EXE = MAGICC_EXE_PATH / "magicc.exe"

# %% [markdown]
# With the MAGICC executable and config file, we can initialise our SCM runner.

# %%
scm_runner = SCIJune2026SCMRunner.from_cmip7_scenariomip_config(
    magicc_exe_path=MAGICC_EXE,
    magicc_prob_distribution_path=MAGICC_CMIP7_SCENARIOMIP_PROBABILISTIC_CONFIG_FILE,
    output_variables=("Surface Air Temperature Change", "Effective Radiative Forcing"),
    historical_emissions_path=HISTORICAL_EMISSIONS_FILE,
    harmonisation_year=harmonisation_year,
    # Generally, you want to run SCMs in parallel
    n_processes=multiprocessing.cpu_count(),
    batch_size_scenarios=15,
)

# %% [markdown]
# If you're reading this on RtD,
# note that we run a greatly reduced number of ensemble members.
# You will likely want to skip this step if running yourself.

# %% editable=true slideshow={"slide_type": ""}
if os.environ.get("READTHEDOCS", "False") == "True":
    scm_runner.climate_models_cfgs["MAGICC7"] = scm_runner.climate_models_cfgs[
        "MAGICC7"
    ][:10]

# %% [markdown]
# And then run

# %%
scm_results = scm_runner(complete)
scm_results

# %% [markdown]
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
#
# The default uses the same settings as in `ScenarioMIP-CMIP7`.

# %%
post_processor = SCIJune2026PostProcessor.from_cmip7_scenariomip_config()
post_processed_results = post_processor(scm_results)

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

# %%
post_processed_results.timeseries_quantile.loc[:, 2000:].openscm.plot_plume(
    style_var="variable",
    quantiles_plumes=(
        (0.5, 0.8),
        ((0.05, 0.95), 0.3),
    ),
)
