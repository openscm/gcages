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
# # How to run the AR6 workflow
#
# Here we demonstrate how to run the workflow that was used in AR6.
# This is intended to demonstrate how to use the package in a familiar context.
# If you want a simpler interface for doing this,
# please see the
# [climate-assessment](https://github.com/iiasa/climate-assessment) package,
# which is a [facade](https://refactoring.guru/design-patterns/facade)
# around gcages.
#
# Note: this is not yet complete, we will add further steps in future.

# %% [markdown]
# ## Imports

# %%
from functools import partial
from pathlib import Path

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pint
import seaborn as sns

from gcages.ar6 import (
    AR6Harmoniser,
    AR6Infiller,
    AR6PreProcessor,
    get_ar6_full_historical_emissions,
)

# %%
# Setup pint
pint.set_application_registry(openscm_units.unit_registry)

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
# [or our docs on naming conventions](../../tutorials/understanding-the-naming-conventions).
# In short, we use the `gcages` naming conventions throughout.
# However, AR6 used the IAMC naming convention as its starting point.
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

# %%
# Put it altogether into a DataFrame
start = pd.DataFrame(
    np.vstack(
        [
            co2_flatline,
            ch4_flatline,
            co2_flatline,
            co2_decline,
            ch4_decline,
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

# %%
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
pre_processor = AR6PreProcessor.from_ar6_config(
    n_processes=None,  # run serially for this demo
)

# %%
# These are the variables (using the IAMC naming convention)
# understood by the workflow as was used in AR6.
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

# %%
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
# In AR6, a specific set of historical emissions was used.
# There is no official home for this, but a copy is stored in the path below
# (and, as above, if you want something that pre-packages everything,
# see the [climate-assessment](https://github.com/iiasa/climate-assessment) package).
#
# Under the hood, the AR6 harmonisation uses the
# [aneris](https://github.com/iiasa/aneris) package.

# %%
AR6_HISTORICAL_EMISSIONS_FILE = Path(
    "tests/regression/ar6/ar6-workflow-inputs/history_ar6.csv"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
# Some trickery to make sure we pick up files in the right path,
# even when building the docs :)
if not AR6_HISTORICAL_EMISSIONS_FILE.exists():
    AR6_HISTORICAL_EMISSIONS_FILE = Path("../..") / AR6_HISTORICAL_EMISSIONS_FILE
    if not AR6_HISTORICAL_EMISSIONS_FILE.exists():
        raise AssertionError

# %% [markdown]
# With this file, we can initialise a harmoniser exactly like that used in AR6.

# %%
harmoniser = AR6Harmoniser.from_ar6_config(
    ar6_historical_emissions_file=AR6_HISTORICAL_EMISSIONS_FILE,
    n_processes=None,  # run serially for this demo
)

# %% [markdown]
# And harmonise

# %%
harmonised = harmoniser(pre_processed)
harmonised

# %% [markdown]
# You can see the modification to the pathways
# as a result of the harmonisation in the plot below.
# In scenarios that have more emissions,
# the same idea is applied to all variables.

# %%
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
# This is the process of inferring any emissions which are not included in the scenarios,
# but are needed for running climate models.
# In our example, this is things like N2O, black carbon, sulfates.
#
# In AR6, a specific infilling database was used.
# There is no official home for all of this, but a copy is stored in the path below
# (and, as above, if you want something that pre-packages everything,
# see the [climate-assessment](https://github.com/iiasa/climate-assessment) package).
#
# Under the hood, the AR6 infilling uses the
# [silicone](https://github.com/GranthamImperial/silicone) package.

# %% editable=true slideshow={"slide_type": ""}
AR6_INFILLING_DB_FILE = Path(
    "tests/regression/ar6/ar6-workflow-inputs/infilling_db_ar6.csv"
)
AR6_INFILLING_DB_CFCS_FILE = Path(
    "tests/regression/ar6/ar6-workflow-inputs/infilling_db_ar6_cfcs.csv"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
# Some trickery to make sure we pick up files in the right path,
# even when building the docs :)
if not AR6_INFILLING_DB_FILE.exists():
    AR6_INFILLING_DB_FILE = Path("../..") / AR6_INFILLING_DB_FILE
    if not AR6_INFILLING_DB_FILE.exists():
        raise AssertionError

if not AR6_INFILLING_DB_CFCS_FILE.exists():
    AR6_INFILLING_DB_CFCS_FILE = Path("../..") / AR6_INFILLING_DB_CFCS_FILE
    if not AR6_INFILLING_DB_CFCS_FILE.exists():
        raise AssertionError

# %% [markdown]
# With the infilling databases, we can initialise our infiller.

# %%
infiller = AR6Infiller.from_ar6_config(
    ar6_infilling_db_file=AR6_INFILLING_DB_FILE,
    ar6_infilling_db_cfcs_file=AR6_INFILLING_DB_CFCS_FILE,
    n_processes=None,  # run serially for this demo
    # To make sure that our outputs remain harmonised
    # (also, turns out that the historical emissions
    # are the same as the CFCs database)
    historical_emissions=get_ar6_full_historical_emissions(AR6_INFILLING_DB_CFCS_FILE),
    harmonisation_year=harmoniser.harmonisation_year,
)

# %% [markdown]
# And infill

# %%
harmonised

# %%
infilled = infiller(harmonised)
infilled

# %% [markdown]
# You can see infilled pathways compared to raw pathways in the below.
# A few things to notice:
#
# - only required variables are infilled
#   (e.g. CH<sub>4</sub> is only infilled for `flatline-co2-only`, not `flatline`)
#   and there can be notable differences in the emissions used without infilling
# - the infilling is largely CO<sub>2</sub> driven,
#   but it's not as simple as 'higher CO<sub>2</sub>' means higher everything else
#   (see e.g. the N<sub>2</sub>O plot where `flatline` has lower N<sub>2</sub>O
#   than `decline` in 2060)
#
# This is not a deep analysis.
# If you want to see the full details of how the infilling worked in AR6,
# see [Lamboll et al., 2020](https://doi.org/10.5194/gmd-13-5259-2020).

# %%
pdf = (
    pix.concat(
        [
            pre_processed.pix.assign(stage="pre_processed"),
            harmonised.pix.assign(stage="harmonised"),
            infilled.pix.assign(stage="infilled"),
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
# The returned data is just the infilled timeseries.
# We can combine these with the harmonised data to create complete scenarios,
# ready for running our simple climate models.

# %%
complete_scenarios = pd.concat([harmonised, infilled])
complete_scenarios

# %% [markdown]
# ## SCM Running
#
# TBD

# %% [markdown]
# ## Post-processing
#
# TBD
