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

from gcages.ar6 import AR6Harmoniser, AR6PreProcessor

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
#
# We will come to the emissions variables that are supported in the next part.
# For now, we are going to create a basic scenario
# with CO2 fossil and CH4 emissions.

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
        ]
    ),
    columns=time,
    index=pd.MultiIndex.from_tuples(
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
        ],
        names=["model", "scenario", "region", "variable", "unit"],
    ),
)
start = start.T.interpolate("index").T
start

# %%
assert False, "Cross-ref the naming conventions notebook and create a version that doesn't need pre-processing"

# %%
relplot_in_emms = partial(
    sns.relplot,
    kind="line",
    linewidth=2.0,
    facet_kws=dict(sharey=False),
    x="year",
    y="value",
    col="variable",
    style="scenario",
    col_wrap=2,
)
fg = relplot_in_emms(
    data=start.melt(ignore_index=False, var_name="year").reset_index(),
)

fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# ## Pre-process
#
# The first step is pre-processing the emissions scenario(s).
# This does some variable processing steps that were need in AR6,
# but shouldn't be needed if you are more careful with your data.
# It also extracts only those variables that are understood by the workflow.

# %%
pre_processor = AR6PreProcessor.from_ar6_config(
    n_processes=None,  # run serially for this demo
)

# %%
# These are the variables understood by the workflow as was used in AR6
pre_processor.emissions_out

# %% [markdown]
# Here we run the pre-processing and get the pre-processed data.
# In our case, this is effectively a no-op as our data is already in the right format.

# %%
pre_processed = pre_processor(start)
pre_processed

# %% [markdown]
# ## Harmonisation
#
# The next step is harmonisation.
# This is the process of aligning the scenarios with historical emissions estimates.
#
# In AR6, a specific set of historical emissions was used.
# There is no official home for this, but a copy is stored in the path below
# (and, as above, if you want something that pre-packages everything,
# see [climate-assessment](https://github.com/iiasa/climate-assessment) package).
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
pdf["variable"] = pdf["variable"].str.replace("AR6 climate diagnostics|Harmonized|", "")

fg = relplot_in_emms(
    data=pdf,
    hue="stage",
)

fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# ## Infilling
#
# TBD

# %% [markdown]
# ## SCM Running
#
# TBD

# %% [markdown]
# ## Post-processing
#
# TBD
