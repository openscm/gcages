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

import openscm_units
import pandas_openscm
import pint
import seaborn as sns
from pandas_openscm.io import load_timeseries_csv

import gcages.cmip7_scenariomip.pre_processing.reaggregation.basic
from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.index_manipulation import split_sectors

# %%
# Setup pint
pint.set_application_registry(openscm_units.unit_registry)

# %%
pandas_openscm.register_pandas_accessor()

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
    "tests/regression/cmip7-scenariomip/test-data/salted-202504-scenariomip-input.csv"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
# Some trickery to make sure we pick up files in the right path,
# even when building the docs :)
if not EXAMPLE_INPUT_FILE.exists():
    EXAMPLE_INPUT_FILE = Path("../..") / EXAMPLE_INPUT_FILE
    if not EXAMPLE_INPUT_FILE.exists():
        raise AssertionError

# %% editable=true slideshow={"slide_type": ""}
start = load_timeseries_csv(
    EXAMPLE_INPUT_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
start.columns.name = "year"
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

data = start.openscm.to_long_data(time_col_name="year")
data = data[
    data["variable"].isin(
        ["Emissions|CO2", "Emissions|CH4", "Emissions|Sulfur", "Emissions|BC"]
    )
    & (data["region"].isin(["World"]) | data["region"].str.startswith("model_1"))
]

fg = sns.relplot(
    data=data,
    hue="scenario",
    style="region",
    kind="line",
    linewidth=2.0,
    alpha=0.7,
    facet_kws=dict(sharey=False),
    x="year",
    y="value",
    col="variable",
    col_wrap=2,
)

for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, linestyle="--", color="gray")
    else:
        ax.set_ylim(ymin=0.0)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Pre-process
#
#
# The first step is pre-processing the emissions scenario(s)
# to compile the sectors which are used for gridding.
# This step also produces data sets
# that can be used with the 'standard' global-only workflows.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Re-aggregator
#
# The major challenge in processing is knowing which re-aggregator to use.
# At present, we only support one re-aggregation method,
# but we may need to support more in future, hence this idea.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# For this data, we need to use the basic `ReaggregatorBasic` class.
# When initialising our re-aggregator,
# we also need to tell it which model regions to use
# (models often report data in aggregate regions too,
# and we need to make sure this doesn't trip things up).

# %% editable=true slideshow={"slide_type": ""}
model_regions = [
    r
    for r in start.index.get_level_values("region").unique()
    if r.startswith("model_1")
]
reaggregator = (
    gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.ReaggregatorBasic(
        model_regions=model_regions
    )
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Pre-processor
#
# Having defined our re-aggregator,
# we can initialise our pre-processor and do the pre-processing.

# %% editable=true slideshow={"slide_type": ""}
pre_processor = CMIP7ScenarioMIPPreProcessor(
    reaggregator=reaggregator,
    n_processes=None,  # run serially
)

# %% editable=true slideshow={"slide_type": ""}
res_pre_processed = pre_processor(start)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# As an aside, it is possible to let gcages
# guess the re-aggregator to use,
# it's also more confusing when it goes wrong though.

# %% editable=true slideshow={"slide_type": ""}
pre_processor_guess_reaggregator = CMIP7ScenarioMIPPreProcessor(
    # Don't specifiy the re-aggregator, let gcages guess
    # reaggregator=reaggregator,
    n_processes=None,  # run serially
)
# The guessing is not so intelligent, so we need to give it a hand
# by stripping out everything except 'World' data
# and model-region data
# (removing all other regional aggregations).
start_guessable = start.loc[
    (start.index.get_level_values("region") == "World")
    | start.index.get_level_values("region").str.startswith("model_1")
]
# The result is the same as the above
_ = pre_processor_guess_reaggregator(start_guessable)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Results
#
# The result contains a few pieces of information.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ##### Global workflow emissions
#
# The first bit of information is data which can be used with the global workflow.

# %% editable=true slideshow={"slide_type": ""}
fg = sns.relplot(
    data=res_pre_processed.global_workflow_emissions.openscm.to_long_data(
        time_col_name="year"
    ),
    hue="scenario",
    kind="line",
    linewidth=2.0,
    alpha=0.7,
    facet_kws=dict(sharey=False),
    x="year",
    y="value",
    col="variable",
    col_order=sorted(
        res_pre_processed.global_workflow_emissions.index.get_level_values(
            "variable"
        ).unique()
    ),
    col_wrap=4,
    height=3.0,
    aspect=1.0,
)

for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, linestyle="--", color="gray")
    else:
        ax.set_ylim(ymin=0.0)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ##### Gridding workflow emissions
#
# The second bit of information is data which can be used with the gridding workflow.
# This contains data at the region-sector level
# for a specific set of sectors.

# %% editable=true slideshow={"slide_type": ""}
data = split_sectors(
    res_pre_processed.gridding_workflow_emissions
).openscm.to_long_data(time_col_name="year")
species_to_plot = "CO2"
data = data[data["species"] == species_to_plot]
# data = data[data["sectors"] == "Energy Sector"]

fg = sns.relplot(
    data=data,
    hue="scenario",
    hue_order=sorted(data["scenario"].unique()),
    style="sectors",
    style_order=sorted(data["sectors"].unique()),
    col="region",
    col_order=sorted(data["region"].unique()),
    kind="line",
    linewidth=2.0,
    alpha=0.7,
    facet_kws=dict(sharey=False),
    x="year",
    y="value",
    col_wrap=3,
    height=4.0,
    aspect=1.0,
)

for ax in fg.axes.flatten():
    ax.axhline(0.0, linestyle="--", color="gray")

fg.fig.suptitle(species_to_plot, y=1.02)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# As part of this reporting,
# we also show which timeseries were assumed to be zero during the processing.
# (Lots of assumptions are not an issue,
# we simply include this for clarity and transparency.)

# %%
res_pre_processed.assumed_zero_emissions

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Harmonisation
#
# The next step is harmonisation.
# This is the process of aligning the scenarios with historical emissions estimates.
# We do this at both the global level
# and the level used for gridding.
#
# Historical emissions aligned with the rest of the CMIP7 exercise are used for this.
# These are processed in this repository:
# https://github.com/iiasa/emissions_harmonization_historical
# and are archived at [TODO Zenodo upload and link].
#
# Under the hood, the harmonisation uses the
# [aneris](https://github.com/iiasa/aneris) package.

# %% [markdown]
# ### Global

# %%
# TBD

# %% [markdown]
# ### Region-sector (i.e. gridding level)

# %%
# TBD

# %% [markdown]
# You can see the modification to the pathways
# as a result of the harmonisation in the plot below.
# Here we obviously only show a selection of timeseries.

# %%
# pdf = (
#     pix.concat(
#         [
#             pre_processed.pix.assign(stage="pre_processed"),
#             harmonised.pix.assign(stage="harmonised"),
#             harmoniser.historical_emissions.pix.assign(
#                 stage="history", scenario="history", model="history"
#             ).loc[pix.isin(variable=pre_processed.pix.unique("variable"))],
#         ]
#     )
#     .melt(ignore_index=False, var_name="year")
#     .reset_index()
# )

# fg = relplot_in_emms(
#     data=pdf,
#     hue="scenario",
#     style="stage",
#     dashes={
#         "history": (1, 1),
#         "pre_processed": (3, 3),
#         "harmonised": "",
#     },
# )

# fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
# fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
# ## Infilling
#
# The next step is infilling.
# This is the process of inferring any emissions
# which are not included in the scenarios,
# but are needed for running climate models.
# In our example, this is things like CFC11, CFC12 and other gases covered
# under the Montreal Protocol
# and other greenhouse gases
# with a warming impact much smaller than the ones already included.
#
# In CMIP7 ScenarioMIP, a specific infilling database was used.
# You can see where it is archived below.
#
# Under the hood, the AR6 infilling uses the
# [silicone](https://github.com/GranthamImperial/silicone) package.

# %%
# TBD - download infiller db

# %% [markdown]
# With the infilling databases, we can initialise our infiller.

# %%
# TBD

# %% [markdown]
# And infill

# %%
# TBD

# %% [markdown]
# You can see infilled pathways compared to raw pathways in the below.
#
# This is not a deep analysis.
# If you want to see the full details of how the infilling works,
# see [Lamboll et al., 2020](https://doi.org/10.5194/gmd-13-5259-2020).

# %%
# pdf = (
#     pix.concat(
#         [
#             pre_processed.pix.assign(stage="pre_processed"),
#             harmonised.pix.assign(stage="harmonised"),
#             infilled.pix.assign(stage="infilled"),
#         ]
#     )
#     .loc[pix.ismatch(variable=["**CO2|Fossil", "**CH4", "**N2O", "**SOx"])]
#     .melt(ignore_index=False, var_name="year")
#     .reset_index()
# )

# fg = relplot_in_emms(
#     data=pdf,
#     hue="scenario",
#     style="stage",
#     dashes={
#         "pre_processed": (3, 3),
#         "harmonised": "",
#         "infilled": (1, 1),
#     },
# )

# fg.axes.flatten()[0].axhline(0.0, linestyle="--", color="gray")
# fg.axes.flatten()[1].set_ylim(ymin=0.0)

# %% [markdown]
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
# and the global harmonised timeseries
# complete_scenarios = pd.concat([harmonised, infilled])
# complete_scenarios

# %%
# MAGICC_EXE_PATH = Path("tests/regression/ar6/ar6-workflow-inputs/magicc-v7.5.3/bin")
# MAGICC_AR6_PROBABILISTIC_CONFIG_FILE = Path(
#     "tests/regression/ar6/ar6-workflow-inputs/magicc-ar6-0fd0f62-f023edb-drawnset/0fd0f62-derived-metrics-id-f023edb-drawnset.json"  # noqa: E501
# )

# %%
# if platform.system() == "Darwin":
#     if platform.processor() == "arm":
#         MAGICC_EXE = MAGICC_EXE_PATH / "magicc-darwin-arm64"
#         os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/gfortran/lib/gcc/current/"  # noqa: E501

# elif platform.system() == "Linux":
#     MAGICC_EXE = MAGICC_EXE_PATH / "magicc"

# elif platform.system() == "Windows":
#     MAGICC_EXE = MAGICC_EXE_PATH / "magicc.exe"

# %% [markdown]
# With the set up done, we can initialise our SCM runner.

# %%
# scm_runner = AR6SCMRunner.from_ar6_config(
#     # Generally, you want to run SCMs in parallel
#     n_processes=multiprocessing.cpu_count(),
#     magicc_exe_path=MAGICC_EXE,
#     magicc_prob_distribution_path=MAGICC_AR6_PROBABILISTIC_CONFIG_FILE,
#     historical_emissions=get_ar6_full_historical_emissions(
#           AR6_INFILLING_DB_CFCS_FILE),
#     harmonisation_year=2015,
#     output_variables=("Surface Air Temperature Change",
#        "Effective Radiative Forcing"),
# )

# %% [markdown]
# If you're reading this on RtD,
# note that we run a greatly reduced number of ensemble members.
# You will likely want to skip this step if running yourself.

# %%
# if os.environ.get("READTHEDOCS", False):
#     scm_runner.climate_models_cfgs["MAGICC7"] = scm_runner.climate_models_cfgs[
#         "MAGICC7"
#     ][:10]

# %% [markdown]
# And then run

# %%
# scm_results = scm_runner(complete_scenarios)
# scm_results

# %% [markdown]
# With these outputs, we can look at raw (i.e. before pre-processing) variables.

# %%
# scm_results.loc[
#     pix.isin(variable=["Effective Radiative Forcing"])
# ].openscm.plot_plume_after_calculating_quantiles(
#     style_var="variable",
#     quantile_over="run_id",
#     quantiles_plumes=(
#         (0.5, 0.8),
#         ((0.05, 0.95), 0.3),
#     ),
# )

# %% [markdown]
# ## Post-processing
#
# The last step is post-processing.
# This handles calculation of key pieces of metadata.

# %%
# post_processor = AR6PostProcessor.from_ar6_config(n_processes=None)
# post_processed_results = post_processor(scm_results)

# %% [markdown]
# For example, the scenario category.

# %%
# post_processed_results.metadata_categories.unstack("metric")

# %% [markdown]
# Exceedance thresholds.

# %%
# post_processed_results.metadata_exceedance_probabilities.unstack("threshold")

# %% [markdown]
# Key warming metrics.

# %%
# post_processed_results.metadata_quantile.loc[
#     pix.isin(quantile=[0.05, 0.5, 0.95])
# ].unstack(["quantile", "metric"]).round(2).sort_index(axis="columns")

# %% [markdown]
# Assessed surface temperatures.

# %%
# post_processed_results.timeseries_quantile.loc[:, 2000:].openscm.plot_plume(
#     style_var="variable",
#     quantiles_plumes=(
#         (0.5, 0.8),
#         ((0.05, 0.95), 0.3),
#     ),
# )
