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
# # How to run the SCI scenarios with the CMIP7 ScenarioMIP workflow
#
# Note: this is not yet complete, we will add further steps in future.

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import multiprocessing
import os
import platform
from functools import partial
from pathlib import Path

import openscm_units
import pandas_indexing as pix
import pandas_openscm
import pint
import seaborn as sns
import pandas as pd
from pandas_openscm.io import load_timeseries_csv

import numpy as np

import gcages.cmip7_scenariomip.pre_processing.reaggregation.basic
from gcages.cmip7_scenariomip.harmonisation import (
    create_cmip7_scenariomip_global_harmoniser,
)
from gcages.cmip7_scenariomip.infilling import CMIP7ScenarioMIPInfiller
from gcages.cmip7_scenariomip.post_processing import CMIP7ScenarioMIPPostProcessor
from gcages.cmip7_scenariomip.pre_processing import CMIP7ScenarioMIPPreProcessor
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_required_timeseries_index,
)
from gcages.cmip7_scenariomip.scm_running import CMIP7ScenarioMIPSCMRunner
from gcages.completeness import get_missing_levels
from gcages.index_manipulation import split_sectors
from gcages.renaming import convert_variable_name, SupportedNamingConventions
from pandas_openscm.index_manipulation import (
        update_index_levels_func,
        set_index_levels_func,
        update_levels_from_other,
    )

# %%
# Setup pint
pint.set_application_registry(openscm_units.unit_registry)

# %%
pandas_openscm.register_pandas_accessors()

from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    REQUIRED_MODEL_REGION_VARIABLES,
)

# %%
# Fallback units for species not present in the input data
SPECIES_UNIT_FALLBACK = {
    'BC': 'Mt BC/yr',
    'CH4': 'Mt CH4/yr',
    'CO': 'Mt CO/yr',
    'N2O': 'kt N2O/yr',
    'NH3': 'Mt NH3/yr',
    'NOx': 'Mt NO2/yr',
    'OC': 'Mt OC/yr',
    'Sulfur': 'Mt SO2/yr',
    'VOC': 'Mt VOC/yr',
}


def distribute_co2_subtotals(start: pd.DataFrame) -> pd.DataFrame:
    """
    For models that only report CO2 as EIP/AFOLU sub-totals with no sector breakdown,
    add proxy sector-level variables so the pre-processor gets the right CO2 values.

    - ``Emissions|CO2|Energy and Industrial Processes`` → ``Emissions|CO2|Energy|Supply``
      (a required fossil sector; ``Energy Sector`` maps to CO2_FOSSIL_SECTORS_GRIDDING)
    - ``Emissions|CO2|AFOLU`` → ``Emissions|CO2|AFOLU|Agricultural Waste Burning``
      (an optional biosphere sector; maps to CO2_BIOSPHERE_SECTORS_GRIDDING)

    These proxies are only added when no required CO2 sector variables are already present,
    so models that do report sector-level data (like IMACLIM 2.0) are unaffected.
    """
    world_data = start[start.index.get_level_values("region") == "World"]
    existing_vars = set(start.index.get_level_values("variable"))
    required_co2_sectors = {
        v for v in REQUIRED_MODEL_REGION_VARIABLES if v.startswith("Emissions|CO2|")
    }

    additions = []

    # EIP total -> Energy|Supply 
    eip_var = "Emissions|CO2|Energy and Industrial Processes"
    fossil_proxy = "Emissions|CO2|Energy|Supply"
    if eip_var in existing_vars and not (existing_vars & required_co2_sectors):
        eip_data = world_data[world_data.index.get_level_values("variable") == eip_var]
        if not eip_data.empty:
            additions.append(eip_data.rename(index={eip_var: fossil_proxy}, level="variable"))

    # AFOLU total -> AFOLU|Agricultural Waste Burning 
    afolu_var = "Emissions|CO2|AFOLU"
    biosphere_proxy = "Emissions|CO2|AFOLU|Agricultural Waste Burning"
    afolu_sub_vars = {v for v in existing_vars if v.startswith("Emissions|CO2|AFOLU|")}
    if afolu_var in existing_vars and not afolu_sub_vars:
        afolu_data = world_data[world_data.index.get_level_values("variable") == afolu_var]
        if not afolu_data.empty:
            additions.append(afolu_data.rename(index={afolu_var: biosphere_proxy}, level="variable"))

    if additions:
        return pd.concat([start, *additions])
    return start


def missing_reporting_zero_hack(reaggregator, model_df, model_regions):
    """Fill required but missing timeseries with zeros, guessing units from data or fallback map."""
    required_index = get_required_timeseries_index(
        model_regions=model_regions,
        world_region=reaggregator.world_region,
        region_level=reaggregator.region_level,
        variable_level=reaggregator.variable_level,
    )

    variable_unit_map = {
        "|".join(v.split("|")[:2]): u
        for v, u in model_df.index.droplevel(
            model_df.index.names.difference(["variable", "unit"])
        )
        .drop_duplicates()
        .to_list()
    }

    def guess_unit(v_in: str) -> str:
        for k, v in variable_unit_map.items():
            if v_in.startswith(f"{k}|") or v_in == k:
                return v
        parts = v_in.split("|")
        if len(parts) >= 2:
            species = parts[1]
            if species in SPECIES_UNIT_FALLBACK:
                return SPECIES_UNIT_FALLBACK[species]
        return None

    tmp_l = []
    for (model_l, scenario), sdf in model_df.groupby(["model", "scenario"]):
        mls = get_missing_levels(
            sdf.index,
            required_index,
            unit_col=reaggregator.unit_level,
        )

        filled_index = update_levels_from_other(mls, {"unit": ("variable", guess_unit)})
        filled_values = np.zeros((mls.shape[0], sdf.shape[1]))

        # For species that report only a World-level total (no sector breakdown),
        # assign the total to the first required model-region sector so that
        # to_global_workflow_emissions sums to the correct non-zero total.
        world_data = sdf[sdf.index.get_level_values("region") == reaggregator.world_region]
        existing_vars = set(sdf.index.get_level_values("variable"))
        distributed_species: set[str] = set()

        missing_vars = filled_index.get_level_values("variable")
        missing_regions = filled_index.get_level_values("region")

        for i, (var, region) in enumerate(zip(missing_vars, missing_regions)):
            if region == reaggregator.world_region or not var.startswith("Emissions|"):
                continue
            parts = var.split("|")
            if len(parts) < 3:
                continue
            species = parts[1]
            if species in distributed_species:
                continue
            total_var = f"Emissions|{species}"
            has_sector_data = any(
                v.startswith(f"Emissions|{species}|") for v in existing_vars
            )
            if has_sector_data or total_var not in existing_vars:
                continue
            # Find the first (alphabetically) missing model-region var for this species
            first_sector_var = min(
                v for v in missing_vars if v.startswith(f"Emissions|{species}|")
            )
            if var != first_sector_var:
                continue
            world_total = world_data[
                world_data.index.get_level_values("variable") == total_var
            ]
            if world_total.empty:
                continue
            filled_values[i] = world_total.values[0]
            distributed_species.add(species)

        zeros_hack = pd.DataFrame(
            filled_values,
            columns=sdf.columns,
            index=filled_index,
        )
        zeros_hack = set_index_levels_func(
            zeros_hack,
            {"model": model_l, "scenario": scenario},
        ).reorder_levels(sdf.index.names)
        sdf_full = pix.concat([sdf, zeros_hack])

        tmp_l.append(sdf_full)

    res = pix.concat(tmp_l)
    return res


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Start

# %% editable=true slideshow={"slide_type": ""}
EXAMPLE_INPUT_FILE = Path(
    "./SCI-2025_v1.0_pathways_ensemble_global.xlsx"
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove_input"]
# Some trickery to make sure we pick up files in the right path,
# even when building the docs :)
if not EXAMPLE_INPUT_FILE.exists():
    EXAMPLE_INPUT_FILE = Path("../..") / EXAMPLE_INPUT_FILE
    if not EXAMPLE_INPUT_FILE.exists():
        raise AssertionError

# %% editable=true slideshow={"slide_type": ""}
df_in = pd.read_excel(EXAMPLE_INPUT_FILE, sheet_name="data")

# %%
# Filtering, filling nans and interpolation
mask = df_in["Variable"].str.startswith("Emissions") | df_in["Variable"].str.startswith("Carbon Removal")
start = df_in[mask]
start.columns = start.columns.str.lower()
start = start.set_index(["model", "scenario","region","variable","unit"])
start.columns = start.columns.astype(int)
start = start.rename_axis("year", axis="columns")
start = start.interpolate(axis=1, method='linear').fillna(0)
all_years = range(start.columns.min(), start.columns.max() + 1)

start = start.reindex(columns=all_years)
start = start.interpolate(axis=1, method='linear')
start

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

# %%
mask = ((start.index.get_level_values("model")== "C-ROADS-5.005") & 
        (start.index.get_level_values("scenario") == "Ratchet-1.5°C-limCDR" ))
start = start[mask]
start

# %%
# Distribute CO2 sub-totals to proxy sector variables before creating the fake region.
# Models that only report EIP/AFOLU totals (no sector breakdown) would otherwise get
# zero CO2 Fossil/Biosphere in global_workflow_emissions.
start = distribute_co2_subtotals(start)

# Producing a fake regional reporting to use the cmip7 reaggregator
model_name = start.index.get_level_values("model").unique()[0]
df = start.reset_index()
df_new = df.copy()
df_new["region"] = f"{model_name}|Fake"
df = pd.concat([df,df_new], axis=0, ignore_index=True)
df = df.set_index(["model","scenario","region","variable","unit"])
df.columns = df.columns.rename("year")

models = df.index.get_level_values("model").unique()
model_regions = [
    r
    for r in df.index.get_level_values("region").unique()
    if any(r.startswith(model) for model in models)
]

reaggregator = (
    gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.ReaggregatorBasic(
        model_regions=model_regions
    )
)
# Adding 0 to what's currently missing
df = missing_reporting_zero_hack(reaggregator, df, model_regions)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Pre-processor
#
# Having defined our re-aggregator,
# we can initialise our pre-processor and do the pre-processing.

# %% editable=true slideshow={"slide_type": ""}
pre_processor = CMIP7ScenarioMIPPreProcessor(
    reaggregator=reaggregator,
    n_processes=None,  # run serially
    run_checks=False,
)

# %% editable=true slideshow={"slide_type": ""}
res_pre_processed = pre_processor(df)

# %% [markdown]
# Removing the artificially added sectors

# %%
res_pre_processed.global_workflow_emissions = res_pre_processed.global_workflow_emissions[(res_pre_processed.global_workflow_emissions != 0).any(axis=1)]

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Results
#
# The result contains a few pieces of information.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ##### Global workflow emissions
#
# The first bit of information is data which can be used with the global workflow.

# %%
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

# %% [markdown]
# ##### Gridding workflow emissions
#
# The second bit of information is data which can be used with the gridding workflow.
# This contains data at the region-sector level
# for a specific set of sectors.

# %%
data = split_sectors(
    res_pre_processed.gridding_workflow_emissions
).openscm.to_long_data(time_col_name="year")
species_to_plot = "CO2"
data = data[data["species"] == species_to_plot]
# data = data[data["sectors"] == "Energy Sector"]

fg = sns.relplot(
    data=data,
    style="scenario",
    style_order=sorted(data["scenario"].unique()),
    hue="sectors",
    hue_order=sorted(data["sectors"].unique()),
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

# %% [markdown]
# As part of this reporting,
# we also show which timeseries were assumed to be zero during the processing.
# (Lots of assumptions are not an issue,
# we simply include this for clarity and transparency.)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Harmonisation
#
# The next step is harmonisation.
# This is the process of aligning the scenarios with historical emissions estimates.
# We do this at both the global level
# and the level used for gridding.
#
# Historical emissions aligned with the rest of the CMIP7 exercise are used for this.
# These were processed in this repository:
# https://github.com/iiasa/emissions_harmonization_historical
# and are archived at https://zenodo.org/records/17845154.
#
# Under the hood, the harmonisation uses the
# [aneris](https://github.com/iiasa/aneris) package.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Global

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
harmonised_global = harmoniser_global(res_pre_processed.global_workflow_emissions)
harmonised_global

# %% [markdown]
# You can see the modification to the pathways as a result of
# the harmonisation in the plot below.

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
        col_wrap=3,
    )

pdf = (
    pix.concat(
        [
            res_pre_processed.global_workflow_emissions.pix.assign(
                stage="pre_processed"
            ),
            harmonised_global.pix.assign(stage="harmonised"),
            harmoniser_global.historical_emissions.pix.assign(
                stage="history", scenario="history", model="history"
            ).loc[
                pix.isin(
                    variable=res_pre_processed.global_workflow_emissions.pix.unique(
                        "variable"
                    )
                )
            ],
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
# I have created the infilling db from the harmonised input data.

# %%
BASE_DIR = Path("tests/regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs")
CMIP7_SCENARIOMIP_INFILLING_FILE = Path("./infilling_db_sci.csv")
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
