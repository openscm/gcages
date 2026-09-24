"""
Run OSCAR v4 (customized mode) on gcages emissions, through OSCAR's IAMC CSV input
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from attrs import define
from pandas_openscm.unit_conversion import convert_unit

from gcages.renaming import (
    SupportedNamingConventions,
    convert_variable_name,
    rename_variables,
)

if TYPE_CHECKING:
    import xarray as xr

OSCAR_HIST_END_YEAR = 2023

# cmip7_scenariomip name to key in vars_map_iamc-ar6-public.yaml
CMIP7_TO_OSCAR_REGISTRY_KEY = {
    **{
        f"Emissions|{s}": f"Emissions|CFC|{s}"
        for s in ("CFC11", "CFC12", "CFC113", "CFC114", "CFC115")
    },
    **{
        f"Emissions|{s}": f"Emissions|HCFC|{s}"
        for s in ("HCFC22", "HCFC141b", "HCFC142b")
    },
    **{
        f"Emissions|{s}": f"Emissions|Halon|{s}"
        for s in ("Halon1211", "Halon1301", "Halon2402")
    },
    "Emissions|CF4": "Emissions|PFC|CF4",
    "Emissions|C8F18": "Emissions|PFC|C8F18",
    "Emissions|HFC|HFC43-10": "Emissions|HFC|HFC43-10mee",
    "Emissions|cC4F8": "Emissions|c-C4F8",
}

# Not passed to OSCAR (dropped with a warning)
NOT_SUPPORTED = {
    # D_Eluc is computed by OSCAR, not a forcing: passing it NaNs the whole ensemble
    "Emissions|CO2|Biosphere": "land-use CO2 comes from baseline_forcing",
    "Emissions|Halon1202": "not in OSCAR's registry",
}

OSCAR_OUTPUT_VARIABLES = pd.DataFrame(
    [
        ("D_Tg", "Surface Air Temperature Change", "K"),
        ("ERF", "Effective Radiative Forcing", "W/m^2"),
        ("ERF_CO2", "Effective Radiative Forcing|CO2", "W/m^2"),
        ("ERF_CH4", "Effective Radiative Forcing|CH4", "W/m^2"),
        ("ERF_N2O", "Effective Radiative Forcing|N2O", "W/m^2"),
        ("ERF_aer", "Effective Radiative Forcing|Aerosols", "W/m^2"),
        ("ERF_O3", "Effective Radiative Forcing|Ozone", "W/m^2"),
        ("CO2", "Atmospheric Concentrations|CO2", "ppm"),
        ("CH4", "Atmospheric Concentrations|CH4", "ppb"),
        ("N2O", "Atmospheric Concentrations|N2O", "ppb"),
        ("d_OHC", "Heat Uptake", "W/m^2"),
    ],
    columns=["oscar", "gcages", "unit"],
)
"""OSCAR -> gcages output name"""


def emissions_to_oscar_csv(
    indf: pd.DataFrame, scen_labels: dict[tuple[str, str], str]
) -> pd.DataFrame:
    """
    Convert gcages emissions (World) to OSCAR's wide IAMC CSV table

    OSCAR never reads the `Unit` column,
    so values are converted to the units its registry expects.

    Parameters
    ----------
    indf
        Input data

    scen_labels
        Scenario labels

    Returns
    -------
    :
        DataFrame needed for OSCAR
    """
    import openscm_units  # noqa: PLC0415
    from oscar._io.handlers import load_var_mapping  # type: ignore # noqa: PLC0415

    registry = load_var_mapping("iamc-ar6-public")["anthropogenic_emissions"]

    emms = indf[~indf.index.isin(list(NOT_SUPPORTED), level="variable")]

    emms = rename_variables(
        emms,
        from_convention=SupportedNamingConventions.GCAGES,
        to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    )
    keys = [
        CMIP7_TO_OSCAR_REGISTRY_KEY.get(v, v)
        for v in emms.index.get_level_values("variable")
    ]
    # Registry units spell some species with a hyphen, which pint cannot parse
    units = [registry[k][3].replace("-", "") for k in keys]
    emms = convert_unit(
        emms,
        pd.Series(units, index=emms.index.droplevel("unit")),
        ur=openscm_units.unit_registry,  # type: ignore[arg-type]
    )

    model_scenario = zip(
        emms.index.get_level_values("model"), emms.index.get_level_values("scenario")
    )
    return pd.DataFrame(
        {
            "Model": "gcages",
            # OSCAR ignores Model and strips whitespace from Scenario
            "Scenario": [scen_labels[ms] for ms in model_scenario],
            "Region": "World",
            "Variable": keys,
            "Unit": [registry[k][3] for k in keys],
        }
    ).join(emms.reset_index(drop=True))


def oscar_results_to_df(
    results: xr.Dataset,
    scen_labels: dict[tuple[str, str], str],
    oscar_variables: list[str],
) -> pd.DataFrame:
    """
    Convert OSCAR results to gcages timeseries, one row per ensemble member

    OSCAR's shared `Historical` run is joined onto each scenario.

    Parameters
    ----------
    results
        OSCAR results

    scen_labels
        Scenario labels

    oscar_variables
        Mapping

    Returns
    -------
    :
        DataFrame needed for OSCAR
    """
    hist = results.sel(scen="Historical", year=slice(None, OSCAR_HIST_END_YEAR))
    scens = results.sel(year=slice(OSCAR_HIST_END_YEAR + 1, None))
    output = OSCAR_OUTPUT_VARIABLES.set_index("oscar")

    out = []
    for (model, scenario), label in scen_labels.items():
        for ovar in oscar_variables:
            hist_df = cast(pd.DataFrame, hist[ovar].to_pandas())
            scen_df = cast(pd.DataFrame, scens[ovar].sel(scen=label).to_pandas())
            df = pd.concat([hist_df, scen_df]).T.rename_axis("run_id")
            name, unit = output.at[ovar, "gcages"], output.at[ovar, "unit"]
            out.append(
                pd.concat(
                    {(model, scenario, name, unit): df},
                    names=["model", "scenario", "variable", "unit"],
                )
            )

    res: pd.DataFrame = pd.concat(out)
    res.columns = res.columns.astype(int)
    res = res.assign(climate_model="OSCARv4-beta2", region="World").set_index(
        ["climate_model", "region"], append=True
    )

    res = res.reorder_levels(
        ["climate_model", "model", "scenario", "region", "run_id", "unit", "variable"]
    )

    return res


@define
class OSCARSCMRunner:
    """OSCAR v4 runner (customized mode, IAMC CSV input, full 500-member ensemble)"""

    output_variables: tuple[str, ...] = ("Surface Air Temperature Change",)
    """
    Variables to include in the output
    """
    hist_type: str = "CMIP7"
    """
    Historical emissions
    """
    model_region: str = "IAMC_R5"
    """
    Model region
    """
    baseline_forcing: str = "scen7-VL"  # every driver we do not supply, incl. land use
    """
    Baseline forcing
    """
    keep_workdir: Path | None = None
    """
    Keep working directory or use temporary
    """

    def __call__(self, emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Run OSCAR on `emissions` (gcages naming convention)

        Parameters
        ----------
        emissions
            Emissions to run

        Returns
        -------
        :
            Raw results from the simple climate model
        """
        import oscar  # type: ignore # noqa: PLC0415
        import xarray as xr  # noqa: PLC0415
        import yaml  # type: ignore # noqa: PLC0415

        oscar_variables = [
            convert_variable_name(v, "gcages", "oscar", OSCAR_OUTPUT_VARIABLES)  # type: ignore[arg-type]
            for v in self.output_variables
        ]
        # Build scenario and parameters
        model_scenarios = emissions.index.droplevel(
            emissions.index.names.difference(["model", "scenario"])  # type: ignore # pandas-stubs out of date
        ).unique()
        scen_labels = {ms: f"gcages{i:05d}" for i, ms in enumerate(model_scenarios)}

        settings = {
            "output_identifier": "gcages",
            "user_files": {
                "csv_inputs": {
                    "atmospheric": {
                        "file": "emissions.csv",
                        "format": "iamc-ar6-public",
                    },
                    "lulcc": {"file": None},
                },
                "compiled_nc": None,
            },
            "scientific_setup": {
                "hist_type": self.hist_type,
                "connect_method": "raw",  # input is already harmonised
                "baseline_forcing": self.baseline_forcing,
                "model_region": self.model_region,
                "projection_end_year": int(max(emissions.columns)),
            },
            "theme": "custom",
            "custom_vars": oscar_variables,
            "plot_user_forcing": False,
            "plot_outputs": False,
        }

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(self.keep_workdir or tmp).resolve()
            project.mkdir(parents=True, exist_ok=True)
            emissions_to_oscar_csv(emissions, scen_labels).to_csv(
                project / "emissions.csv", index=False
            )
            (project / "settings_gcages.yaml").write_text(yaml.safe_dump(settings))
            # An absolute project path overrides OSCAR's projects folder
            oscar.run(mode="customized", project=str(project), experiment="gcages")
            results = xr.load_dataset(project / "results" / "gcages_results.nc")

        res = oscar_results_to_df(results, scen_labels, oscar_variables)
        if res.isnull().any().any():
            msg = "NaN in OSCAR output"
            raise AssertionError(msg)

        return res
