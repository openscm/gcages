"""
The CMIP7 ScenarioMIP marker scenarios and their published results
"""

from __future__ import annotations

import re

import pandas as pd
from attrs import define

from gcages.exceptions import MissingOptionalDependencyError
from gcages.testing import assert_frame_equal
from gcages.workflows.cmip7_scenariomip.inputs import (
    Downloader,
    PoochDownloader,
    ZenodoFile,
    convert_cmip7_scenariomip_names_to_gcages,
    get_file,
)


@define(frozen=True)
class CMIP7ScenarioMIPMarker:
    """
    A CMIP7 ScenarioMIP marker scenario
    """

    model: str
    """Model"""

    scenario: str
    """
    Name of the scenario as submitted

    This is the name used in the raw data
    and in the harmonisation methods used by the workflow.
    """

    ssp_submission_version: int
    """Version of the submitted data (in the ssp_submission database) that was used"""

    published_scenario: str
    """
    Name of the scenario in the published results

    The scenarios were renamed between submission and publication.
    This is also the name used on the ScenarioMIP ixmp4 platforms
    (e.g. scenariomip-internal).
    """

    ssp_submission_data_checksum: str
    """
    Checksum of the data used in the workflow

    This is the checksum (calculated with
    [get_data_checksum][gcages.hashing.get_data_checksum])
    of the timeseries used in the workflow
    (see [get_used_locator][gcages.workflows.cmip7_scenariomip.pre_processing.get_used_locator])
    from version `ssp_submission_version` in the ssp_submission database.
    It can be used to check that downloaded data is what we expect.
    """  # noqa: E501


CMIP7_SCENARIOMIP_MARKERS: tuple[CMIP7ScenarioMIPMarker, ...] = (
    CMIP7ScenarioMIPMarker(
        model="AIM 3.0",
        scenario="SSP2 - Low Overshoot_a",
        ssp_submission_version=25,
        published_scenario="Low-to-Negative - SSP2 (Marker)",
        ssp_submission_data_checksum=(
            "55a44324f3b8cfb1f0f92f7e290e4c17ae804cf16c237aa2415a8ed0d58e9a49"
        ),
    ),
    CMIP7ScenarioMIPMarker(
        model="COFFEE 1.6",
        scenario="SSP2 - Medium-Low Emissions",
        ssp_submission_version=14,
        published_scenario="Medium-to-Low - SSP2 (Marker)",
        ssp_submission_data_checksum=(
            "fd3cfdf0e8b38542ed579efa9c30204986c0b44ffee99213e5777080d20b86f9"
        ),
    ),
    CMIP7ScenarioMIPMarker(
        model="GCAM 8s",
        scenario="SSP3 - High Emissions",
        ssp_submission_version=3,
        published_scenario="High - SSP3 (Marker)",
        ssp_submission_data_checksum=(
            "e4e8aacec4f795f03def13b0dbd90564a7ec35b8f13832622cb262c1614bdf13"
        ),
    ),
    CMIP7ScenarioMIPMarker(
        model="IMAGE 3.4",
        scenario="SSP2 - Medium Emissions",
        ssp_submission_version=29,
        published_scenario="Medium - SSP2 (Marker)",
        ssp_submission_data_checksum=(
            "5214fa6de1e0b99e2180bad86386336012045ba4309e8871c29b10e125b7d390"
        ),
    ),
    CMIP7ScenarioMIPMarker(
        model="MESSAGEix-GLOBIOM-GAINS 2.1-M-R12",
        scenario="SSP2 - Low Emissions",
        ssp_submission_version=24,
        published_scenario="Low - SSP2 (Marker)",
        ssp_submission_data_checksum=(
            "9ecc1f20555bf31aaf6efd25cc258f19bbb451118f1c0a39227b0e0e554787fa"
        ),
    ),
    CMIP7ScenarioMIPMarker(
        model="REMIND-MAgPIE 3.5-4.11",
        scenario="SSP1 - Very Low Emissions",
        ssp_submission_version=5,
        published_scenario="Very Low - SSP1 (Marker)",
        ssp_submission_data_checksum=(
            "baeeeca7ae0da7de4fe054e7f1036550720ac48e1a9f1ee0f887b24f30a0868e"
        ),
    ),
    CMIP7ScenarioMIPMarker(
        model="WITCH 6.0",
        scenario="SSP5 - Medium-Low Emissions_a",
        ssp_submission_version=34,
        published_scenario="High-to-Low - SSP5 (Marker)",
        ssp_submission_data_checksum=(
            "e89522269f622c9c2e9682474e91d9856b27e95da71a0a524b17852ab5ec62e9"
        ),
    ),
)
"""
The CMIP7 ScenarioMIP marker scenarios

These are the scenarios in v0.2 of the harmonised and infilled emissions
(https://zenodo.org/records/19825038).
"""

PUBLISHED_EMISSIONS_FILE = ZenodoFile(
    record_id="19825038",
    filename="ScenarioMIP_emissions_marker_scenarios_v0.2.xlsx",
    md5="de849eb901770cb11e74a6ceb28a066d",
)
"""v0.2 of the harmonised and infilled emissions for the marker scenarios"""


def convert_published_emissions_to_gcages(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the published harmonised and infilled emissions to the names we use

    Parameters
    ----------
    raw
        Results as loaded from the `data` sheet of [PUBLISHED_EMISSIONS_FILE][(m).]

    Returns
    -------
    :
        Harmonised and infilled emissions, using gcages' naming convention
        and the scenario names used when the scenarios were submitted
        (i.e. the names in the output of
        [CMIP7ScenarioMIPWorkflowRunner][gcages.workflows.cmip7_scenariomip.CMIP7ScenarioMIPWorkflowRunner]).
    """
    prefix = "Climate Assessment|Harmonized and Infilled|"
    # Aggregates that are calculated after infilling,
    # so are not part of the infilled emissions
    aggregates = (
        "Emissions|CO2",
        "Emissions|GHG AR6GWP100",
        "Emissions|Kyoto GHG AR6GWP100",
    )
    submitted_scenario_names = {
        (m.model, m.published_scenario): m.scenario for m in CMIP7_SCENARIOMIP_MARKERS
    }

    res = raw[raw["variable"].str.startswith(f"{prefix}Emissions|")].copy()
    res["variable"] = res["variable"].str.removeprefix(prefix)
    res = res[~res["variable"].isin(aggregates)]
    res["scenario"] = [
        submitted_scenario_names[(m, s)] for m, s in zip(res["model"], res["scenario"])
    ]
    # The published results spell HFC43-10's unit differently
    res["unit"] = res["unit"].replace({"kt HFC4310mee/yr": "kt HFC4310/yr"})
    res = convert_cmip7_scenariomip_names_to_gcages(
        res.set_index(["model", "scenario", "region", "variable", "unit"])
    )
    res.columns = res.columns.astype(int)
    res.columns.name = "year"
    # Years before the harmonisation year are not included in the results
    res = res.dropna(how="all", axis="columns")

    return res


def load_published_emissions(downloader: Downloader | None = None) -> pd.DataFrame:
    """
    Load the published harmonised and infilled emissions for the marker scenarios

    Parameters
    ----------
    downloader
        Downloader to use to get the data from Zenodo

        If not supplied, we use [PoochDownloader][gcages.workflows.cmip7_scenariomip.inputs.PoochDownloader]
        with its default cache directory.

    Returns
    -------
    :
        Harmonised and infilled emissions,
        converted with [convert_published_emissions_to_gcages][(m).]
    """  # noqa: E501
    try:
        import openpyxl  # type: ignore # openpyxl has no type hints # noqa: F401, PLC0415
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "load_published_emissions", requirement="openpyxl"
        ) from exc

    if downloader is None:
        downloader = PoochDownloader()

    raw = pd.read_excel(
        get_file(PUBLISHED_EMISSIONS_FILE, downloader=downloader), sheet_name="data"
    )

    return convert_published_emissions_to_gcages(raw)


@define
class PublishedClimateAssessment:
    """
    Published climate assessment of the marker scenarios

    Converted to the format used in
    [PostProcessingResult][gcages.post_processing.PostProcessingResult].
    """

    timeseries_quantile: pd.DataFrame
    """Quantiles of assessed GSAT and effective radiative forcing over time"""

    metadata_quantile: pd.Series[float]
    """Quantiles of peak warming and warming in 2100"""

    metadata_exceedance_probabilities: pd.Series[float]
    """Probability of exceeding different warming levels"""


PUBLISHED_PERCENTILES = {
    "33rd Percentile": 0.33,
    "Median": 0.5,
    "67th Percentile": 0.67,
}
"""Map from the names of the published percentiles to quantiles"""

PUBLISHED_UNITS = {"°C": "K", "W/m^2": "W/m^2"}
"""Map from the published units to the units we use"""

PUBLISHED_TIMESERIES_PATTERN = re.compile(
    r"^Climate Assessment\|(?P<variable>.+)\|(?P<percentile>[^|]+) "
    r"\[(?P<climate_model>[^\]]+)\]$"
)
"""Pattern of the names of the published climate assessment timeseries"""

PUBLISHED_METADATA_QUANTILE_PATTERN = re.compile(
    r"^Climate Assessment\|(?P<metric>Peak Warming|Warming in 2100)\|"
    r"(?P<percentile>[^|]+) \[(?P<climate_model>[^,]+), (?P<unit>[^\]]+)\]$"
)
"""Pattern of the names of the published peak and 2100 warming metadata"""

PUBLISHED_METADATA_EXCEEDANCE_PATTERN = re.compile(
    r"^Climate Assessment\|Exceedance Probability (?P<threshold>[0-9.]+)"
    r"(?P<threshold_unit>[^ ]+) \[(?P<climate_model>[^,]+), (?P<unit>[^\]]+)\]$"
)
"""Pattern of the names of the published exceedance probability metadata"""

ASSESSED_GSAT_VARIABLE = "Surface Temperature (GSAT)"
"""Name of the assessed GSAT variable"""


def get_submitted_scenario_names(
    model: pd.Series[str], published_scenario: pd.Series[str]
) -> list[str]:
    """
    Get the submitted scenario names from the published scenario names
    """
    submitted_scenario_names = {
        (m.model, m.published_scenario): m.scenario for m in CMIP7_SCENARIOMIP_MARKERS
    }

    return [submitted_scenario_names[(m, s)] for m, s in zip(model, published_scenario)]


def convert_published_climate_assessment_to_gcages(
    data: pd.DataFrame, meta: pd.DataFrame
) -> PublishedClimateAssessment:
    """
    Convert the published climate assessment to the format we use

    Parameters
    ----------
    data
        Data as loaded from the `data` sheet of [PUBLISHED_EMISSIONS_FILE][(m).]

    meta
        Data as loaded from the `meta` sheet of [PUBLISHED_EMISSIONS_FILE][(m).]

    Returns
    -------
    :
        Published climate assessment,
        using the format used in
        [PostProcessingResult][gcages.post_processing.PostProcessingResult]
        and the scenario names used when the scenarios were submitted.
    """
    # Timeseries
    timeseries = data[
        data["variable"].str.startswith("Climate Assessment|")
        & ~data["variable"].str.startswith("Climate Assessment|Harmonized and Infilled")
    ].copy()
    parsed = timeseries["variable"].str.extract(PUBLISHED_TIMESERIES_PATTERN.pattern)
    if parsed.isnull().any().any():
        msg = f"Could not parse {timeseries['variable'][parsed.isnull().any(axis=1)]}"
        raise ValueError(msg)

    timeseries["variable"] = parsed["variable"]
    timeseries["climate_model"] = parsed["climate_model"]
    timeseries["quantile"] = parsed["percentile"].map(PUBLISHED_PERCENTILES)
    timeseries["unit"] = timeseries["unit"].map(PUBLISHED_UNITS)
    timeseries["scenario"] = get_submitted_scenario_names(
        timeseries["model"], timeseries["scenario"]
    )
    timeseries_quantile = timeseries.set_index(
        ["climate_model", "model", "region", "scenario", "unit", "variable", "quantile"]
    )
    timeseries_quantile.columns = timeseries_quantile.columns.astype(int)
    timeseries_quantile = timeseries_quantile.dropna(how="all", axis="columns")

    # Metadata
    meta = meta.copy()
    meta["scenario"] = get_submitted_scenario_names(meta["model"], meta["scenario"])
    meta = meta.set_index(["model", "scenario"])

    quantile_l = []
    exceedance_l = []
    for column in meta.columns:
        match_quantile = PUBLISHED_METADATA_QUANTILE_PATTERN.match(column)
        match_exceedance = PUBLISHED_METADATA_EXCEEDANCE_PATTERN.match(column)
        if match_quantile is not None:
            metric = "max" if match_quantile["metric"] == "Peak Warming" else 2100
            quantile_l.append(
                meta[column]
                .astype(float)
                .to_frame("value")
                .assign(
                    climate_model=match_quantile["climate_model"],
                    region="World",
                    unit=PUBLISHED_UNITS[match_quantile["unit"]],
                    variable=ASSESSED_GSAT_VARIABLE,
                    metric=metric,
                    quantile=PUBLISHED_PERCENTILES[match_quantile["percentile"]],
                )
            )

        elif match_exceedance is not None:
            exceedance_l.append(
                meta[column]
                .astype(float)
                .to_frame("value")
                .assign(
                    climate_model=match_exceedance["climate_model"],
                    region="World",
                    unit=match_exceedance["unit"],
                    variable="Exceedance probability",
                    threshold=float(match_exceedance["threshold"]),
                    threshold_unit=PUBLISHED_UNITS[match_exceedance["threshold_unit"]],
                )
            )

    metadata_quantile = (
        pd.concat(quantile_l)
        .reset_index()
        .set_index(
            [
                "climate_model",
                "model",
                "region",
                "scenario",
                "unit",
                "variable",
                "metric",
                "quantile",
            ]
        )["value"]
    )
    metadata_exceedance_probabilities = (
        pd.concat(exceedance_l)
        .reset_index()
        .set_index(
            [
                "climate_model",
                "model",
                "region",
                "scenario",
                "unit",
                "variable",
                "threshold",
                "threshold_unit",
            ]
        )["value"]
    )

    return PublishedClimateAssessment(
        timeseries_quantile=timeseries_quantile,
        metadata_quantile=metadata_quantile,
        metadata_exceedance_probabilities=metadata_exceedance_probabilities,
    )


def load_published_climate_assessment(
    downloader: Downloader | None = None,
) -> PublishedClimateAssessment:
    """
    Load the published climate assessment of the marker scenarios

    Parameters
    ----------
    downloader
        Downloader to use to get the data from Zenodo

        If not supplied, we use [PoochDownloader][gcages.workflows.cmip7_scenariomip.inputs.PoochDownloader]
        with its default cache directory.

    Returns
    -------
    :
        Published climate assessment,
        converted with [convert_published_climate_assessment_to_gcages][(m).]
    """  # noqa: E501
    try:
        import openpyxl  # noqa: F401, PLC0415
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "load_published_climate_assessment", requirement="openpyxl"
        ) from exc

    if downloader is None:
        downloader = PoochDownloader()

    fp = get_file(PUBLISHED_EMISSIONS_FILE, downloader=downloader)

    return convert_published_climate_assessment_to_gcages(
        data=pd.read_excel(fp, sheet_name="data"),
        meta=pd.read_excel(fp, sheet_name="meta"),
    )


def check_against_published_emissions(
    infilled: pd.DataFrame,
    published: pd.DataFrame,
    marker: CMIP7ScenarioMIPMarker,
) -> str | None:
    """
    Check infilled emissions against the published infilled emissions

    Parameters
    ----------
    infilled
        Infilled emissions (output of the workflow)

    published
        Published infilled emissions
        (from [load_published_emissions][(m).])

    marker
        Marker to check

    Returns
    -------
    :
        `None` if the infilled emissions match, otherwise a description of the problem
    """

    def select_marker(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[
            (df.index.get_level_values("model") == marker.model)
            & (df.index.get_level_values("scenario") == marker.scenario)
        ]

    exp = select_marker(published)
    res = select_marker(infilled)
    try:
        assert_frame_equal(res.loc[:, exp.columns].reorder_levels(exp.index.names), exp)  # type: ignore # pandas-stubs confused
    except AssertionError as exc:
        return str(exc)

    return None
