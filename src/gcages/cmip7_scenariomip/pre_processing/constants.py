"""
Constants used throughout the pre-processing

Throughout this module, we aim to follow the following conventions:

- "World" means global total
- "Required" means that the thing of interest is compulsory in the input data
- "INPUT" means that the values are in the input data naming convention
"""

from __future__ import annotations

DOMESTIC_AVIATION_SECTOR_INPUT: str = "Energy|Demand|Transportation|Domestic Aviation"
"""
Name for the domestic aviation sector (input naming convention)
"""

INTERNATIONAL_AVIATION_SECTOR_INPUT: str = (
    "Energy|Demand|Bunkers|International Aviation"
)
"""
Name for the international aviation sector (input naming convention)
"""

AVIATION_SECTOR_REAGGREGATED: str = "Aviation"
"""
Sector name used for the re-aggreated aviation sector
"""

AVIATION_SECTOR_GRIDDING: str = "Aircraft"
"""
Sector name used for the aviation sector when gridding
"""

TRANSPORTATION_SECTOR_INPUT: str = "Energy|Demand|Transportation"
"""
Name for the transportation sector as a whole (input naming convention)
"""

TRANSPORTATION_SECTOR_REAGGREGATED: str = "Transportation excluding aviation"
"""
Sector name used for the re-aggreated transport sector
"""

TRANSPORTATION_SECTOR_GRIDDING: str = "Transportation Sector"
"""
Sector name used for the transport sector when gridding
"""

REQUIRED_AGRICULTURE_SECTOR_GRIDDING_COMPONENTS_INPUT: tuple[str, ...] = (
    "AFOLU|Agriculture",
)
"""
Optional sectors (input naming convention) that make up the gridding agriculture sector
"""

OPTIONAL_AGRICULTURE_SECTOR_GRIDDING_COMPONENTS_INPUT: tuple[str, ...] = (
    "AFOLU|Land|Land Use and Land-Use Change",
    "AFOLU|Land|Harvested Wood Products",
    "AFOLU|Land|Other",
    "AFOLU|Land|Wetlands",
)
"""
Required sectors (input naming convention) that make up the gridding agriculture sector
"""

AGRICULTURE_SECTOR_REAGGREGATED: str = "Agriculture"
"""
Sector name used for the re-aggreated agriculture sector
"""

AGRICULTURE_SECTOR_GRIDDING: str = "Agriculture"
"""
Sector name used for the transport sector when gridding
"""

REQUIRED_INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT: tuple[str, ...] = (
    "Energy|Demand|Industry",
    "Industrial Processes",
)
"""
Optional sectors (input naming convention) that make up the gridding industrial sector
"""

OPTIONAL_INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT: tuple[str, ...] = (
    "Energy|Demand|Other Sector",
    "Other",
)
"""
Required sectors (input naming convention) that make up the gridding industrial sector
"""

INDUSTRIAL_SECTOR_REAGGREGATED: str = "Industrial"
"""
Sector name used for the re-aggreated industrial sector
"""

INDUSTRIAL_SECTOR_GRIDDING: str = "Industrial Sector"
"""
Sector name used for the industrial sector when gridding
"""

REQUIRED_GRIDDING_SECTORS_WORLD_INPUT: tuple[str, ...] = (
    INTERNATIONAL_AVIATION_SECTOR_INPUT,
    "Energy|Demand|Bunkers|International Shipping",
)
"""
Sectors (input naming) required in the input at the world level for gridding
"""

REQUIRED_GRIDDING_SECTORS_MODEL_REGION_INPUT: tuple[str, ...] = (
    "Energy|Supply",
    *REQUIRED_INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT,
    "Energy|Demand|Residential and Commercial and AFOFI",
    "Product Use",
    # Technically, domestic aviation could be reported just
    # at the world level and it would be fine.
    # In practice, no-one does that and the logic is much simpler
    # if we assume it has to be reported regionally
    # (because then domestic aviation and transport are on the same regional 'grid')
    # so do that for now.
    DOMESTIC_AVIATION_SECTOR_INPUT,
    TRANSPORTATION_SECTOR_INPUT,
    "Waste",
    # Compulsory agriculture parts
    "AFOLU|Agriculture",
    "AFOLU|Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
)
"""
Sectors (input naming) required in the input at the model region level for gridding
"""

REQUIRED_GRIDDING_SECTORS_MODEL_REGION_INPUT_CO2_EXCEPTIONS: tuple[str, ...] = (
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning",
)
"""
Sectors (input naming) at the model region level that aren't required for CO2

These are burning sectors because burning should come from the model internally
to avoid double counting.
In general, these exceptions are super messy
but we think this is the best available interpretation of everything.
"""

OPTIONAL_GRIDDING_SECTORS_MODEL_REGION_INPUT: tuple[str, ...] = (
    *OPTIONAL_INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT,
    "Other Capture and Removal",
    "AFOLU|Land|Land Use and Land-Use Change",
    "AFOLU|Land|Harvested Wood Products",
    "AFOLU|Land|Other",
    "AFOLU|Land|Wetlands",
    "AFOLU|Land|Fires|Peat Burning",
)
"""
Sectors (input naming) that are optional at the model region level for gridding
"""

REQUIRED_GRIDDING_SECTORS_WORLD_GRIDDING: tuple[str, ...] = (
    AVIATION_SECTOR_GRIDDING,
    "International Shipping",
)
"""
Sectors (gridding naming) required for gridding at the world level
"""

REQUIRED_GRIDDING_SECTORS_MODEL_REGION_GRIDDING: tuple[str, ...] = (
    "Energy Sector",
    INDUSTRIAL_SECTOR_GRIDDING,
    "Residential Commercial Other",
    "Solvents Production and Application",
    TRANSPORTATION_SECTOR_GRIDDING,
    "Waste",
    AGRICULTURE_SECTOR_GRIDDING,
    "Agricultural Waste Burning",
    "Forest Burning",
    "Grassland Burning",
    "Peat Burning",
)
"""
Sectors (gridding naming) required for gridding at the model region level
"""

REQUIRED_GRIDDING_SPECIES_INPUT: tuple[str, ...] = (
    "CO2",
    "CH4",
    "N2O",
    "BC",
    "CO",
    "NH3",
    "OC",
    "NOx",
    "Sulfur",
    "VOC",
)
"""
Required species for gridding (in the input naming convention)
"""

REQUIRED_WORLD_VARIABLES_INPUT: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_INPUT
    for sector in REQUIRED_GRIDDING_SECTORS_WORLD_INPUT
)
"""
Variables required at the world level (input naming convention)
"""

OPTIONAL_GRIDDING_SECTORS_WORLD_INPUT: tuple[str, ...] = ()
"""
Sectors (input naming) that are optional at the world level for gridding
"""

OPTIONAL_WORLD_VARIABLES_INPUT: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_INPUT
    for sector in OPTIONAL_GRIDDING_SECTORS_WORLD_INPUT
)
"""
Optional variables at the world level (input naming convention)
"""

ALL_WORLD_VARIABLES_INPUT: tuple[str, ...] = tuple(
    (*REQUIRED_WORLD_VARIABLES_INPUT, *OPTIONAL_WORLD_VARIABLES_INPUT)
)
"""
All variables considered at the world level (input naming convention)
"""

INDEPENDENT_WORLD_VARIABLES_INPUT: tuple[str, ...] = ALL_WORLD_VARIABLES_INPUT
"""
Variables that are independent at the world level (input naming convention)
"""

REQUIRED_MODEL_REGION_VARIABLES_INPUT: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_INPUT
    for sector in REQUIRED_GRIDDING_SECTORS_MODEL_REGION_INPUT
    if not (
        species == "CO2"
        and sector in REQUIRED_GRIDDING_SECTORS_MODEL_REGION_INPUT_CO2_EXCEPTIONS
    )
)

OPTIONAL_MODEL_REGION_VARIABLES_INPUT: tuple[str, ...] = tuple(
    (
        *(
            f"Emissions|{species}|{sector}"
            for species in REQUIRED_GRIDDING_SPECIES_INPUT
            for sector in OPTIONAL_GRIDDING_SECTORS_MODEL_REGION_INPUT
        ),
        *(
            f"Emissions|CO2|{sector}"
            for sector in REQUIRED_GRIDDING_SECTORS_MODEL_REGION_INPUT_CO2_EXCEPTIONS
        ),
    )
)
"""
Optional variables at the model region level (input naming convention)
"""

ALL_MODEL_REGION_VARIABLES_INPUT: tuple[str, ...] = tuple(
    (*REQUIRED_MODEL_REGION_VARIABLES_INPUT, *OPTIONAL_MODEL_REGION_VARIABLES_INPUT)
)
"""
All variables considered at the model region level (input naming convention)
"""

INDEPENDENT_MODEL_REGION_VARIABLES_INPUT: tuple[str, ...] = tuple(
    v
    for v in ALL_MODEL_REGION_VARIABLES_INPUT
    # Avoid double counting
    if DOMESTIC_AVIATION_SECTOR_INPUT not in v
)
"""
Variables that are independent at the model region level (input naming convention)
"""

REQUIRED_WORLD_VARIABLES_GRIDDING: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_INPUT
    for sector in REQUIRED_GRIDDING_SECTORS_WORLD_GRIDDING
)
"""
Variables required at the world level (gridding naming convention)
"""

REQUIRED_MODEL_REGION_VARIABLES_GRIDDING: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_INPUT
    for sector in REQUIRED_GRIDDING_SECTORS_MODEL_REGION_GRIDDING
)
"""
Variables required at the model region level (gridding naming convention)
"""

REAGGREGATED_TO_GRIDDING_SECTOR_MAP_WORLD: dict[str, str] = {
    AVIATION_SECTOR_REAGGREGATED: AVIATION_SECTOR_GRIDDING,
    "Energy|Demand|Bunkers|International Shipping": "International Shipping",
}
"""
Map from re-aggreated variables to sectors used for gridding at the world level
"""

REAGGREGATED_TO_GRIDDING_SECTOR_MAP_MODEL_REGION: dict[str, str] = {
    "Energy|Supply": "Energy Sector",
    INDUSTRIAL_SECTOR_REAGGREGATED: INDUSTRIAL_SECTOR_GRIDDING,
    "Energy|Demand|Residential and Commercial and AFOFI": "Residential Commercial Other",  # noqa: E501
    "Product Use": "Solvents Production and Application",
    TRANSPORTATION_SECTOR_REAGGREGATED: TRANSPORTATION_SECTOR_GRIDDING,
    "Waste": "Waste",
    AGRICULTURE_SECTOR_REAGGREGATED: AGRICULTURE_SECTOR_GRIDDING,
    "AFOLU|Agricultural Waste Burning": "Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning": "Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning": "Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning": "Peat Burning",
}
"""
Map from re-aggreated variables to sectors used for gridding at the model region level
"""

CO2_FOSSIL_SECTORS_GRIDDING: tuple[str, ...] = (
    AVIATION_SECTOR_GRIDDING,
    "International Shipping",
    "Energy Sector",
    INDUSTRIAL_SECTOR_GRIDDING,
    "Residential Commercial Other",
    "Solvents Production and Application",
    TRANSPORTATION_SECTOR_GRIDDING,
    "Waste",
)
"""
Sectors that come from fossil CO2 reservoirs (gridding naming convention)

Not a perfect split with [CO2_FOSSIL_SECTORS_GRIDDING][(m).],
but the best we can do.
"""

CO2_BIOSPHERE_SECTORS_GRIDDING: tuple[str, ...] = (
    # Agriculture in biosphere because most of its emissions
    # are land carbon cycle (but not all, probably, in reality)
    AGRICULTURE_SECTOR_GRIDDING,
    "Agricultural Waste Burning",
    "Forest Burning",
    "Grassland Burning",
    "Peat Burning",
)
"""
Sectors that come from biospheric CO2 reservoirs (gridding naming convention)

Not a perfect split with [CO2_FOSSIL_SECTORS_GRIDDING][(m).],
but the best we can do.
"""
