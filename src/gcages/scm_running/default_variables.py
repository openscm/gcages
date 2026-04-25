"""
Default variables to use when running simple climate models (SCMs)
"""

DEFAULT_OUTPUT_VARIABLES: tuple[str, ...] = (
    # GSAT
    "Surface Air Temperature Change",
    # TODO: delete as not available from all SCMs
    # GMST
    "Surface Air Ocean Blended Temperature Change",
    # ERFs
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Anthropogenic",
    "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Effective Radiative Forcing|Ozone",
    # Heat uptake
    "Heat Uptake",
    # Atmospheric concentrations
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    # TODO: delete as not available from all SCMs
    # Carbon cycle
    "Net Atmosphere to Land Flux|CO2",
    "Net Atmosphere to Ocean Flux|CO2",
    # TODO: delete as not available from all SCMs
    # Permafrost
    "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
)
"""
Default output variables

Note that it can be a bit of work
to get these variables to actually appear in the output,
depending on which simple climate model you're using.
"""
