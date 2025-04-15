# MAGICC - A Simple Climate Model

The 'Model for the Assessment of Greenhouse Gas Induced Climate Change' (MAGICC), is a widely used simple
climate-carbon cycle model. It has informed recent IPCC assessments and continues to be used by several Integrated
Assessment Modeling groups.

If you make use of MAGICC7, please cite the following papers which describe MAGICC and its recent updates:

Meinshausen, M., Raper, S. C. B., and Wigley, T. M. L.: Emulating coupled atmosphere-ocean and carbon cycle models with
a simpler model, MAGICC6 – Part 1: Model description and calibration, Atmos. Chem. Phys., 11, 1417–1456,
https://doi.org/10.5194/acp-11-1417-2011, 2011.

Meinshausen, M., Nicholls, Z. R. J., Lewis, J., Gidden, M. J., Vogel, E., Freund, M., Beyerle, U., Gessner, C., Nauels,
A., Bauer, N., Canadell, J. G., Daniel, J. S., John, A., Krummel, P. B., Luderer, G., Meinshausen, N., Montzka, S. A.,
Rayner, P. J., Reimann, S., Smith, S. J., van den Berg, M., Velders, G. J. M., Vollmer, M. K., and Wang, R. H. J.:
The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500, Geosci. Model Dev.,
13, 3571–3605, https://doi.org/10.5194/gmd-13-3571-2020, 2020.

Copyright (c) MAGICC contributors

## License

See `LICENSE`. The license is issued by MAGICC7 Plus IP Co LTD (Australian Business Number 79 650 622 085), which has
contributer license agreements with all contributors and hence has the right to issue a license for MAGICC.

### Getting started

This bundle contains portable, prebuilt binaries for Windows, Linux and macOS:

| Platform        | Binary Location           |
|-----------------|---------------------------|
| Linux x64       | `bin/magicc`              |
| Windows x64     | `bin/magicc.exe`          |
| macOS Apple M1  | `bin/magicc-darwin-arm64` |
| macOS Intel x86 | `bin/magicc-darwin-amd64` |

When running MAGICC on newer versions of macOS, a prompt may occur stating that MAGICC cannot be run because
the “developer cannot be verified”. If this happens, close the prompt and go to
Settings > Security & Privacy > General. There will be an option to “Allow Anyway” which should be clicked.
The next time MAGICC is run, the prompt should include a “Run” option. Clicking this will allow the executable
to be run. These steps only need to be performed once and the binary should then run without a prompt appearing.

MAGICC expects a `run` directory which contains the base configuration files, namely:
* `MAGCFG_DEFAULTALL.CFG` - Full list of default parameters. This namelist is loaded first. The first thing you will likely want to do is open up `MAGCFG_DEFAULTALL.CFG` and set e.g. `OUT_TEMPERATURE=1` so that running MAGICC will produce output (we make no assumptions about what you want to output by default). Alternately, you can use the `OUT_DYNAMIC_VARS` flag e.g. `OUT_DYNAMIC_VARS='DAT_SURFACE_TEMP', 'DAT_TOTAL_INCLVOLCANIC_ERF'`.
* `MAGCFG_USER.CFG` - Definition of the parameter set for a single MAGICC run. Parameter values override the defaults
    specified in `MAGCFG_DEFAULTALL.CFG`. This configuration file may load input files which could override the parameters
    in `MAGCFG_USER.CFG`.
* `MAGCFG_NMLYEARS.CFG` - Defines the simulation period.

By default, the run directory will be located in `../run` with respect to the directory from which the binary is called.
The location of the run directory can be overridden by specifying a different path using the `-r` or `--run-dir` arguments.
For example, the following will source the run configuration from the `custom_run` directory:
```
./magicc -r custom_run_dir
```

MAGICC may load other input (.IN, .MON), scenario (.SCEN) or tuning (.CFG) files depending
on the parameters in `MAGCFG_USER.CFG`. By default, these input files are located in the `run` directory. An alternative
input directory can be specified using the `PATHNAME_INFILES` parameter. When loading files, MAGICC will look in the current
working directory first, then the run directory and finally the input directory. If a particular file is not found in
any of these directories than an error is raised.

MAGICC output is written to `../out` (again relative to the binary) with respect to the directory from which the binary is
called by default. This can be overridden by specifying the `PATHNAME_OUTFILES` parameter. Depending on the flag `OUT_ASCII_BINARY`,
either ASCII or BINARY output is produced. Output is only produced for the 'modules' (for want of a better word) for
which the `OUT_XXXX` parameter is set to 1 in the namelist, e.g. `OUT_SEALEVEL  = 1` tells MAGICC to output all results
relevant to sea level rise. A custom set of output variables can be specified using the `OUT_DYNAMIC_VARS` parameter.

[pymagicc](https://github.com/openscm/pymagicc), a Python wrapper for running MAGICC, has been developed to simplify the
above steps. It is the recommended method for running MAGICC7.

### Contributing

Thanks for your interest in contributing! There are many ways to contribute to MAGICC. Contact Malte Meinshausen
<malte.meinshausen@unimelb.edu.au> or Zebedee Nicholls <zebedee.nicholls@unimelb.edu.au> for more information on how
you may be able to contribute. Note that you will need to enter into a contributor license agreement before any
contributions can be accepted.
