"""
Global Climate Assessment of Global Emission Scenarios.

Typically (but not always) produced by Integrated Assessment Models (IAMs).
"""

import importlib.metadata

__version__ = importlib.metadata.version("gcages")

import os

if os.environ.get("USE_BILLIARD_INSTEAD_OF_MULTIPROCESSING", "false") == "true":
    import sys
    import billiard

    sys.modules["multiprocessing"] = billiard

    import multiprocessing

    print(f"Using billiard {multiprocessing.__version__} instead of multiprocessing")
