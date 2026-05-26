"""gigi-dream — synthetic data generation via GIGI's DREAM brain primitive.

A small, focused tool that generates structurally faithful synthetic records
from any tabular file. Built on the same Welford + diagonal-Gaussian +
high-temperature Langevin approach as GIGI's ``/brain/dream`` endpoint.

Quick start::

    from gigi_dream import dream

    real = [
        {"age": 30, "country": "US", "salary": 75000},
        {"age": 45, "country": "CA", "salary": 95000},
        # ... more real records ...
    ]
    result = dream(real, n_samples=1000, temperature=2.0)
    print(result.records[0])
    # {"age": 38.7, "country": "US", "salary": 84210.3}

Or from the command line::

    gigi-dream customers.csv -n 1000 -T 2.0 -o test_customers.csv

See https://github.com/nurdymuny/gigi-dream for full documentation.
"""

from .algorithm import (
    ColumnFit,
    DreamResult,
    dream_local,
    fit_columns,
)
from .backends import GigiBackend, LocalBackend
from .core import dream

__version__ = "0.1.0"
__all__ = [
    "ColumnFit",
    "DreamResult",
    "GigiBackend",
    "LocalBackend",
    "dream",
    "dream_local",
    "fit_columns",
]
