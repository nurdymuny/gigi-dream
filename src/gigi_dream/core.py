"""Top-level public API for gigi-dream.

The single entry point users typically need is :func:`dream`. For lower-level
access, see :mod:`gigi_dream.algorithm` and :mod:`gigi_dream.backends`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .algorithm import DreamResult
from .backends import LocalBackend, _BaseBackend


def dream(
    records: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    n_samples: int = 100,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    backend: Optional[_BaseBackend] = None,
) -> DreamResult:
    """Generate synthetic records that match the structure of the input.

    The high-level entry point. Pass a list of dicts (your real records) and
    get back ``n_samples`` synthetic dicts with the same columns and a
    distribution that matches the input — wider at higher temperature, tighter
    at lower temperature.

    Parameters
    ----------
    records : sequence of dicts, optional
        Real records to learn the distribution from. Required for
        :class:`LocalBackend`; :class:`GigiBackend` ignores this argument
        (reads from its configured bundle instead).
    n_samples : int, default 100
        Number of synthetic records to produce.
    temperature : float, default 1.0
        Sampling temperature. ``T = 1.0`` matches the empirical distribution.
        ``T > 1.0`` is DREAM mode — wider, more novel records. ``T < 1.0``
        produces tighter-than-original samples.
    seed : int, optional
        Random seed for reproducibility.
    backend : backend instance, optional
        Which backend to use. Defaults to :class:`LocalBackend`.

    Returns
    -------
    DreamResult
        The synthetic records and metadata.

    Example
    -------
    Basic usage::

        from gigi_dream import dream

        real = [
            {"age": 30, "country": "US", "salary": 75000},
            {"age": 45, "country": "CA", "salary": 95000},
            {"age": 28, "country": "US", "salary": 68000},
            ...
        ]
        result = dream(real, n_samples=1000, temperature=1.0, seed=42)
        for r in result.records[:3]:
            print(r)

    Higher temperature for more diverse synthetic data::

        diverse = dream(real, n_samples=1000, temperature=3.0)

    Using a remote GIGI instance::

        from gigi_dream import GigiBackend, dream
        backend = GigiBackend(
            url="http://localhost:3142",
            api_key="dev-local",
            bundle="customers",
            fields=["age", "salary"],
        )
        result = dream(n_samples=1000, backend=backend)
    """
    if backend is None:
        backend = LocalBackend()

    return backend.dream(
        records,
        n_samples=n_samples,
        temperature=temperature,
        seed=seed,
    )
