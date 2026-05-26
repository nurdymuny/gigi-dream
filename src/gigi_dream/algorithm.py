"""Pure-numpy synthetic data generation — the Welford-streaming heart of gigi-dream.

Implements the local fallback for GIGI's DREAM brain primitive. DREAM is the
high-temperature mode of SAMPLE: novel-but-plausible draws from the bundle's
empirical density. For tabular data, the density is approximated by an
independent (diagonal) Gaussian per numeric column plus an empirical-frequency
sampler per categorical column.

For higher-fidelity sampling on anisotropic or correlated data, use
``GigiBackend`` — it calls GIGI's ``/brain/dream`` endpoint which uses the
engine's full Kähler-aware Welford fit including L13.3 diagonal-Gaussian
support and L13.7 denominator floor for numerical stability.

The math (numeric columns):
    Per-column Welford accumulators:
        count_n  := number of observations
        mean_n   := sum(x_i) / count_n
        M2_n     := sum((x_i - mean_n)^2)
    Variance:
        var_n    := M2_n / count_n
    DREAM sample at temperature T:
        x_synth  := mean + sqrt(T) * sqrt(var_n) * Normal(0, 1)
    At T = 1, this is faithful SAMPLE. At T > 1, samples spread wider than
    the original data — that's the "REM-sleep" / novelty mode.

For categorical columns we sample from the empirical frequency distribution
of observed values. Temperature does not affect categoricals in v0 (a future
version could use temperature to flatten the distribution toward uniform).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# Sentinel that signals "this column is non-numeric and should be sampled
# from its empirical distribution."
_CATEGORICAL = object()


@dataclass
class ColumnFit:
    """Welford fit for a single column of tabular data.

    For numeric columns, holds mean and variance. For categorical columns,
    holds the value→frequency table.

    Attributes
    ----------
    name : str
        Column name.
    kind : str
        ``"numeric"`` or ``"categorical"``.
    mean : float
        Mean of numeric values. Undefined (0.0) for categorical columns.
    sigma : float
        Standard deviation (sqrt of population variance). Floored by
        ``sigma_floor`` to prevent zero-sigma columns from degenerating.
    n_observed : int
        Number of observations the fit was computed from.
    values : list, optional
        For categorical columns: distinct observed values in order seen.
    weights : list of float, optional
        For categorical columns: normalized frequency of each value in
        ``values`` (sums to 1.0).
    """

    name: str
    kind: str
    mean: float = 0.0
    sigma: float = 0.0
    n_observed: int = 0
    values: Optional[List[Any]] = None
    weights: Optional[List[float]] = None


@dataclass
class DreamResult:
    """Result of generating synthetic data from a fit.

    Attributes
    ----------
    records : list of dict
        The synthetic records, in order. Each is a ``{column_name: value}``
        dict matching the column structure of the input.
    n_samples : int
        Number of synthetic records produced.
    temperature : float
        Sampling temperature used. T = 1 is faithful; T > 1 is DREAM mode.
    columns : list of ColumnFit
        The per-column fits the sampler was drawn from. Useful for
        introspection — you can see exactly what distribution each column
        was modeled with.
    backend : str
        Which backend produced the result.
    """

    records: List[Dict[str, Any]] = field(default_factory=list)
    n_samples: int = 0
    temperature: float = 1.0
    columns: List[ColumnFit] = field(default_factory=list)
    backend: str = "local"


def _infer_kind(values: Sequence[Any]) -> str:
    """Decide whether a column is numeric or categorical based on its values.

    A column is numeric if every non-null value is parseable as a float.
    Otherwise categorical. Booleans count as categorical.
    """
    saw_any = False
    for v in values:
        if v is None:
            continue
        saw_any = True
        if isinstance(v, bool):
            return "categorical"
        if isinstance(v, (int, float, np.integer, np.floating)):
            continue
        # Try string-to-float
        if isinstance(v, str):
            try:
                float(v)
            except (ValueError, TypeError):
                return "categorical"
            continue
        return "categorical"
    return "numeric" if saw_any else "categorical"


def fit_columns(
    records: Sequence[Dict[str, Any]],
    *,
    column_names: Optional[Sequence[str]] = None,
    sigma_floor: float = 1e-3,
) -> List[ColumnFit]:
    """Fit a per-column model (numeric → Gaussian; categorical → frequencies).

    Parameters
    ----------
    records : sequence of dicts
        Input records. Each dict represents one row; keys are column names.
    column_names : sequence of str, optional
        Column order. If not given, inferred from the first record's keys.
    sigma_floor : float, default 1e-3
        Lower bound on per-column sigma. Prevents zero-variance columns
        (e.g., a column where all values are identical) from collapsing
        to zero spread under DREAM sampling.

    Returns
    -------
    list of ColumnFit
        One fit per column in the input order.
    """
    if not records:
        return []
    if column_names is None:
        column_names = list(records[0].keys())

    fits: List[ColumnFit] = []
    for col in column_names:
        raw_values = [r.get(col) for r in records]
        kind = _infer_kind(raw_values)

        if kind == "numeric":
            numeric_values = []
            for v in raw_values:
                if v is None:
                    continue
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    continue
            if not numeric_values:
                fits.append(ColumnFit(name=col, kind="numeric"))
                continue
            arr = np.asarray(numeric_values, dtype=float)
            mean = float(arr.mean())
            sigma = float(arr.std(ddof=0))
            sigma = max(sigma, sigma_floor)
            fits.append(
                ColumnFit(
                    name=col,
                    kind="numeric",
                    mean=mean,
                    sigma=sigma,
                    n_observed=len(numeric_values),
                )
            )
        else:
            # Categorical: empirical frequency distribution
            counts: Dict[Any, int] = {}
            for v in raw_values:
                if v is None:
                    continue
                counts[v] = counts.get(v, 0) + 1
            if not counts:
                fits.append(ColumnFit(name=col, kind="categorical", values=[], weights=[]))
                continue
            values = list(counts.keys())
            total = sum(counts.values())
            weights = [counts[v] / total for v in values]
            fits.append(
                ColumnFit(
                    name=col,
                    kind="categorical",
                    n_observed=total,
                    values=values,
                    weights=weights,
                )
            )

    return fits


def dream_local(
    records: Sequence[Dict[str, Any]],
    *,
    n_samples: int = 100,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    sigma_floor: float = 1e-3,
    column_names: Optional[Sequence[str]] = None,
) -> DreamResult:
    """Generate synthetic records from a tabular input using local Welford fit.

    Pure-numpy / pure-Python; no external services required. Algorithm parity
    with GIGI's ``/brain/dream`` for the diagonal-Gaussian case.

    Parameters
    ----------
    records : sequence of dicts
        Real records to learn the distribution from.
    n_samples : int, default 100
        Number of synthetic records to produce.
    temperature : float, default 1.0
        Sampling temperature.
        - ``T = 1.0``: faithful samples (matches the empirical distribution)
        - ``T > 1.0``: DREAM mode — wider spread, more novel records
        - ``T < 1.0``: tighter than original (sharper modes)
    seed : int, optional
        Random seed for reproducibility.
    sigma_floor : float, default 1e-3
        Lower bound on per-numeric-column sigma. See :func:`fit_columns`.
    column_names : sequence of str, optional
        Column order in output records. Defaults to the order in the first
        input record.

    Returns
    -------
    DreamResult
        Synthetic records, the fit they came from, and metadata.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if not records or n_samples == 0:
        return DreamResult(
            records=[],
            n_samples=0,
            temperature=temperature,
            columns=fit_columns(records, column_names=column_names, sigma_floor=sigma_floor),
            backend="local",
        )

    fits = fit_columns(records, column_names=column_names, sigma_floor=sigma_floor)
    rng = np.random.default_rng(seed)
    temp_scale = math.sqrt(max(temperature, 0.0))

    synthetic: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        rec: Dict[str, Any] = {}
        for f in fits:
            if f.kind == "numeric":
                rec[f.name] = float(f.mean + temp_scale * f.sigma * rng.standard_normal())
            else:
                if f.values:
                    idx = int(rng.choice(len(f.values), p=f.weights))
                    rec[f.name] = f.values[idx]
                else:
                    rec[f.name] = None
        synthetic.append(rec)

    return DreamResult(
        records=synthetic,
        n_samples=n_samples,
        temperature=temperature,
        columns=fits,
        backend="local",
    )
