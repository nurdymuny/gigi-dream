"""Backends for gigi-dream — local sampler and GIGI-backed sampler.

- :class:`LocalBackend` — pure-numpy Welford + diagonal-Gaussian sampler.
- :class:`GigiBackend` — calls a running GIGI instance's
  ``POST /v1/bundles/{name}/brain/dream`` endpoint for higher-fidelity
  sampling on anisotropic, correlated, or multivariate data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .algorithm import DreamResult, dream_local


class _BaseBackend:
    """Abstract base — backends must implement :meth:`dream`."""

    name: str = "base"

    def dream(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        n_samples: int = 100,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> DreamResult:
        raise NotImplementedError


class LocalBackend(_BaseBackend):
    """Pure-numpy synthetic data generation, no external services.

    The default. Diagonal-Gaussian fit per numeric column, empirical-frequency
    fit per categorical column.

    Example::

        from gigi_dream import LocalBackend, dream

        backend = LocalBackend()
        result = dream(real_records, n_samples=1000, backend=backend)
    """

    name = "local"

    def dream(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        n_samples: int = 100,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> DreamResult:
        """Generate synthetic records via local Welford fit.

        See :func:`gigi_dream.algorithm.dream_local` for full parameter docs.
        """
        return dream_local(
            records,
            n_samples=n_samples,
            temperature=temperature,
            seed=seed,
        )


@dataclass
class GigiBackend(_BaseBackend):
    """Synthetic data generation backed by a running GIGI instance.

    Calls ``POST /v1/bundles/{bundle}/brain/dream`` with the supplied fields.
    Requires the bundle to exist; this backend does not auto-create bundles
    in v0.1.

    Parameters
    ----------
    url : str
        GIGI server URL (e.g. ``"http://localhost:3142"``).
    bundle : str
        Bundle name to sample from.
    fields : list of str
        Field names to include in the sampled output.
    api_key : str, optional
        API key sent in the ``Authorization: Bearer`` header.
    timeout : float, default 30.0
        Request timeout in seconds.

    Example::

        from gigi_dream import GigiBackend, dream

        backend = GigiBackend(
            url="http://localhost:3142",
            api_key="dev-local",
            bundle="customers",
            fields=["age", "income", "city"],
        )
        result = dream(records=None, n_samples=1000, backend=backend)
        #   ↑ records argument ignored — GigiBackend reads from the bundle directly
    """

    url: str
    bundle: str
    fields: List[str]
    api_key: Optional[str] = None
    timeout: float = 30.0
    fit_mode: str = "diagonal"  # "isotropic" or "diagonal"
    sigma_floor_epsilon: float = 1e-3

    name: str = "gigi"

    def dream(
        self,
        records: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        n_samples: int = 100,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> DreamResult:
        """Generate synthetic records by calling GIGI's /brain/dream endpoint.

        The ``records`` argument is ignored — GigiBackend reads from the
        configured bundle directly.
        """
        # Lazy import — package doesn't require requests for LocalBackend users
        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "GigiBackend requires the `requests` package. "
                "Install with: pip install gigi-dream[gigi]"
            ) from e

        endpoint = f"{self.url.rstrip('/')}/v1/bundles/{self.bundle}/brain/dream"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "fields": self.fields,
            "n_samples": n_samples,
            "temperature": temperature,
            "fit_mode": self.fit_mode,
            "sigma_floor_epsilon": self.sigma_floor_epsilon,
        }
        if seed is not None:
            payload["seed"] = seed

        try:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"GigiBackend request failed: {e}. "
                f"Falling back to LocalBackend is recommended if GIGI is unavailable."
            ) from e

        data = response.json()
        # GIGI returns "samples" as a list of vectors aligned to "fields".
        # Translate into the dict-per-record shape gigi-dream uses everywhere.
        samples = data.get("samples", [])
        synthetic = [
            {field: float(sample[i]) for i, field in enumerate(self.fields)}
            for sample in samples
        ]
        return DreamResult(
            records=synthetic,
            n_samples=len(synthetic),
            temperature=temperature,
            columns=[],  # GIGI reports per-field fit info separately; not unpacked in v0.1
            backend="gigi",
        )
