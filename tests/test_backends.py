"""Tests for the backend selection surface."""

from __future__ import annotations

import pytest

from gigi_dream import DreamResult, GigiBackend, LocalBackend, dream


def make_records():
    return [
        {"age": 30, "country": "US", "salary": 75000},
        {"age": 45, "country": "CA", "salary": 95000},
        {"age": 28, "country": "US", "salary": 68000},
        {"age": 51, "country": "UK", "salary": 110000},
    ] * 25  # 100 records


def test_default_backend_is_local():
    result = dream(make_records(), n_samples=10)
    assert result.backend == "local"


def test_local_backend_explicitly():
    backend = LocalBackend()
    result = dream(make_records(), n_samples=10, backend=backend)
    assert isinstance(result, DreamResult)
    assert result.backend == "local"


def test_local_backend_passes_through_temperature():
    result = dream(make_records(), n_samples=10, temperature=2.5)
    assert result.temperature == 2.5


def test_gigi_backend_constructor():
    """GigiBackend can be constructed without connecting."""
    backend = GigiBackend(
        url="http://localhost:3142",
        api_key="dev-local",
        bundle="customers",
        fields=["age", "salary"],
    )
    assert backend.name == "gigi"
    assert backend.bundle == "customers"
    assert backend.fields == ["age", "salary"]


def test_gigi_backend_raises_on_unreachable():
    """GigiBackend raises RuntimeError when GIGI is unreachable."""
    backend = GigiBackend(
        url="http://localhost:1",  # unreachable
        bundle="x",
        fields=["a"],
        timeout=1.0,
    )
    with pytest.raises(RuntimeError, match="GigiBackend"):
        backend.dream(None, n_samples=5)
