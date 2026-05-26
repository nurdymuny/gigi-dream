"""Tests for the local diagonal-Gaussian sampler."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gigi_dream.algorithm import (
    ColumnFit,
    DreamResult,
    dream_local,
    fit_columns,
)


# ─── Synthetic input data ────────────────────────────────────────────────────


def make_customers(n=200, seed=0):
    """Generate n synthetic 'customer' records with mixed column types."""
    rng = np.random.default_rng(seed)
    countries = rng.choice(["US", "CA", "UK", "DE"], size=n, p=[0.5, 0.2, 0.2, 0.1])
    ages = rng.normal(35, 8, n).clip(18, 80)
    incomes = rng.normal(75000, 25000, n).clip(20000, 300000)
    return [
        {"age": float(ages[i]), "country": str(countries[i]), "income": float(incomes[i])}
        for i in range(n)
    ]


# ─── fit_columns ─────────────────────────────────────────────────────────────


def test_fit_columns_identifies_numeric_and_categorical():
    """fit_columns correctly classifies numeric vs categorical columns."""
    records = make_customers(50)
    fits = fit_columns(records)
    by_name = {f.name: f for f in fits}
    assert by_name["age"].kind == "numeric"
    assert by_name["income"].kind == "numeric"
    assert by_name["country"].kind == "categorical"


def test_fit_numeric_mean_and_sigma_are_reasonable():
    """Numeric fit produces a sensible mean and sigma."""
    records = make_customers(500)
    fits = fit_columns(records)
    age_fit = next(f for f in fits if f.name == "age")
    assert 30 < age_fit.mean < 40
    assert 5 < age_fit.sigma < 12


def test_fit_categorical_weights_sum_to_one():
    """Categorical fit produces normalized weights."""
    records = make_customers(500)
    fits = fit_columns(records)
    country_fit = next(f for f in fits if f.name == "country")
    assert country_fit.values is not None
    assert country_fit.weights is not None
    assert math.isclose(sum(country_fit.weights), 1.0, abs_tol=1e-9)


def test_fit_empty_records_returns_empty():
    """Empty input → empty fit list."""
    assert fit_columns([]) == []


def test_fit_floors_zero_variance_columns():
    """A constant numeric column gets floored sigma, not literal zero."""
    records = [{"x": 5.0} for _ in range(20)]
    fits = fit_columns(records, sigma_floor=0.001)
    x_fit = fits[0]
    assert x_fit.kind == "numeric"
    assert x_fit.mean == 5.0
    assert x_fit.sigma >= 0.001  # floored


# ─── dream_local: shape ──────────────────────────────────────────────────────


def test_dream_returns_requested_number_of_samples():
    records = make_customers(100)
    result = dream_local(records, n_samples=37, seed=1)
    assert result.n_samples == 37
    assert len(result.records) == 37


def test_dream_returns_dream_result_type():
    records = make_customers(50)
    result = dream_local(records, n_samples=10)
    assert isinstance(result, DreamResult)
    assert result.backend == "local"


def test_dream_synthetic_records_have_same_columns_as_input():
    records = make_customers(100)
    result = dream_local(records, n_samples=20)
    for r in result.records:
        assert set(r.keys()) == {"age", "country", "income"}


def test_dream_zero_samples_returns_empty():
    records = make_customers(50)
    result = dream_local(records, n_samples=0)
    assert result.n_samples == 0
    assert result.records == []


def test_dream_empty_input_returns_empty():
    result = dream_local([], n_samples=10)
    assert result.n_samples == 0
    assert result.records == []


# ─── dream_local: statistical faithfulness ──────────────────────────────────


def test_dream_at_temperature_one_matches_input_mean():
    """At T=1, synthetic mean should be close to input mean."""
    records = make_customers(1000, seed=42)
    result = dream_local(records, n_samples=1000, temperature=1.0, seed=99)
    real_ages = [r["age"] for r in records]
    synth_ages = [r["age"] for r in result.records]
    real_mean = float(np.mean(real_ages))
    synth_mean = float(np.mean(synth_ages))
    # Within ~2 standard errors of the input mean
    assert abs(synth_mean - real_mean) < 2.0


def test_dream_higher_temperature_increases_spread():
    """Higher temperature produces wider synthetic distribution."""
    records = make_customers(500, seed=0)
    low_t = dream_local(records, n_samples=2000, temperature=1.0, seed=7)
    high_t = dream_local(records, n_samples=2000, temperature=4.0, seed=7)
    low_std = float(np.std([r["age"] for r in low_t.records]))
    high_std = float(np.std([r["age"] for r in high_t.records]))
    assert high_std > 1.5 * low_std


def test_dream_categorical_samples_from_input_values():
    """Synthetic categorical values are always values that appeared in the input."""
    records = make_customers(200, seed=3)
    result = dream_local(records, n_samples=500, seed=4)
    input_countries = {r["country"] for r in records}
    synth_countries = {r["country"] for r in result.records}
    # Every synthetic country must have appeared in the input
    assert synth_countries.issubset(input_countries)


def test_dream_seed_makes_output_reproducible():
    """Same seed → same synthetic records (deterministic)."""
    records = make_customers(100, seed=0)
    a = dream_local(records, n_samples=50, seed=123)
    b = dream_local(records, n_samples=50, seed=123)
    # Numeric fields should match to floating-point precision
    for ra, rb in zip(a.records, b.records):
        assert math.isclose(ra["age"], rb["age"], rel_tol=1e-12)
        assert math.isclose(ra["income"], rb["income"], rel_tol=1e-12)
        assert ra["country"] == rb["country"]


def test_dream_different_seeds_produce_different_records():
    """Different seeds → different (not identical) outputs."""
    records = make_customers(100)
    a = dream_local(records, n_samples=50, seed=1)
    b = dream_local(records, n_samples=50, seed=2)
    # At least some records should differ
    differences = sum(
        1 for ra, rb in zip(a.records, b.records) if ra["age"] != rb["age"]
    )
    assert differences > 40  # Almost all should differ


# ─── dream_local: edge cases ────────────────────────────────────────────────


def test_dream_handles_missing_values():
    """Records with missing fields don't break the fit."""
    records = [
        {"a": 1.0, "b": "x"},
        {"a": 2.0},  # missing b
        {"b": "y"},  # missing a
        {"a": 3.0, "b": "x"},
    ]
    result = dream_local(records, n_samples=10)
    assert result.n_samples == 10
    for r in result.records:
        assert "a" in r
        assert "b" in r


def test_dream_negative_n_samples_raises():
    """Negative n_samples is a usage error."""
    with pytest.raises(ValueError):
        dream_local([{"x": 1.0}], n_samples=-1)


def test_dream_temperature_zero_collapses_to_means():
    """Temperature 0 means every numeric sample equals the column mean."""
    records = [{"x": 1.0}, {"x": 3.0}, {"x": 5.0}]
    result = dream_local(records, n_samples=20, temperature=0.0)
    for r in result.records:
        assert math.isclose(r["x"], 3.0, abs_tol=1e-9)  # mean of [1, 3, 5]
