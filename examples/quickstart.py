"""gigi-dream quickstart — three escalating examples.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


def example_basic():
    """Basic faithful sampling."""
    print("=" * 64)
    print("  Example 1 - basic faithful sampling (T = 1.0)")
    print("=" * 64)
    rng = np.random.default_rng(0)
    real = []
    for i in range(200):
        real.append({
            "age": float(np.clip(rng.normal(35, 8), 18, 80)),
            "country": str(rng.choice(["US", "CA", "UK", "DE"], p=[0.5, 0.2, 0.2, 0.1])),
            "salary": float(np.clip(rng.normal(75000, 25000), 20000, 300000)),
        })

    result = dream(real, n_samples=1000, temperature=1.0, seed=42)

    print(f"  input: 200 records, 3 columns")
    print(f"  synth: {result.n_samples} records, T={result.temperature}")
    print()
    print(f"  fitted columns:")
    for col in result.columns:
        if col.kind == "numeric":
            print(f"    {col.name:10}  numeric     mean={col.mean:10.2f}  sigma={col.sigma:10.2f}")
        else:
            top = sorted(zip(col.values, col.weights), key=lambda kv: -kv[1])[:3]
            top_str = ", ".join(f"{v} ({w:.0%})" for v, w in top)
            print(f"    {col.name:10}  categorical {top_str}")
    print()
    print(f"  first 3 synthetic records:")
    for r in result.records[:3]:
        print(f"    {r}")


def example_temperature():
    """Same data, different temperatures."""
    print()
    print("=" * 64)
    print("  Example 2 - temperature changes spread")
    print("=" * 64)
    rng = np.random.default_rng(1)
    real = [{"x": float(rng.normal(10, 1))} for _ in range(500)]

    for T in [0.5, 1.0, 2.0, 4.0]:
        result = dream(real, n_samples=2000, temperature=T, seed=42)
        synth_std = float(np.std([r["x"] for r in result.records]))
        print(f"    T = {T:.1f}:  synthetic sigma = {synth_std:.3f}")
    print()
    print(f"  input sigma was ~1.0. Note synthetic sigma scales with sqrt(T).")


def example_reproducibility():
    """Demonstrate seed reproducibility."""
    print()
    print("=" * 64)
    print("  Example 3 - reproducibility (seed)")
    print("=" * 64)
    rng = np.random.default_rng(2)
    real = [{"x": float(rng.normal(0, 1))} for _ in range(100)]

    a = dream(real, n_samples=5, seed=42)
    b = dream(real, n_samples=5, seed=42)
    c = dream(real, n_samples=5, seed=999)

    fmt = lambda result: [f"{r['x']:.4f}" for r in result.records]
    print(f"  seed=42 first  call: {fmt(a)}")
    print(f"  seed=42 second call: {fmt(b)}")
    print(f"  seed=999 call:       {fmt(c)}")
    print()
    same = all(
        abs(ra["x"] - rb["x"]) < 1e-9 for ra, rb in zip(a.records, b.records)
    )
    different = any(
        abs(ra["x"] - rc["x"]) > 1e-9 for ra, rc in zip(a.records, c.records)
    )
    print(f"  same seed -> same records:       {same}")
    print(f"  different seed -> different:     {different}")


def main():
    example_basic()
    example_temperature()
    example_reproducibility()


if __name__ == "__main__":
    main()
