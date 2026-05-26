"""Ops / SRE examples — 5 use cases for gigi-dream."""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


def example_latency_percentiles():
    """Per-service latency percentiles — synthetic for load tests."""
    print("=" * 64)
    print("  1. Latency percentiles — 1000 service-minutes, 6 percentiles")
    print("=" * 64)
    rng = np.random.default_rng(0)
    real = []
    for _ in range(1000):
        # Generate consistent percentiles (sorted)
        base = rng.normal(50, 10)
        real.append({
            "service":  str(rng.choice(["api", "auth", "payments", "search", "ml-infer"])),
            "p50_ms":   float(np.clip(base, 1, 5000)),
            "p75_ms":   float(np.clip(base * 1.5, 1, 5000)),
            "p90_ms":   float(np.clip(base * 2.5, 1, 5000)),
            "p95_ms":   float(np.clip(base * 4, 1, 5000)),
            "p99_ms":   float(np.clip(base * 8, 1, 10000)),
            "rps":      float(np.clip(rng.lognormal(5, 1), 1, 100_000)),
        })
    result = dream(real, n_samples=5000, temperature=1.0, seed=42)
    print(f"  → {result.n_samples} synthetic service-minutes")
    print(f"  → note: percentile ordering NOT preserved in v0 (independent column sampling)")


def example_error_rates():
    """Error-rate metrics per service per minute — sparse signal."""
    print()
    print("=" * 64)
    print("  2. Error rates — 2000 service-minutes, 5 categories")
    print("=" * 64)
    rng = np.random.default_rng(1)
    real = []
    for _ in range(2000):
        real.append({
            "service":      str(rng.choice(["frontend", "api", "worker", "cache", "db"])),
            "requests":     int(np.clip(rng.lognormal(7, 1), 1, 100_000)),
            "errors_4xx":   int(np.clip(rng.gamma(1, 5), 0, 1000)),
            "errors_5xx":   int(np.clip(rng.exponential(0.5), 0, 200)),
            "error_rate":   float(np.clip(rng.beta(1, 50), 0, 1)),
            "alert_fired":  bool(rng.random() < 0.04),
        })
    result = dream(real, n_samples=8000, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic minutes")


def example_deploy_events():
    """Deploy events — version bumps, durations, outcomes."""
    print()
    print("=" * 64)
    print("  3. Deploy events — 800 deployments, mixed shape")
    print("=" * 64)
    rng = np.random.default_rng(2)
    outcomes = ["success", "rollback", "partial_failure", "timeout"]
    real = []
    for _ in range(800):
        real.append({
            "service":         str(rng.choice(["api", "worker", "frontend", "db-migration", "infra"])),
            "version":         f"v{rng.integers(1, 20)}.{rng.integers(0, 50)}.{rng.integers(0, 50)}",
            "duration_min":    float(np.clip(rng.gamma(3, 4), 1, 120)),
            "outcome":         str(rng.choice(outcomes, p=[0.85, 0.08, 0.04, 0.03])),
            "lines_changed":   int(np.clip(rng.lognormal(5, 1.5), 1, 50_000)),
            "rollback_needed": bool(rng.random() < 0.10),
        })
    result = dream(real, n_samples=3000, temperature=1.0, seed=12)
    print(f"  → {result.n_samples} synthetic deploys")


def example_service_health():
    """Service health metrics — CPU, mem, disk."""
    print()
    print("=" * 64)
    print("  4. Service health — 6000 host-minutes, 6 metrics")
    print("=" * 64)
    rng = np.random.default_rng(3)
    real = []
    for _ in range(6000):
        real.append({
            "host_class":   str(rng.choice(["web", "worker", "db", "cache", "gpu"], p=[0.30, 0.30, 0.15, 0.15, 0.10])),
            "cpu_pct":      float(np.clip(rng.beta(2, 5) * 100, 0, 100)),
            "mem_pct":      float(np.clip(rng.beta(4, 4) * 100, 0, 100)),
            "disk_pct":     float(np.clip(rng.beta(3, 5) * 100, 0, 100)),
            "net_mb_s":     float(np.clip(rng.lognormal(2, 1.5), 0, 10000)),
            "load_avg":     float(np.clip(rng.gamma(2, 1), 0, 50)),
        })
    result = dream(real, n_samples=20000, temperature=1.0, seed=8)
    print(f"  → {result.n_samples} synthetic host-minutes")


def example_incident_tickets():
    """Incident tickets — severity, MTTR, type."""
    print()
    print("=" * 64)
    print("  5. Incident tickets — 250 incidents, heavy-tailed MTTR")
    print("=" * 64)
    rng = np.random.default_rng(4)
    severities = ["S1", "S2", "S3", "S4"]
    types = ["latency", "errors", "outage", "data_quality", "security", "capacity"]
    real = []
    for _ in range(250):
        sev = str(rng.choice(severities, p=[0.05, 0.20, 0.45, 0.30]))
        mttr_factor = {"S1": 8, "S2": 24, "S3": 72, "S4": 240}[sev]
        real.append({
            "severity":          sev,
            "type":              str(rng.choice(types)),
            "mttr_hours":        float(np.clip(rng.lognormal(np.log(mttr_factor), 0.8), 0.1, 1000)),
            "ttr_responder_min": int(np.clip(rng.gamma(2, 8), 1, 300)),
            "n_responders":      int(np.clip(rng.poisson(2), 1, 15)),
            "post_mortem_done":  bool(rng.random() < 0.78),
            "preventable":       bool(rng.random() < 0.55),
        })
    result = dream(real, n_samples=1000, temperature=1.0, seed=11)
    print(f"  → {result.n_samples} synthetic incidents")


def main():
    example_latency_percentiles()
    example_error_rates()
    example_deploy_events()
    example_service_health()
    example_incident_tickets()


if __name__ == "__main__":
    main()
