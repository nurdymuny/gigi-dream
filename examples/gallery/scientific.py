"""Scientific examples — 4 use cases for gigi-dream."""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


def example_lab_experiments():
    """Chemistry / physics lab readings — controlled & response variables."""
    print("=" * 64)
    print("  1. Lab experiments — 600 trials, 8 columns")
    print("=" * 64)
    rng = np.random.default_rng(0)
    real = []
    for _ in range(600):
        real.append({
            "experiment_id":  int(rng.integers(1, 21)),
            "compound":       str(rng.choice(["A", "B", "C", "D", "E", "F"])),
            "temperature_k":  float(np.clip(rng.normal(298, 15), 200, 500)),
            "pressure_atm":   float(np.clip(rng.normal(1.0, 0.1), 0.1, 10)),
            "concentration_M": float(np.clip(rng.lognormal(-1, 0.6), 1e-4, 5)),
            "yield_pct":      float(np.clip(rng.beta(7, 3) * 100, 0, 100)),
            "rate_const":     float(np.clip(rng.lognormal(0, 1.5), 1e-3, 1e3)),
            "replicate":      int(rng.integers(1, 4)),
        })
    result = dream(real, n_samples=2000, temperature=1.0, seed=42)
    print(f"  → {result.n_samples} synthetic trials")


def example_particle_collisions():
    """Particle physics — collision energies, momenta, particle types."""
    print()
    print("=" * 64)
    print("  2. Particle collisions — 3000 events, 7 channels")
    print("=" * 64)
    rng = np.random.default_rng(1)
    particles = ["electron", "muon", "tau", "photon", "jet"]
    real = []
    for _ in range(3000):
        real.append({
            "collision_e_gev":  float(np.clip(rng.normal(13000, 100), 7000, 14000)),
            "pt_gev":           float(np.clip(rng.exponential(40), 1, 1000)),
            "eta":              float(np.clip(rng.normal(0, 1.5), -5, 5)),
            "phi":              float(rng.uniform(-np.pi, np.pi)),
            "particle":         str(rng.choice(particles, p=[0.25, 0.20, 0.05, 0.20, 0.30])),
            "missing_e_gev":    float(np.clip(rng.exponential(30), 0, 500)),
            "n_tracks":         int(np.clip(rng.poisson(28), 0, 200)),
        })
    result = dream(real, n_samples=10000, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic events")


def example_astronomical_observations():
    """Astronomical catalog — sky coordinates + magnitudes."""
    print()
    print("=" * 64)
    print("  3. Astronomical sources — 800 objects, 6 columns")
    print("=" * 64)
    rng = np.random.default_rng(2)
    real = []
    for _ in range(800):
        real.append({
            "ra_deg":     float(rng.uniform(0, 360)),
            "dec_deg":    float(np.clip(rng.normal(0, 30), -90, 90)),
            "v_mag":      float(np.clip(rng.normal(13, 4), -2, 22)),
            "redshift":   float(np.clip(rng.exponential(0.5), 0, 8)),
            "obj_class":  str(rng.choice(["star", "galaxy", "quasar", "asteroid", "unknown"],
                                          p=[0.45, 0.40, 0.05, 0.05, 0.05])),
            "snr":        float(np.clip(rng.lognormal(2, 1), 1, 1000)),
        })
    result = dream(real, n_samples=3000, temperature=1.0, seed=12)
    print(f"  → {result.n_samples} synthetic sources")
    print(f"  → object classes: {sorted(set(r['obj_class'] for r in result.records))}")


def example_genomic_variants():
    """Genomic variants — chromosome, position, type, allele frequency."""
    print()
    print("=" * 64)
    print("  4. Genomic variants — 5000 sites, mixed types")
    print("=" * 64)
    rng = np.random.default_rng(3)
    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    types = ["SNV", "INS", "DEL", "DUP", "INV"]
    real = []
    for _ in range(5000):
        real.append({
            "chrom":         str(rng.choice(chroms)),
            "position":      int(rng.integers(1, 250_000_000)),
            "variant_type":  str(rng.choice(types, p=[0.80, 0.08, 0.08, 0.03, 0.01])),
            "ref_allele":    str(rng.choice(["A", "C", "G", "T"])),
            "alt_allele":    str(rng.choice(["A", "C", "G", "T"])),
            "allele_freq":   float(np.clip(rng.beta(0.5, 5), 0, 1)),
            "qual":          float(np.clip(rng.lognormal(4, 1.5), 0, 5000)),
            "filter_pass":   bool(rng.random() < 0.92),
            "in_dbsnp":      bool(rng.random() < 0.45),
        })
    result = dream(real, n_samples=20000, temperature=1.0, seed=8)
    print(f"  → {result.n_samples} synthetic variants")
    print(f"  → note: synthetic positions ignore real genome structure (no chrom-length awareness)")


def main():
    example_lab_experiments()
    example_particle_collisions()
    example_astronomical_observations()
    example_genomic_variants()


if __name__ == "__main__":
    main()
