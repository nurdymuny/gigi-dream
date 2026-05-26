"""ML / data science examples — 5 use cases for gigi-dream."""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


def example_training_set_augmentation():
    """Augment a small training set to improve model robustness."""
    print("=" * 64)
    print("  1. Training set augmentation — 200 real → 2000 synthetic, T=2.0")
    print("=" * 64)
    rng = np.random.default_rng(0)
    real = []
    for _ in range(200):
        x1 = float(rng.normal(0, 1))
        x2 = float(rng.normal(0, 1))
        x3 = float(rng.normal(0, 1))
        # Three classes by simple decision boundary on x1+x2
        if x1 + x2 > 0.5:
            label = "A"
        elif x1 + x2 < -0.5:
            label = "B"
        else:
            label = "C"
        real.append({"x1": x1, "x2": x2, "x3": x3, "label": label})
    augmented = dream(real, n_samples=2000, temperature=2.0, seed=42)
    print(f"  → {augmented.n_samples} synthetic training samples (10× input)")
    class_mix = {}
    for r in augmented.records:
        class_mix[r["label"]] = class_mix.get(r["label"], 0) + 1
    print(f"  → class mix: {class_mix}")


def example_tabular_features():
    """Generic tabular ML features — numeric + categorical."""
    print()
    print("=" * 64)
    print("  2. Tabular ML features — 1000 rows × 20 columns")
    print("=" * 64)
    rng = np.random.default_rng(1)
    real = []
    for _ in range(1000):
        rec = {}
        for i in range(16):
            rec[f"feat_{i:02d}"] = float(rng.normal(0, 1))
        rec["cat_a"] = str(rng.choice(["red", "green", "blue"]))
        rec["cat_b"] = str(rng.choice(["small", "medium", "large", "xl"]))
        rec["bool_c"] = bool(rng.random() < 0.4)
        rec["target"] = float(np.clip(rng.normal(0.5, 0.2), 0, 1))
        real.append(rec)
    result = dream(real, n_samples=3000, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic rows with {len(result.columns)} columns")


def example_classification_dataset():
    """Multi-class classification — 5 classes, imbalanced."""
    print()
    print("=" * 64)
    print("  3. Multi-class classification — 800 rows, 5 classes, imbalanced")
    print("=" * 64)
    rng = np.random.default_rng(2)
    classes = ["cat_a", "cat_b", "cat_c", "cat_d", "cat_e"]
    probs = [0.45, 0.25, 0.15, 0.10, 0.05]
    real = []
    for _ in range(800):
        label_idx = int(rng.choice(len(classes), p=probs))
        # Features with class-specific means
        real.append({
            "label":   classes[label_idx],
            "feat_1":  float(rng.normal(label_idx, 1)),
            "feat_2":  float(rng.normal(0, 1 + label_idx * 0.5)),
            "feat_3":  float(rng.normal(label_idx % 2, 1)),
            "score":   float(np.clip(rng.beta(2 + label_idx, 5), 0, 1)),
        })
    result = dream(real, n_samples=3000, temperature=1.0, seed=12)
    print(f"  → {result.n_samples} synthetic rows")
    print(f"  → class distribution preserved by categorical sampler")


def example_time_series_features():
    """Time-series features — sliding-window aggregates."""
    print()
    print("=" * 64)
    print("  4. Time-series features — 2000 windows × 12 aggregates")
    print("=" * 64)
    rng = np.random.default_rng(3)
    real = []
    for _ in range(2000):
        # Simulated window stats (mean, std, min, max, etc.)
        rec = {}
        for w in ["1m", "5m", "15m"]:
            rec[f"mean_{w}"] = float(rng.normal(0, 1))
            rec[f"std_{w}"]  = float(np.clip(rng.gamma(2, 0.5), 0.01, 5))
            rec[f"max_{w}"]  = float(rng.normal(2, 1))
            rec[f"min_{w}"]  = float(rng.normal(-2, 1))
        real.append(rec)
    result = dream(real, n_samples=10000, temperature=1.0, seed=8)
    print(f"  → {result.n_samples} synthetic windows × {len(result.columns)} stats")


def example_embedding_like_data():
    """High-dimensional vectors — embeddings or learned representations."""
    print()
    print("=" * 64)
    print("  5. High-D embeddings — 500 vectors × 50 dims")
    print("=" * 64)
    rng = np.random.default_rng(4)
    real = []
    for _ in range(500):
        # L2-normalized-ish unit vector
        v = rng.standard_normal(50)
        v = v / float(np.linalg.norm(v))
        rec = {f"d{i:02d}": float(v[i]) for i in range(50)}
        rec["label"] = str(rng.choice(["A", "B", "C"]))
        real.append(rec)
    result = dream(real, n_samples=2000, temperature=1.0, seed=11)
    print(f"  → {result.n_samples} synthetic 50-D vectors")
    print(f"  → note: per-column independent sampling does NOT preserve unit-norm constraint")
    print(f"  →       for unit-norm sampling, use GigiBackend (anisotropic + L2-normalized fit)")


def main():
    example_training_set_augmentation()
    example_tabular_features()
    example_classification_dataset()
    example_time_series_features()
    example_embedding_like_data()


if __name__ == "__main__":
    main()
