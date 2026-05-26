"""E-commerce examples — 6 use cases for gigi-dream."""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


def example_orders():
    """Order history — items, totals, discounts, channel."""
    print("=" * 64)
    print("  1. Orders — 2000 orders, 7 columns")
    print("=" * 64)
    rng = np.random.default_rng(0)
    channels = ["web", "mobile_app", "marketplace", "phone", "store"]
    statuses = ["delivered", "shipped", "processing", "cancelled", "returned"]
    real = []
    for _ in range(2000):
        real.append({
            "subtotal":    float(np.exp(rng.normal(4.0, 0.9))),
            "discount":    float(np.clip(rng.beta(2, 8) * 50, 0, 200)),
            "tax":         float(rng.uniform(0, 20)),
            "shipping":    float(rng.choice([0, 5.99, 9.99, 14.99, 24.99], p=[0.4, 0.2, 0.2, 0.15, 0.05])),
            "n_items":     int(np.clip(rng.poisson(2.5), 1, 20)),
            "channel":     str(rng.choice(channels, p=[0.45, 0.35, 0.10, 0.02, 0.08])),
            "status":      str(rng.choice(statuses, p=[0.70, 0.12, 0.10, 0.04, 0.04])),
        })
    result = dream(real, n_samples=5000, temperature=1.0, seed=42)
    print(f"  → {result.n_samples} synthetic orders")
    print(f"  → channel mix: {sorted(set(r['channel'] for r in result.records))}")


def example_customers():
    """Customer profiles — demographics + lifetime value."""
    print()
    print("=" * 64)
    print("  2. Customer profiles — 1500 customers, 8 columns")
    print("=" * 64)
    rng = np.random.default_rng(1)
    real = []
    for _ in range(1500):
        real.append({
            "age":              int(np.clip(rng.normal(38, 13), 18, 85)),
            "signup_country":   str(rng.choice(["US", "CA", "UK", "AU", "DE", "FR", "JP"],
                                               p=[0.55, 0.10, 0.10, 0.05, 0.07, 0.07, 0.06])),
            "n_orders":         int(np.clip(rng.gamma(1.5, 4), 1, 200)),
            "ltv":              float(np.exp(rng.normal(6.0, 1.3))),  # log-normal
            "days_since_last":  int(np.clip(rng.exponential(60), 0, 800)),
            "email_subscribed": bool(rng.random() < 0.72),
            "premium_member":   bool(rng.random() < 0.18),
            "loyalty_tier":     str(rng.choice(["bronze", "silver", "gold", "platinum"],
                                               p=[0.50, 0.30, 0.15, 0.05])),
        })
    result = dream(real, n_samples=10000, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic customers")
    ltv = sorted(r["ltv"] for r in result.records)
    print(f"  → LTV percentiles: p50=${ltv[5000]:.0f}  p95=${ltv[9500]:.0f}  p99=${ltv[9900]:.0f}")


def example_products():
    """Product catalog — price, category, rating, attributes."""
    print()
    print("=" * 64)
    print("  3. Product catalog — 800 SKUs, mixed types")
    print("=" * 64)
    rng = np.random.default_rng(2)
    cats = ["electronics", "apparel", "home", "books", "toys", "beauty", "grocery", "outdoor"]
    real = []
    for _ in range(800):
        real.append({
            "category":   str(rng.choice(cats)),
            "price":      float(np.exp(rng.normal(3.5, 1.1))),
            "rating":     float(np.clip(rng.beta(8, 2) * 5, 1, 5)),
            "n_reviews":  int(np.clip(rng.gamma(1.5, 30), 0, 5000)),
            "in_stock":   bool(rng.random() < 0.86),
            "is_eco":     bool(rng.random() < 0.22),
            "weight_kg":  float(np.clip(rng.lognormal(0.3, 0.9), 0.05, 50)),
        })
    result = dream(real, n_samples=2000, temperature=1.5, seed=12)
    print(f"  → {result.n_samples} synthetic products")


def example_reviews():
    """Reviews — rating + text-length + helpful votes."""
    print()
    print("=" * 64)
    print("  4. Reviews — 3000 reviews, 5 columns")
    print("=" * 64)
    rng = np.random.default_rng(3)
    real = []
    for _ in range(3000):
        # Ratings are heavily skewed toward 5
        rating = int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.05, 0.10, 0.20, 0.60]))
        real.append({
            "rating":        rating,
            "n_words":       int(np.clip(rng.lognormal(4.5, 1.0), 1, 2000)),
            "helpful_votes": int(np.clip(rng.gamma(0.5, 5), 0, 500)),
            "verified":      bool(rng.random() < 0.78),
            "with_photo":    bool(rng.random() < 0.12),
        })
    result = dream(real, n_samples=10000, temperature=1.0, seed=4)
    avg_rating = float(np.mean([r["rating"] for r in result.records]))
    print(f"  → {result.n_samples} synthetic reviews")
    print(f"  → avg rating: real={float(np.mean([r['rating'] for r in real])):.2f}  synth={avg_rating:.2f}")


def example_cart_events():
    """Cart events — sparse, narrow, large count."""
    print()
    print("=" * 64)
    print("  5. Cart events — 5000 events, narrow shape")
    print("=" * 64)
    rng = np.random.default_rng(5)
    events = ["add", "remove", "view", "wishlist", "checkout_start", "checkout_complete"]
    real = []
    for _ in range(5000):
        real.append({
            "event":         str(rng.choice(events, p=[0.42, 0.10, 0.30, 0.08, 0.07, 0.03])),
            "device":        str(rng.choice(["mobile", "desktop", "tablet"], p=[0.62, 0.32, 0.06])),
            "quantity":      int(np.clip(rng.poisson(1.3) + 1, 1, 10)),
            "session_age_s": float(np.clip(rng.gamma(2.0, 60.0), 1, 3600)),
        })
    result = dream(real, n_samples=20000, temperature=1.0, seed=8)
    print(f"  → {result.n_samples} synthetic events (4x input)")


def example_returns():
    """Returns/RMAs — reason categorical + refund amount + processing time."""
    print()
    print("=" * 64)
    print("  6. Returns/RMAs — 500 returns, categorical reasons")
    print("=" * 64)
    rng = np.random.default_rng(6)
    reasons = ["didnt_fit", "not_as_described", "damaged", "no_longer_needed",
               "better_price_found", "quality_issue", "wrong_item"]
    real = []
    for _ in range(500):
        real.append({
            "reason":           str(rng.choice(reasons)),
            "refund_amount":    float(np.clip(rng.normal(45, 30), 0, 500)),
            "processing_days":  int(np.clip(rng.gamma(2.5, 1.5), 1, 30)),
            "with_receipt":     bool(rng.random() < 0.88),
            "exchange_offered": bool(rng.random() < 0.31),
        })
    result = dream(real, n_samples=2000, temperature=1.0, seed=11)
    print(f"  → {result.n_samples} synthetic returns")
    print(f"  → reason distribution preserved (top 3): "
          f"{sorted(set(r['reason'] for r in result.records))[:3]}")


def main():
    example_orders()
    example_customers()
    example_products()
    example_reviews()
    example_cart_events()
    example_returns()


if __name__ == "__main__":
    main()
