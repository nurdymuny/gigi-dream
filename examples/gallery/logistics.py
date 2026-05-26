"""Logistics examples — 5 use cases for gigi-dream."""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


def example_shipments():
    """Shipment records — origin, dest, weight, carrier."""
    print("=" * 64)
    print("  1. Shipments — 3000 records, mixed types")
    print("=" * 64)
    rng = np.random.default_rng(0)
    carriers = ["UPS", "FedEx", "USPS", "DHL", "OnTrac"]
    real = []
    for _ in range(3000):
        real.append({
            "origin_state":     str(rng.choice(["CA", "NY", "TX", "IL", "WA", "OH", "GA"])),
            "dest_state":       str(rng.choice(["CA", "NY", "TX", "IL", "WA", "OH", "GA", "FL", "NJ"])),
            "weight_kg":        float(np.clip(rng.lognormal(0.7, 1.0), 0.1, 200)),
            "distance_km":      float(np.clip(rng.gamma(2, 800), 10, 7000)),
            "carrier":          str(rng.choice(carriers, p=[0.30, 0.30, 0.20, 0.10, 0.10])),
            "service_level":    str(rng.choice(["ground", "2day", "overnight"], p=[0.65, 0.25, 0.10])),
            "transit_days":     int(np.clip(rng.gamma(2, 1.5), 1, 14)),
            "delivered_on_time": bool(rng.random() < 0.91),
        })
    result = dream(real, n_samples=10000, temperature=1.0, seed=42)
    print(f"  → {result.n_samples} synthetic shipments")
    print(f"  → on-time rate: real={sum(r['delivered_on_time'] for r in real)/len(real):.1%}, "
          f"synth={sum(r['delivered_on_time'] for r in result.records)/result.n_samples:.1%}")


def example_warehouse_inventory():
    """Warehouse inventory — SKU, quantity, location, bin."""
    print()
    print("=" * 64)
    print("  2. Warehouse inventory — 2000 SKU-locations")
    print("=" * 64)
    rng = np.random.default_rng(1)
    real = []
    for _ in range(2000):
        real.append({
            "warehouse_id":   int(rng.integers(1, 21)),
            "aisle":          str(rng.choice([f"A{i:02d}" for i in range(1, 51)])),
            "quantity":       int(np.clip(rng.lognormal(3, 1.2), 0, 10000)),
            "weight_kg":      float(np.clip(rng.lognormal(0.5, 1.5), 0.01, 500)),
            "value_usd":      float(np.clip(rng.lognormal(3.5, 1.3), 1, 50000)),
            "days_in_stock":  int(np.clip(rng.exponential(45), 0, 365)),
            "low_stock_flag": bool(rng.random() < 0.13),
        })
    result = dream(real, n_samples=8000, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic inventory records")


def example_route_metrics():
    """Route stops + distances + times — delivery-route planning."""
    print()
    print("=" * 64)
    print("  3. Routes — 1200 route-days, 6 columns")
    print("=" * 64)
    rng = np.random.default_rng(2)
    real = []
    for _ in range(1200):
        real.append({
            "n_stops":            int(np.clip(rng.poisson(28), 5, 80)),
            "total_distance_km":  float(np.clip(rng.normal(180, 60), 20, 600)),
            "drive_time_min":     float(np.clip(rng.normal(310, 90), 60, 720)),
            "service_time_min":   float(np.clip(rng.gamma(3, 25), 30, 480)),
            "overtime_min":       float(np.clip(rng.exponential(15), 0, 120)),
            "fuel_used_l":        float(np.clip(rng.normal(40, 12), 5, 120)),
        })
    result = dream(real, n_samples=5000, temperature=1.0, seed=12)
    print(f"  → {result.n_samples} synthetic route-days")


def example_delivery_performance():
    """Delivery performance — outcome categorical, time/distance metrics."""
    print()
    print("=" * 64)
    print("  4. Delivery performance — 5000 deliveries, classification target")
    print("=" * 64)
    rng = np.random.default_rng(3)
    outcomes = ["on_time", "delayed_minor", "delayed_major", "failed_delivery", "exception"]
    real = []
    for _ in range(5000):
        real.append({
            "promised_window_h": float(rng.choice([2, 4, 8, 24, 48], p=[0.10, 0.20, 0.30, 0.30, 0.10])),
            "actual_minutes":    float(np.clip(rng.gamma(2, 60), 5, 600)),
            "weather_severity":  str(rng.choice(["clear", "rain", "snow", "extreme"], p=[0.65, 0.22, 0.10, 0.03])),
            "outcome":           str(rng.choice(outcomes, p=[0.82, 0.10, 0.04, 0.02, 0.02])),
            "attempts":          int(rng.choice([1, 2, 3], p=[0.92, 0.06, 0.02])),
            "signature_required": bool(rng.random() < 0.31),
        })
    result = dream(real, n_samples=15000, temperature=1.0, seed=8)
    print(f"  → {result.n_samples} synthetic deliveries")
    print(f"  → outcome mix preserved across categorical sampling")


def example_fleet_telemetry():
    """Fleet — vehicle, mileage, fuel-efficiency, maintenance flags."""
    print()
    print("=" * 64)
    print("  5. Fleet — 800 vehicle-days, 6 columns")
    print("=" * 64)
    rng = np.random.default_rng(4)
    real = []
    for _ in range(800):
        real.append({
            "vehicle_class":   str(rng.choice(["van", "box_truck", "semi", "pickup"], p=[0.30, 0.40, 0.20, 0.10])),
            "mileage_today":   float(np.clip(rng.normal(220, 70), 0, 600)),
            "fuel_used_gal":   float(np.clip(rng.normal(25, 10), 0, 100)),
            "mpg":             float(np.clip(rng.normal(8.5, 2.5), 4, 25)),
            "engine_hours":    float(np.clip(rng.normal(7.5, 2.5), 0, 16)),
            "maintenance_due": bool(rng.random() < 0.08),
        })
    result = dream(real, n_samples=3000, temperature=1.0, seed=11)
    print(f"  → {result.n_samples} synthetic vehicle-days")


def main():
    example_shipments()
    example_warehouse_inventory()
    example_route_metrics()
    example_delivery_performance()
    example_fleet_telemetry()


if __name__ == "__main__":
    main()
