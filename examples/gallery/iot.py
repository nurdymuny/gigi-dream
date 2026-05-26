"""IoT / sensor examples — 5 use cases for gigi-dream."""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


def example_industrial_sensors():
    """Manufacturing floor — temperature, pressure, vibration across 50 sensors."""
    print("=" * 64)
    print("  1. Industrial sensors — 10000 readings, 7 channels")
    print("=" * 64)
    rng = np.random.default_rng(0)
    real = []
    for _ in range(10000):
        real.append({
            "sensor_id":         int(rng.integers(1, 51)),
            "temp_c":            float(np.clip(rng.normal(68, 8), 15, 200)),
            "pressure_psi":      float(np.clip(rng.normal(125, 18), 0, 300)),
            "vibration_mm_s":    float(np.clip(rng.gamma(2, 0.5), 0.01, 20)),
            "humidity_pct":      float(np.clip(rng.normal(45, 12), 0, 100)),
            "motor_rpm":         int(np.clip(rng.normal(1750, 65), 0, 3600)),
            "fault_flag":        bool(rng.random() < 0.012),
        })
    result = dream(real, n_samples=30000, temperature=1.0, seed=42)
    print(f"  → {result.n_samples} synthetic readings")
    print(f"  → fault rate: real={sum(r['fault_flag'] for r in real)/len(real):.2%}, "
          f"synth={sum(r['fault_flag'] for r in result.records)/result.n_samples:.2%}")


def example_smart_home():
    """Smart home device telemetry — energy use, occupancy, settings."""
    print()
    print("=" * 64)
    print("  2. Smart home — 4000 readings across device types")
    print("=" * 64)
    rng = np.random.default_rng(1)
    devices = ["thermostat", "doorbell", "lock", "lights", "outlet", "speaker"]
    real = []
    for _ in range(4000):
        real.append({
            "device_type":     str(rng.choice(devices)),
            "power_watts":     float(np.clip(rng.gamma(1.5, 8), 0.1, 500)),
            "occupied":        bool(rng.random() < 0.55),
            "ambient_lux":     float(np.clip(rng.normal(180, 120), 0, 1500)),
            "setpoint_c":      float(rng.choice([18, 19, 20, 21, 22, 23, 24], p=[0.05, 0.10, 0.25, 0.30, 0.20, 0.07, 0.03])),
            "online":          bool(rng.random() < 0.97),
        })
    result = dream(real, n_samples=12000, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic device readings")


def example_weather():
    """Weather station — multi-variable atmospheric observations."""
    print()
    print("=" * 64)
    print("  3. Weather stations — 2500 hourly observations, 8 channels")
    print("=" * 64)
    rng = np.random.default_rng(2)
    real = []
    for _ in range(2500):
        real.append({
            "temp_c":        float(np.clip(rng.normal(14, 11), -25, 45)),
            "dewpoint_c":    float(np.clip(rng.normal(8, 9), -30, 30)),
            "pressure_hpa":  float(np.clip(rng.normal(1013, 8), 950, 1050)),
            "wind_speed_mps": float(np.clip(rng.gamma(2, 1.5), 0, 40)),
            "wind_dir_deg":  float(rng.uniform(0, 360)),
            "humidity_pct":  float(np.clip(rng.normal(65, 20), 0, 100)),
            "precip_mm":     float(np.clip(rng.exponential(0.5), 0, 80)),
            "visibility_km": float(np.clip(rng.normal(20, 8), 0.1, 50)),
        })
    result = dream(real, n_samples=8000, temperature=1.0, seed=12)
    print(f"  → {result.n_samples} synthetic observations")


def example_vehicle_telemetry():
    """Vehicle telemetry — speed, fuel, location-coded categorical."""
    print()
    print("=" * 64)
    print("  4. Vehicle telemetry — 6000 readings from fleet")
    print("=" * 64)
    rng = np.random.default_rng(3)
    real = []
    for _ in range(6000):
        real.append({
            "vehicle_id":     int(rng.integers(1, 201)),
            "speed_kmh":      float(np.clip(rng.normal(65, 25), 0, 200)),
            "fuel_pct":       float(np.clip(rng.uniform(5, 100), 0, 100)),
            "engine_temp_c":  float(np.clip(rng.normal(92, 6), 60, 130)),
            "rpm":            int(np.clip(rng.normal(2200, 600), 500, 6500)),
            "gear":           int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.10, 0.15, 0.20, 0.30, 0.20])),
            "engine_on":      bool(rng.random() < 0.88),
        })
    result = dream(real, n_samples=20000, temperature=1.0, seed=8)
    print(f"  → {result.n_samples} synthetic telemetry rows")


def example_agriculture():
    """Agricultural sensors — soil moisture, light, plant health."""
    print()
    print("=" * 64)
    print("  5. Agriculture sensors — 1500 plot-readings, 6 channels")
    print("=" * 64)
    rng = np.random.default_rng(4)
    crops = ["corn", "soy", "wheat", "rice", "potato", "tomato"]
    real = []
    for _ in range(1500):
        real.append({
            "crop":              str(rng.choice(crops)),
            "soil_moisture_pct": float(np.clip(rng.normal(38, 12), 0, 100)),
            "soil_temp_c":       float(np.clip(rng.normal(18, 6), -5, 40)),
            "light_par_umol":    float(np.clip(rng.normal(1200, 400), 0, 2500)),
            "ph":                float(np.clip(rng.normal(6.5, 0.5), 4.5, 8.5)),
            "ec_ds_per_m":       float(np.clip(rng.normal(1.5, 0.5), 0.1, 5.0)),
            "rainfall_mm":       float(np.clip(rng.exponential(2), 0, 80)),
        })
    result = dream(real, n_samples=5000, temperature=1.0, seed=11)
    print(f"  → {result.n_samples} synthetic plot-readings")


def main():
    example_industrial_sensors()
    example_smart_home()
    example_weather()
    example_vehicle_telemetry()
    example_agriculture()


if __name__ == "__main__":
    main()
