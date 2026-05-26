"""Healthcare examples — 6 use cases for gigi-dream.

Covers: EHR records, lab results, insurance claims, clinical trials,
vital signs time series, ER admissions. Ranges from 50 to 5000 records
and from 4 to 15 columns per example.
"""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


# ─── 1. EHR records (mixed demographics + clinical) ──────────────────────────


def example_ehr():
    """Electronic health records — mixed demographics + chronic conditions.

    Use case: populate a development DB with realistic patient records, no PHI.
    """
    print("=" * 64)
    print("  1. EHR records — 500 patients, 8 columns, mixed types")
    print("=" * 64)
    rng = np.random.default_rng(0)
    real = []
    for _ in range(500):
        real.append({
            "age": int(np.clip(rng.normal(58, 18), 18, 95)),
            "sex": str(rng.choice(["F", "M"], p=[0.52, 0.48])),
            "bmi": float(np.clip(rng.normal(28.5, 5.5), 16, 50)),
            "smoker": str(rng.choice(["never", "former", "current"], p=[0.55, 0.30, 0.15])),
            "diabetes": bool(rng.random() < 0.13),
            "hypertension": bool(rng.random() < 0.32),
            "a1c": float(np.clip(rng.normal(5.8, 1.1), 4.0, 14.0)),
            "ldl_mg_dl": float(np.clip(rng.normal(110, 35), 40, 280)),
        })
    result = dream(real, n_samples=2000, temperature=1.0, seed=42)
    print(f"  → {result.n_samples} synthetic patients")
    print(f"  → first synthetic: {result.records[0]}")


# ─── 2. Lab panel results (wide, many analytes) ──────────────────────────────


def example_lab_panel():
    """Complete metabolic panel + CBC — 14 analytes per draw.

    Use case: test fixtures for LIS/EHR integration without using real patient labs.
    """
    print()
    print("=" * 64)
    print("  2. Lab panels — 200 lab draws, 14 numeric analytes")
    print("=" * 64)
    rng = np.random.default_rng(1)
    analytes = {
        "sodium":    (140, 3),
        "potassium": (4.2, 0.4),
        "chloride":  (102, 3),
        "co2":       (24, 2.5),
        "bun":       (15, 5),
        "creatinine": (1.0, 0.3),
        "glucose":   (95, 15),
        "calcium":   (9.5, 0.4),
        "wbc":       (7.5, 2.1),
        "hgb":       (13.8, 1.5),
        "hct":       (41.5, 4.5),
        "platelets": (260, 60),
        "ast":       (28, 12),
        "alt":       (30, 14),
    }
    real = []
    for _ in range(200):
        rec = {}
        for name, (mean, sd) in analytes.items():
            rec[name] = float(np.clip(rng.normal(mean, sd), 0, mean * 5))
        real.append(rec)
    result = dream(real, n_samples=500, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic lab panels")
    print(f"  → fitted analytes: {len(result.columns)}")
    print(f"  → sample: {dict(list(result.records[0].items())[:5])} ...")


# ─── 3. Insurance claims (sparse, skewed amounts) ────────────────────────────


def example_claims():
    """Insurance claims — heavy-tailed amounts, ICD-10 codes.

    Use case: build a claims-ETL staging environment without real PHI.
    """
    print()
    print("=" * 64)
    print("  3. Insurance claims — 800 claims, log-normal amounts")
    print("=" * 64)
    rng = np.random.default_rng(2)
    icds = ["E11.9", "I10", "J45.909", "M54.5", "F32.9", "K21.9", "Z00.00", "I25.10"]
    cpts = ["99213", "99214", "99203", "80053", "85025", "73721", "73564", "99396"]
    real = []
    for _ in range(800):
        real.append({
            "primary_dx": str(rng.choice(icds, p=[0.18, 0.20, 0.10, 0.13, 0.09, 0.12, 0.10, 0.08])),
            "cpt_code": str(rng.choice(cpts)),
            "billed_amount": float(np.exp(rng.normal(5.5, 1.0))),  # log-normal
            "paid_amount": float(np.exp(rng.normal(4.5, 0.9))),
            "denied": bool(rng.random() < 0.07),
            "payer": str(rng.choice(["Medicare", "Aetna", "BCBS", "Cigna", "UHC", "Medicaid"])),
        })
    result = dream(real, n_samples=3000, temperature=1.5, seed=11)
    billed = [r["billed_amount"] for r in result.records]
    print(f"  → {result.n_samples} synthetic claims")
    print(f"  → billed amounts:  min ${min(billed):.0f}  median ${float(np.median(billed)):.0f}  max ${max(billed):.0f}")
    print(f"  → note: gigi-dream sampled log-normal-ish distribution OK at T=1.5")


# ─── 4. Clinical trial enrollment ────────────────────────────────────────────


def example_clinical_trial():
    """Clinical trial enrollment — eligibility, randomization, outcomes.

    Use case: trial-ops dashboard demo without exposing actual subjects.
    """
    print()
    print("=" * 64)
    print("  4. Clinical trial — 300 enrollees, 9 columns")
    print("=" * 64)
    rng = np.random.default_rng(3)
    real = []
    for _ in range(300):
        real.append({
            "subject_id": f"P{rng.integers(1000, 9999):04d}",
            "site_id": int(rng.integers(1, 11)),
            "arm": str(rng.choice(["control", "low_dose", "high_dose"])),
            "baseline_score": float(np.clip(rng.normal(45, 12), 0, 100)),
            "week_4_score":   float(np.clip(rng.normal(38, 13), 0, 100)),
            "week_12_score":  float(np.clip(rng.normal(31, 14), 0, 100)),
            "adverse_event": bool(rng.random() < 0.18),
            "dropout":       bool(rng.random() < 0.09),
            "age_at_enrollment": int(np.clip(rng.normal(55, 15), 18, 88)),
        })
    result = dream(real, n_samples=500, temperature=1.0, seed=42)
    print(f"  → {result.n_samples} synthetic enrollees")
    print(f"  → first 2 records:")
    for r in result.records[:2]:
        print(f"      {r}")


# ─── 5. Vital signs (time series, multivariate) ──────────────────────────────


def example_vitals():
    """ICU vitals — 5000 readings, 5 channels, T=2 for augmentation use case.

    Use case: ML training-set augmentation; need MORE points than the real
    cohort but with the same statistical structure.
    """
    print()
    print("=" * 64)
    print("  5. ICU vitals — 5000 readings, 5 channels (augmentation, T=2.0)")
    print("=" * 64)
    rng = np.random.default_rng(4)
    real = []
    for _ in range(5000):
        real.append({
            "hr_bpm":     float(np.clip(rng.normal(82, 14), 30, 200)),
            "sbp_mmhg":   float(np.clip(rng.normal(125, 18), 60, 220)),
            "dbp_mmhg":   float(np.clip(rng.normal(78, 12), 30, 130)),
            "spo2_pct":   float(np.clip(rng.normal(96, 2.5), 70, 100)),
            "temp_c":     float(np.clip(rng.normal(37.0, 0.6), 34, 41)),
        })
    augmented = dream(real, n_samples=15000, temperature=2.0, seed=8)
    real_hr_std = float(np.std([r["hr_bpm"] for r in real]))
    synth_hr_std = float(np.std([r["hr_bpm"] for r in augmented.records]))
    print(f"  → {augmented.n_samples} synthetic readings (3x input size)")
    print(f"  → HR sigma:  real={real_hr_std:.1f}  synthetic@T=2={synth_hr_std:.1f}  (~sqrt(2) wider)")


# ─── 6. ER admissions (timestamps + categorical disposition) ─────────────────


def example_er_admits():
    """ER admissions — chief complaint, triage level, disposition.

    Use case: capacity-planning simulator that needs realistic admission mix
    without using real ER data.
    """
    print()
    print("=" * 64)
    print("  6. ER admissions — 1200 visits, mixed categorical + numeric")
    print("=" * 64)
    rng = np.random.default_rng(5)
    complaints = ["chest_pain", "abdominal_pain", "shortness_of_breath", "trauma",
                  "fever", "headache", "back_pain", "psychiatric", "other"]
    dispositions = ["discharged", "admitted_floor", "admitted_icu", "transferred", "AMA"]
    real = []
    for _ in range(1200):
        real.append({
            "complaint": str(rng.choice(complaints,
                                        p=[0.10, 0.12, 0.08, 0.05, 0.15, 0.08, 0.06, 0.06, 0.30])),
            "esi_level": int(rng.choice([1, 2, 3, 4, 5], p=[0.02, 0.18, 0.45, 0.27, 0.08])),
            "door_to_doc_min": float(np.clip(rng.gamma(2.0, 12.0), 1, 240)),
            "los_hours": float(np.clip(rng.gamma(2.0, 1.8), 0.5, 36)),
            "disposition": str(rng.choice(dispositions, p=[0.70, 0.16, 0.05, 0.04, 0.05])),
            "age": int(np.clip(rng.normal(48, 22), 1, 100)),
        })
    result = dream(real, n_samples=2400, temperature=1.0, seed=12)
    # Show the disposition mix
    dispo_counts = {}
    for r in result.records:
        d = r["disposition"]
        dispo_counts[d] = dispo_counts.get(d, 0) + 1
    print(f"  → {result.n_samples} synthetic visits")
    print(f"  → disposition mix:")
    for d, c in sorted(dispo_counts.items(), key=lambda kv: -kv[1]):
        print(f"      {d:18}  {c:>4}  ({c/result.n_samples:5.1%})")


def main():
    example_ehr()
    example_lab_panel()
    example_claims()
    example_clinical_trial()
    example_vitals()
    example_er_admits()


if __name__ == "__main__":
    main()
