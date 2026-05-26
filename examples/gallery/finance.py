"""Finance examples — 6 use cases for gigi-dream.

Covers: bank transactions, investment portfolios, loan applications,
credit card spend, market data, fraud signals.
"""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


# ─── 1. Bank transactions ────────────────────────────────────────────────────


def example_transactions():
    """Bank account transactions — categorical merchant + heavy-tailed amount.

    Use case: test fixtures for payment-rail integration (PRISM-style) without
    using real customer data.
    """
    print("=" * 64)
    print("  1. Bank transactions — 1500 txns, mixed categorical + numeric")
    print("=" * 64)
    rng = np.random.default_rng(0)
    categories = ["grocery", "gas", "restaurant", "online", "utilities", "rent",
                  "payroll", "transfer", "ATM", "fee", "subscription"]
    rails = ["ACH", "WIRE", "RTP", "CARD"]
    real = []
    for _ in range(1500):
        real.append({
            "amount":   float(np.exp(rng.normal(3.5, 1.2))),  # log-normal
            "category": str(rng.choice(categories)),
            "rail":     str(rng.choice(rails, p=[0.45, 0.15, 0.10, 0.30])),
            "merchant_country": str(rng.choice(["US", "CA", "UK", "DE", "FR", "JP"],
                                               p=[0.78, 0.05, 0.05, 0.04, 0.04, 0.04])),
            "is_debit":     bool(rng.random() < 0.62),
            "is_recurring": bool(rng.random() < 0.18),
        })
    result = dream(real, n_samples=5000, temperature=1.0, seed=42)
    print(f"  → {result.n_samples} synthetic transactions")
    print(f"  → first 3 records:")
    for r in result.records[:3]:
        print(f"      ${r['amount']:8.2f}  {r['category']:14}  {r['rail']:5}  {r['merchant_country']}")


# ─── 2. Investment portfolios ────────────────────────────────────────────────


def example_portfolios():
    """Holdings — ticker, share count, sector, market value.

    Use case: dev portfolio-management dashboard without real client holdings.
    """
    print()
    print("=" * 64)
    print("  2. Investment portfolios — 800 holdings, sector-weighted")
    print("=" * 64)
    rng = np.random.default_rng(1)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "JPM", "JNJ",
               "PG", "XOM", "UNH", "V", "WMT", "DIS", "HD", "BAC"]
    sectors = ["Tech", "Tech", "Tech", "Tech", "Tech", "Tech", "Finance", "Healthcare",
               "ConsStaples", "Energy", "Healthcare", "Finance", "ConsStaples",
               "Comm", "ConsDisc", "Finance"]
    real = []
    for _ in range(800):
        idx = int(rng.integers(0, len(tickers)))
        real.append({
            "ticker": tickers[idx],
            "sector": sectors[idx],
            "shares": int(np.clip(rng.lognormal(4.5, 1.2), 1, 10000)),
            "avg_cost": float(np.clip(rng.normal(150, 80), 5, 800)),
            "market_value": float(np.clip(rng.lognormal(9.0, 1.5), 100, 1_000_000)),
            "gain_loss_pct": float(rng.normal(0.05, 0.18)),
        })
    result = dream(real, n_samples=2000, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic holdings")
    print(f"  → fitted ticker distribution: {len(result.columns[0].values)} distinct tickers")


# ─── 3. Loan applications ────────────────────────────────────────────────────


def example_loan_apps():
    """Loan applications — income, credit, decision label.

    Use case: ML training-set augmentation for credit-decision models.
    Higher T to encourage more boundary-case records.
    """
    print()
    print("=" * 64)
    print("  3. Loan applications — 600 apps, classification features (T=1.5)")
    print("=" * 64)
    rng = np.random.default_rng(2)
    real = []
    for _ in range(600):
        income = float(np.exp(rng.normal(11.0, 0.6)))  # log-normal income
        credit = int(np.clip(rng.normal(710, 80), 350, 850))
        # Decision is correlated with income & credit, but we sample independently in v0
        # (this is the documented limitation — gigi-dream v0 doesn't preserve correlation)
        approved = (income > 50000) and (credit > 650) and (rng.random() < 0.85)
        real.append({
            "annual_income": income,
            "credit_score":  credit,
            "loan_amount":   float(np.clip(rng.normal(28000, 18000), 1000, 200000)),
            "term_months":   int(rng.choice([12, 24, 36, 48, 60, 72])),
            "purpose":       str(rng.choice(["debt_consolidation", "home_improvement",
                                             "auto", "medical", "wedding", "other"])),
            "approved":      bool(approved),
        })
    result = dream(real, n_samples=2000, temperature=1.5, seed=11)
    print(f"  → {result.n_samples} synthetic apps (augmented training set)")
    print(f"  → approval rate (real):    {sum(r['approved'] for r in real) / len(real):.1%}")
    print(f"  → approval rate (synth):   {sum(r['approved'] for r in result.records) / result.n_samples:.1%}")


# ─── 4. Credit card spend (per-category aggregates) ──────────────────────────


def example_card_spend():
    """Card spend rolled up to per-month per-category — wide format.

    Use case: build dashboards on synthetic spend tracking.
    """
    print()
    print("=" * 64)
    print("  4. Card spend — 1000 customer-months, 11 spend categories")
    print("=" * 64)
    rng = np.random.default_rng(3)
    cats = ["grocery", "dining", "fuel", "travel", "entertainment", "shopping",
            "utilities", "rent", "health", "subscriptions", "other"]
    real = []
    for _ in range(1000):
        rec = {"month": int(rng.integers(1, 13))}
        for c in cats:
            # heavy-tailed spend per category
            rec[c] = float(np.clip(rng.lognormal(4.5, 1.3), 0, 8000))
        real.append(rec)
    result = dream(real, n_samples=3000, temperature=1.0, seed=4)
    print(f"  → {result.n_samples} synthetic customer-months × {len(cats)} categories")
    print(f"  → first record (truncated):")
    r0 = result.records[0]
    print(f"      month={r0['month']}, grocery=${r0['grocery']:.0f}, dining=${r0['dining']:.0f}, travel=${r0['travel']:.0f}")


# ─── 5. Market data — OHLCV time series ──────────────────────────────────────


def example_market_data():
    """OHLCV time series — typical financial data shape.

    Use case: backtest harness synthetic data; need realistic OHLCV without
    licensing market data.
    """
    print()
    print("=" * 64)
    print("  5. Market OHLCV — 1000 bars, 5 numeric columns")
    print("=" * 64)
    rng = np.random.default_rng(5)
    # Generate via random walk so close prices have structure
    price = 100.0
    real = []
    for _ in range(1000):
        rng_open = float(price * (1 + rng.normal(0, 0.002)))
        rng_close = float(price * (1 + rng.normal(0, 0.012)))
        high = max(rng_open, rng_close) * (1 + abs(rng.normal(0, 0.005)))
        low  = min(rng_open, rng_close) * (1 - abs(rng.normal(0, 0.005)))
        vol  = int(np.clip(rng.lognormal(13.5, 0.4), 100_000, 50_000_000))
        real.append({"open": rng_open, "high": high, "low": low, "close": rng_close, "volume": vol})
        price = rng_close
    result = dream(real, n_samples=5000, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic bars")
    print(f"  → note: gigi-dream v0 doesn't preserve OHLC ordering constraints (H>=O,C; L<=O,C)")
    print(f"  →       use GigiBackend with anisotropic fit for constrained sampling")


# ─── 6. Fraud signal features ────────────────────────────────────────────────


def example_fraud_features():
    """Feature vectors for a fraud-detection model — heavily imbalanced.

    Use case: minority-class augmentation for ML model training.
    """
    print()
    print("=" * 64)
    print("  6. Fraud features — 3000 txns (97% normal, 3% fraud)")
    print("=" * 64)
    rng = np.random.default_rng(6)
    real = []
    for _ in range(3000):
        is_fraud = rng.random() < 0.03
        amt_log = rng.normal(5.5 if not is_fraud else 7.5, 1.0)
        real.append({
            "amount":              float(np.exp(amt_log)),
            "velocity_24h":        int(np.clip(rng.poisson(2 if not is_fraud else 8), 0, 100)),
            "merchant_risk_score": float(np.clip(rng.beta(2, 8) if not is_fraud else rng.beta(7, 3), 0, 1)),
            "device_seen_before":  bool(rng.random() < (0.85 if not is_fraud else 0.3)),
            "country_mismatch":    bool(rng.random() < (0.04 if not is_fraud else 0.45)),
            "label":               "fraud" if is_fraud else "normal",
        })
    # Filter to fraud cases and augment
    fraud_real = [r for r in real if r["label"] == "fraud"]
    print(f"  → real: {len(real)} total ({len(fraud_real)} fraud, {len(real)-len(fraud_real)} normal)")
    augmented_fraud = dream(fraud_real, n_samples=300, temperature=1.5, seed=42)
    print(f"  → augmented fraud cases (T=1.5): {augmented_fraud.n_samples}")
    print(f"  → new training set: {len(real) - len(fraud_real)} normal + {len(fraud_real)} real fraud + {augmented_fraud.n_samples} synth fraud")


def main():
    example_transactions()
    example_portfolios()
    example_loan_apps()
    example_card_spend()
    example_market_data()
    example_fraud_features()


if __name__ == "__main__":
    main()
