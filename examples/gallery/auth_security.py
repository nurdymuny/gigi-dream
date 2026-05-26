"""Auth / Security examples — 5 use cases for gigi-dream."""

from __future__ import annotations

import numpy as np

from gigi_dream import dream


def example_user_sessions():
    """User sessions — duration, device, location."""
    print("=" * 64)
    print("  1. User sessions — 4000 sessions, 7 columns")
    print("=" * 64)
    rng = np.random.default_rng(0)
    real = []
    for _ in range(4000):
        real.append({
            "duration_min":   float(np.clip(rng.exponential(12), 0.1, 360)),
            "page_views":     int(np.clip(rng.poisson(5), 1, 100)),
            "device_type":    str(rng.choice(["desktop", "mobile", "tablet"], p=[0.55, 0.38, 0.07])),
            "country":        str(rng.choice(["US", "UK", "DE", "JP", "BR", "IN", "AU"],
                                             p=[0.45, 0.10, 0.08, 0.10, 0.07, 0.12, 0.08])),
            "browser":        str(rng.choice(["chrome", "safari", "firefox", "edge", "other"],
                                             p=[0.62, 0.18, 0.08, 0.08, 0.04])),
            "is_authenticated": bool(rng.random() < 0.42),
            "bot_score":      float(np.clip(rng.beta(2, 10), 0, 1)),
        })
    result = dream(real, n_samples=12000, temperature=1.0, seed=42)
    print(f"  → {result.n_samples} synthetic sessions")


def example_api_logs():
    """API access logs — endpoint, status, latency."""
    print()
    print("=" * 64)
    print("  2. API access logs — 8000 entries, 6 columns")
    print("=" * 64)
    rng = np.random.default_rng(1)
    endpoints = ["/users", "/users/{id}", "/orders", "/orders/{id}", "/auth/login",
                 "/auth/refresh", "/products", "/search", "/health", "/metrics"]
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    real = []
    for _ in range(8000):
        status = int(rng.choice([200, 201, 204, 400, 401, 403, 404, 429, 500, 502],
                                p=[0.65, 0.10, 0.05, 0.04, 0.03, 0.02, 0.04, 0.02, 0.03, 0.02]))
        real.append({
            "endpoint":     str(rng.choice(endpoints)),
            "method":       str(rng.choice(methods, p=[0.65, 0.15, 0.05, 0.05, 0.10])),
            "status":       status,
            "latency_ms":   float(np.clip(rng.lognormal(2.5, 1.0), 1, 10000)),
            "bytes_out":    int(np.clip(rng.lognormal(7, 1.5), 0, 1_000_000)),
            "authenticated": bool(rng.random() < 0.78),
        })
    result = dream(real, n_samples=25000, temperature=1.0, seed=7)
    print(f"  → {result.n_samples} synthetic log entries")


def example_login_events():
    """Login attempts — success/fail, time, geo."""
    print()
    print("=" * 64)
    print("  3. Login events — 3000 attempts (with some fails)")
    print("=" * 64)
    rng = np.random.default_rng(2)
    failure_reasons = [None, "bad_password", "account_locked", "mfa_failed",
                       "captcha_failed", "rate_limited"]
    real = []
    for _ in range(3000):
        success = rng.random() < 0.92
        real.append({
            "success":        success,
            "failure_reason": "" if success else str(rng.choice(failure_reasons[1:])),
            "country":        str(rng.choice(["US", "CN", "RU", "DE", "BR", "IN", "NG", "UA"])),
            "ip_class":       str(rng.choice(["residential", "datacenter", "vpn", "tor", "mobile"],
                                              p=[0.55, 0.20, 0.12, 0.03, 0.10])),
            "attempt_seq":    int(np.clip(rng.geometric(0.5), 1, 20)),
            "duration_ms":    float(np.clip(rng.normal(450, 150), 50, 3000)),
            "mfa_used":       bool(rng.random() < 0.62),
        })
    result = dream(real, n_samples=10000, temperature=1.0, seed=12)
    print(f"  → {result.n_samples} synthetic login attempts")
    print(f"  → success rate: real={sum(r['success'] for r in real)/len(real):.1%}, "
          f"synth={sum(r['success'] for r in result.records)/result.n_samples:.1%}")


def example_permission_grants():
    """Permission grant audit — role, resource, scope."""
    print()
    print("=" * 64)
    print("  4. Permission grants — 1500 records, RBAC-style")
    print("=" * 64)
    rng = np.random.default_rng(3)
    roles = ["viewer", "editor", "admin", "owner", "guest"]
    resources = ["repo", "dashboard", "dataset", "model", "pipeline", "secret", "bucket"]
    actions = ["read", "write", "delete", "share", "admin"]
    real = []
    for _ in range(1500):
        real.append({
            "role":          str(rng.choice(roles, p=[0.40, 0.30, 0.15, 0.05, 0.10])),
            "resource_type": str(rng.choice(resources)),
            "action":        str(rng.choice(actions, p=[0.40, 0.30, 0.10, 0.10, 0.10])),
            "granted":       bool(rng.random() < 0.78),
            "ttl_seconds":   int(np.clip(rng.lognormal(11, 1.5), 60, 31536000)),
        })
    result = dream(real, n_samples=6000, temperature=1.0, seed=8)
    print(f"  → {result.n_samples} synthetic permission grants")


def example_security_incidents():
    """Security incident tickets — severity, type, time-to-resolve."""
    print()
    print("=" * 64)
    print("  5. Security incidents — 400 tickets, heavy-tailed TTR")
    print("=" * 64)
    rng = np.random.default_rng(4)
    severities = ["low", "medium", "high", "critical"]
    types = ["phishing", "malware", "data_leak", "credential_stuffing", "ddos",
             "insider_threat", "supply_chain", "ransomware"]
    real = []
    for _ in range(400):
        sev = str(rng.choice(severities, p=[0.45, 0.35, 0.15, 0.05]))
        # Higher severity → larger TTR-skewing factor
        ttr_factor = {"low": 1, "medium": 3, "high": 12, "critical": 48}[sev]
        real.append({
            "severity":         sev,
            "type":             str(rng.choice(types)),
            "ttr_hours":        float(np.clip(rng.lognormal(np.log(ttr_factor), 1.0), 0.1, 720)),
            "false_positive":   bool(rng.random() < 0.18),
            "n_affected_users": int(np.clip(rng.lognormal(1.5, 2), 0, 100_000)),
            "automated_alert":  bool(rng.random() < 0.65),
        })
    result = dream(real, n_samples=1500, temperature=1.0, seed=11)
    print(f"  → {result.n_samples} synthetic incidents")


def main():
    example_user_sessions()
    example_api_logs()
    example_login_events()
    example_permission_grants()
    example_security_incidents()


if __name__ == "__main__":
    main()
