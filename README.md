# gigi-dream

**Synthetic data generation for test fixtures, dev environments, and privacy-aware demos.** Statistically faithful records that aren't real records.

```python
from gigi_dream import dream

real_customers = [
    {"age": 30, "country": "US", "salary": 75000},
    {"age": 45, "country": "CA", "salary": 95000},
    {"age": 28, "country": "US", "salary": 68000},
    # ... 100 more ...
]

result = dream(real_customers, n_samples=1000, temperature=1.0, seed=42)
print(result.records[0])
# {"age": 32.7, "country": "US", "salary": 73210.3}
```

```bash
$ gigi-dream customers.csv -n 1000 -o test_customers.csv
  source:      customers.csv
  output:      test_customers.csv
  backend:     local
  temperature: 1.0
  n_samples:   1000
  columns:     5
```

## What it's for

Anywhere you need data that *looks like your real data* but isn't your real data:

- **Test fixtures** — populate test databases with records that exercise edge cases
- **Dev environments** — stop hand-rolling fake data; learn it from prod
- **Staging** — anonymized demos with statistically faithful behavior
- **ML augmentation** — extra training records sampled from the empirical density
- **Privacy-conscious onboarding** — let new hires explore data shape without seeing real PII

gigi-dream is intentionally narrow: **per-column distribution sampling, nothing else.** Other "DREAM" features (multivariate, correlated, anisotropic, fiber-bundle native) live in the [GIGI engine](https://davisgeometric.com) — gigi-dream exposes one specific brain primitive as the smallest possible installable tool.

## Install

```bash
pip install gigi-dream
```

Optional: install with GIGI backend (requires `requests`):

```bash
pip install "gigi-dream[gigi]"
```

Optional: install with Parquet support (requires pandas + pyarrow):

```bash
pip install "gigi-dream[parquet]"
```

## Quick start

### Library

```python
from gigi_dream import dream

# Learn the distribution from real data
real = [
    {"age": 30, "country": "US", "salary": 75000},
    {"age": 45, "country": "CA", "salary": 95000},
    {"age": 28, "country": "US", "salary": 68000},
    {"age": 51, "country": "UK", "salary": 110000},
    # ... more records ...
]

# Generate 1000 synthetic records at temperature 1.0 (faithful)
result = dream(real, n_samples=1000, temperature=1.0, seed=42)

# Inspect what was learned
for col in result.columns:
    if col.kind == "numeric":
        print(f"  {col.name}: numeric  mean={col.mean:.1f} sigma={col.sigma:.1f}")
    else:
        print(f"  {col.name}: categorical {len(col.values)} values")

# Use the synthetic records anywhere you'd use real ones
for r in result.records[:5]:
    print(r)
```

### CLI

```bash
# Generate 1000 synthetic CSV records
gigi-dream customers.csv -n 1000 -o test_customers.csv

# Higher temperature = wider spread, more novel records
gigi-dream customers.csv -n 1000 -T 3.0 -o exotic_customers.csv

# Output to stdout for piping into other tools
gigi-dream customers.csv -n 100 | head

# Output JSON instead of CSV
gigi-dream customers.csv -n 100 --format json -o synth.json

# Reproducible — same seed gives same output
gigi-dream customers.csv -n 100 --seed 42 -o snapshot.csv

# Just inspect the column distributions, don't sample
gigi-dream customers.csv --inspect
```

Supported input formats: `.csv`, `.json`, `.jsonl` / `.ndjson`, `.parquet` (with `[parquet]` extra).
Supported output formats: same.

## Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--num` / `-n` | 100 | Number of synthetic records |
| `--temperature` / `-T` | 1.0 | 1.0 = faithful; > 1.0 = wider; < 1.0 = tighter |
| `--seed` | none | Reproducibility |

**Temperature notes:**
- `T = 1.0` — synthetic distribution matches the real one (~variance, ~range)
- `T = 2.0–4.0` — DREAM mode; ~1.4–2× wider spread; "novel-but-plausible"
- `T = 0.3–0.7` — synthesize tight samples near the mode; useful for "typical case" demos
- `T = 0` — every sample equals the per-column mean (degenerate)

## How it works (v0)

gigi-dream fits an **independent per-column model** to your input:

- **Numeric columns** → diagonal Gaussian with Welford-streamed mean and variance. Sample: `μ + √T × σ × N(0,1)`.
- **Categorical / string / boolean columns** → empirical frequency distribution. Sample: weighted choice from observed values.

Each column is sampled independently. **Correlations between columns are NOT preserved in v0.** If your data has strong inter-column structure (e.g., income correlates with age), use `GigiBackend` instead — GIGI's `/brain/dream` endpoint uses the engine's full Kähler-aware fit including the L13.3 diagonal-Gaussian variant of the [brain primitives](https://github.com/nurdymuny/gigi/blob/main/BRAIN_PRIMITIVES_CONSUMER_GUIDE.md).

## Two backends

**`LocalBackend`** (default) — pure-numpy, no infrastructure required. Use this 99% of the time.

```python
from gigi_dream import LocalBackend, dream
result = dream(real_records, backend=LocalBackend())
```

**`GigiBackend`** — calls a running GIGI instance's `/brain/dream` endpoint. Higher-fidelity sampling for anisotropic, correlated, or multivariate data. Useful when your data is already in a GIGI bundle.

```python
from gigi_dream import GigiBackend, dream

backend = GigiBackend(
    url="http://localhost:3142",
    api_key="dev-local",
    bundle="customers",
    fields=["age", "salary"],
)
result = dream(n_samples=1000, backend=backend)
```

## What gigi-dream isn't

- **Not a differential-privacy tool.** It provides *statistical faithfulness*, not formal DP guarantees. If you need ε-differential privacy, use a DP-specific library (e.g., `diffprivlib`, `tumult-analytics`).
- **Not a relational data generator.** Single tables only; no FK constraints, no schema relationships. (DHOOM supports nested bundles natively, so a future version could.)
- **Not a model-based synthesizer.** No GANs, no diffusion. The "model" is the per-column Welford fit. That's intentional — small, fast, transparent.

## License

MIT. Free for any use, commercial or otherwise. See [LICENSE](LICENSE).

## Related

- [GIGI](https://davisgeometric.com) — the fiber-bundle database engine; gigi-dream's `GigiBackend` calls it. DREAM is one of twelve [brain primitives](https://github.com/nurdymuny/gigi/blob/main/BRAIN_PRIMITIVES_CONSUMER_GUIDE.md).
- [EpisodeKit](https://github.com/nurdymuny/episodekit) — change-point detection using GIGI's EPISODIC primitive. Sibling project.
- [gigi-mind](https://github.com/nurdymuny/gigi-mind) — VS Code extension exposing all twelve brain primitives. Sibling project.

## Status

**v0.1.0** — stable for the documented surface (CSV/JSON/JSONL + LocalBackend + CLI + GigiBackend skeleton). API may evolve in 0.x; will stabilize at 1.0.
