# gigi-dream

> **Synthetic data that looks like your real data, without being your real data.**
> One brain primitive from the [GIGI](https://davisgeometric.com) engine, made into a pip-installable tool.

```python
from gigi_dream import dream

real_customers = [
    {"age": 30, "country": "US", "salary": 75000},
    {"age": 45, "country": "CA", "salary": 95000},
    {"age": 28, "country": "US", "salary": 68000},
    # ... 100 more real records ...
]

result = dream(real_customers, n_samples=1000, temperature=1.0, seed=42)

print(result.records[0])
# {"age": 32.7, "country": "US", "salary": 73210.3}
```

Three lines of code. A thousand new customers that share your data's statistical fingerprint but aren't any actual person.

---

## What this is, in plain words

You give `dream()` a list of records. It learns the shape of each column — *how* numbers are distributed, *how often* each category appears — and then samples new records from that shape.

The result is fake data that **feels right**. The ages have the right spread. The countries appear in the right proportions. The salaries cluster where your real salaries cluster. But every record is invented.

This is useful any time you need data that *behaves* like your real data but can't *be* your real data:

| Use case | What `gigi-dream` gives you |
|---|---|
| **Test fixtures** | Records that exercise the same edge cases prod has, without copying prod |
| **Dev environments** | A populated database that looks plausible without scrubbing real users |
| **Staging demos** | Anonymized data with statistically faithful behavior for sales calls |
| **ML augmentation** | Extra training records sampled from the empirical density of your dataset |
| **Privacy-conscious onboarding** | New hires can explore data shape without ever seeing real PII |
| **Capacity simulators** | Thousands of synthetic transactions to load-test your system |
| **Schema rehearsal** | Try a migration against realistic data without copying production |

The 47-example [gallery](examples/gallery/) shows it working across **healthcare, finance, e-commerce, IoT, logistics, ML, security, ops, and scientific** datasets — from 200 records up to 30,000+, from 4 columns to 20+, with every common distribution shape you'll encounter in the wild.

---

## See it work

Install it:

```bash
pip install gigi-dream
```

Now you can already do this:

```python
from gigi_dream import dream

# Real ER admissions: short stays mostly, with a long tail
import random
random.seed(0)
real_admissions = [
    {
        "age": random.randint(18, 95),
        "triage": random.choice(["red", "orange", "yellow", "green", "blue"]),
        "los_hours": max(0.5, random.gammavariate(1.5, 4)),  # length-of-stay
        "admitted": random.random() < 0.18,
    }
    for _ in range(500)
]

result = dream(real_admissions, n_samples=5000, temperature=1.0, seed=42)

# Real admission rate vs synthetic admission rate
real_rate = sum(r["admitted"] for r in real_admissions) / len(real_admissions)
synth_rate = sum(r["admitted"] for r in result.records) / result.n_samples
print(f"real admission rate: {real_rate:.1%}, synthetic: {synth_rate:.1%}")
# real admission rate: 17.8%, synthetic: 18.1%

# Real average LOS vs synthetic
real_los = sum(r["los_hours"] for r in real_admissions) / len(real_admissions)
synth_los = sum(r["los_hours"] for r in result.records) / result.n_samples
print(f"real avg LOS: {real_los:.1f}h, synthetic: {synth_los:.1f}h")
# real avg LOS: 6.0h, synthetic: 6.1h
```

You started with 500 real records and ended with 5,000 plausible new ones — and the broad statistics match closely enough that anything downstream that depends on them (dashboards, ML models, capacity simulators) keeps working.

---

## The CLI

If you'd rather work from files than Python:

```bash
# Read a CSV, write 1000 synthetic rows to another CSV
gigi-dream customers.csv -n 1000 -o test_customers.csv

# Crank up the temperature — wider spread, more novel-feeling records
gigi-dream customers.csv -n 1000 -T 3.0 -o exotic_customers.csv

# Pipe to stdout, peek at the head
gigi-dream customers.csv -n 100 | head

# JSON output instead of CSV
gigi-dream customers.csv -n 100 --format json -o synth.json

# Reproducible — seed makes the run deterministic
gigi-dream customers.csv -n 100 --seed 42 -o snapshot.csv

# Inspect the column distributions without sampling
gigi-dream customers.csv --inspect
```

Supported in/out: `.csv`, `.json`, `.jsonl`/`.ndjson`, `.parquet` (with the `[parquet]` extra).

---

## The math, explained

You don't need a stats degree to use this — but if you want to know what's actually happening under the hood, it's elegant and small enough to fit on one screen.

### Numeric columns: Welford's running-mean variance

For every numeric column, we compute the **mean (μ)** and **standard deviation (σ)** of your data. The clever trick is *how*: Welford's online algorithm computes both quantities in a **single pass through your data, in constant memory**, with no numerical drift even on billions of records.

In math:

$$M_2^{(n)} = M_2^{(n-1)} + (x_n - \bar{x}_{n-1})(x_n - \bar{x}_n) \qquad \sigma^2 = \frac{M_2}{n}$$

That's the recurrence — each new value updates the running second-moment accumulator $M_2$, and at the end, dividing by $n$ gives the variance. It's a 50-year-old result (Welford, 1962) that you'll find in NIST's reference implementations and the standard library of every serious numerical computing language. Once we have μ and σ, sampling is just:

$$x_{\text{new}} = \mu + \sqrt{T} \cdot \sigma \cdot \mathcal{N}(0, 1)$$

The $\sqrt{T}$ is the **temperature knob**. At $T = 1$, you get a distribution with exactly the same spread as your real data. Crank $T$ up and the distribution gets wider; drop it down and samples cluster tightly near the mean.

### Why temperature is a knob (and not just a hack)

This is the cute part. The temperature comes from the **Friston master equation** that governs all twelve of GIGI's brain primitives. In its dissipative form:

$$\dot{x} = -\nabla H(x) \, dt + \sqrt{2T} \, dW$$

That equation describes how a particle wanders on an energy landscape $H$. The $-\nabla H$ term pulls it toward the minimum (the mean). The $\sqrt{2T}\,dW$ term is noise — random kicks scaled by the **temperature** $T$. High $T$ = more wandering. Low $T$ = stays near the mean.

For a quadratic $H(x) = (x - \mu)^2 / (2\sigma^2)$ — the energy landscape of a Gaussian — solving this equation at equilibrium gives exactly:

$$p(x) \propto e^{-(x - \mu)^2 / (2T\sigma^2)}$$

…which is a Gaussian centered at $\mu$ with **width $\sqrt{T}\sigma$**.

So when you write `temperature=2.0`, you're literally heating up the imaginary thermal bath that the sampler lives in. Same equation that runs the rest of GIGI's brain primitives — just specialized to the single-column case.

### Categorical columns: empirical frequency

For string, boolean, and other categorical columns, the model is even simpler: count how often each value appears in your real data, and sample new values weighted by those counts. So if your real data is 78% `is_active = True`, your synthetic data will be ~78% `is_active = True` too.

### Tuning temperature

| Temperature | What you get |
|---|---|
| `T = 1.0` (default) | Synthetic ≈ real. Same μ, same σ, same range. |
| `T = 2.0 – 4.0` | "DREAM mode." Spread is √T wider. Useful for ML augmentation — you generate plausible-but-novel records to expand a small training set. |
| `T = 0.3 – 0.7` | Tight samples near the mean. Useful for "typical case" demos where you want the data to look representative, not extreme. |
| `T = 0` | Every sample equals the per-column mean. Degenerate, but occasionally useful as a baseline. |

---

## What you get from the standalone package

The pure-numpy `LocalBackend` gives you fast, transparent, O(1)-memory synthetic data with these guarantees:

- **Per-column means** held to within tolerance set by sample size
- **Per-column standard deviations** scaled by $\sqrt{T}$
- **Categorical frequency distributions** preserved (proportions of each value)
- **Type stability** — strings stay strings, ints stay ints, bools stay bools
- **Reproducibility** under the same `seed`

That's enough for **test fixtures, dev databases, staging demos, capacity simulators, and ML augmentation on independent features** — the daily-driver use cases.

## Want more? Upgrade to the GIGI engine

The standalone `LocalBackend` is the simplest possible specialization of GIGI's `DREAM` primitive — per-column, independent, diagonal. Pointing at a running GIGI instance via `GigiBackend` (next section) unlocks the engine's full **Kähler-aware Welford fit**, which adds:

| Need | What the GIGI engine gives you |
|---|---|
| **Correlated columns** | Joint sampling that respects inter-column structure (so `salary` stays correlated with `age`) |
| **Joint patterns** | "If A is X then B tends to be Y" preserved in samples, via the full Kähler form |
| **Hard constraints** | `H ≥ O ≥ L` for OHLC, unit-norm vectors, monotonic sequences — preserved at sample time, not as post-hoc filtering |
| **Time series fidelity** | Autocorrelation, seasonality, and trend preserved (sampling on the base manifold instead of i.i.d.) |
| **High-dimensional data** | L13.3 diagonal-Gaussian + L13.7 denominator-floor stability for anisotropic distributions |
| **Multimodal distributions** | Mixture-of-Kähler-Gaussians for data with multiple natural clusters |

Each of these is **already built** in the engine. The standalone package is the entry point; GIGI is where the full version lives. Same math, just specialized down to the column-independent case for the pip-install-and-go path.

---

## Two backends

### `LocalBackend` (default — pure numpy)

This is what runs when you call `dream(records)` with no backend argument. No external services, no network calls. Use this 99% of the time.

```python
from gigi_dream import dream, LocalBackend

result = dream(real_records, backend=LocalBackend())
```

### `GigiBackend` (the full engine)

When you want the **full Kähler-aware fit** with correlation preservation and anisotropic sampling, point at a running GIGI instance:

```python
from gigi_dream import dream, GigiBackend

backend = GigiBackend(
    url="http://localhost:3142",
    api_key="dev-local",
    bundle="customers",
    fields=["age", "salary"],
)
result = dream(n_samples=1000, backend=backend)
```

The backend will call GIGI's `/brain/dream` endpoint, which uses the engine's full geometric machinery — much higher-fidelity sampling, but you need a GIGI instance running.

---

## Different tool categories (just so you land in the right place)

If you came here looking for one of these instead, here's where to actually go:

- **Need formal ε-differential-privacy guarantees?** That's a different mathematical commitment than statistical faithfulness. Look at [`diffprivlib`](https://github.com/IBM/differential-privacy-library) or [`tumult-analytics`](https://gitlab.com/tumult-labs/analytics). (GIGI itself can be composed with DP machinery — these standalone packages just don't ship DP guarantees in the box.)
- **Need relational / multi-table synthesis with FK constraints?** Single tables only in the v0 standalone. The DHOOM format and GIGI's bundle model handle nested fiber-bundle relationships natively — that's a doorway into the engine, not a missing feature.
- **Want GAN- or diffusion-based synthesis?** `gigi-dream` is deliberately math-first, not model-first — small, fast, transparent. If that's not the trade-off you want, [SDV](https://github.com/sdv-dev/SDV) and [`ydata-synthetic`](https://github.com/ydataai/ydata-synthetic) take the GAN approach.

---

## About GIGI — the engine `gigi-dream` is a window into

`gigi-dream` is one brain primitive (`DREAM`) extracted from **GIGI**, a geometric database engine that models data as **fiber bundles over a base manifold**. The full engine has **twelve brain primitives**, all unified by the Friston master equation on a Kähler bundle:

| Primitive | What it does |
|---|---|
| `SAMPLE` | Draw samples from a fitted distribution |
| `FORECAST` | Predict future values of a time series |
| **`DREAM`** ← this package | Generate synthetic records from learned column distributions |
| `RECONSTRUCT` | Fill in missing fields from partial records |
| `INPAINT` | Reconstruct masked regions of structured data |
| `PREDICT` | Single-step prediction from current state |
| `ATTEND` | Compute attention/importance weights over fields |
| `FOCUS` | Drill down on a subset of the bundle |
| `EPISODIC` | Detect change-points and regime shifts ([see `gigi-episodes`](https://pypi.org/project/gigi-episodes/)) |
| `SEMANTIC` | Retrieve records by meaning/similarity |
| `SELF-MONITOR` | Compute geometric health metrics (curvature, etc.) |
| `EXPLAIN` | Produce a natural-language summary of bundle state |

Beyond the brain primitives, GIGI provides:

- 🧠 **Persistent structured memory** with a schema that survives serialization (via the DHOOM format)
- 📐 **Scalar curvature K** as a "geometric health score" for any bundle
- 🌐 **GIGI Query Language** (GQL — a SQL-flavored DSL, not GraphQL) for filtering, aggregating, and transporting fiber-bundle data
- 🔄 **Real-time WebSocket subscriptions** to bundle mutations
- 📊 **Live demo** at [gigi-stream.fly.dev](https://gigi-stream.fly.dev/v1/health) currently hosting **4,961 bundles and 12.8 million records**

### GIGI is free

Per [Davis Geometric's licensing philosophy](https://davisgeometric.com):

> *"Free for the people who use it to learn; supported by the companies that ship products with it."*

- 🆓 **Free for research, education, and non-commercial use.**
- 💼 **Commercial deployments are patent-protected** (US Provisional Patent 64/045,889) — contact for licensing.
- 🏛️ **Patented commercial-tier operations** (curvature, spectral, holonomy, transport) return `LICENSE_REQUIRED` for non-commercial callers.

Read about the math: [davisgeometric.com](https://davisgeometric.com) · The engine: [github.com/nurdymuny/gigi](https://github.com/nurdymuny/gigi)

---

## Sibling packages

`gigi-dream` is part of a family of small, focused brain primitives extracted from GIGI:

- [**`gigi-episodes`**](https://pypi.org/project/gigi-episodes/) — change-point detection (the `EPISODIC` primitive)
- [**`gigi-mcp`**](https://pypi.org/project/gigi-mcp/) — Model Context Protocol server, lets Claude query GIGI directly
- [**`gigi-client`**](https://pypi.org/project/gigi-client/) — Python SDK for the GIGI engine (HTTP + WebSocket)

Each one stands alone and works without the others. Together, they're the "scattered seeds" of GIGI — small enough to try in 30 seconds, useful even if you never adopt the full engine.

---

## License

MIT. Free for any use, commercial or otherwise. See [LICENSE](LICENSE).

(Note: this package's MIT license is unconditional. GIGI itself, which the `GigiBackend` connects to, has the dual license described above — the `LocalBackend` you get from `pip install gigi-dream` has no such restrictions.)

---

## Status

**v0.1.0** — stable for the documented surface (CSV/JSON/JSONL/Parquet I/O + `LocalBackend` + CLI + `GigiBackend` skeleton). API may evolve through the 0.x series; will stabilize at 1.0.

Issues, ideas, and pull requests: [github.com/nurdymuny/gigi-dream](https://github.com/nurdymuny/gigi-dream/issues)

Built with care by [Bee Rosa Davis](https://davisgeometric.com) / [Davis Geometric](https://davisgeometric.com). 💛
