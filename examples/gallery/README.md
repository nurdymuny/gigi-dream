# gigi-dream examples gallery

**47 worked examples across 8 industries.** Each example shows gigi-dream applied to a different shape of data, problem type, or use case. Together they cover the full range of where the v0 LocalBackend works well and where its limitations show.

## Run them all

```bash
# Run a single domain's examples
python examples/gallery/healthcare.py
python examples/gallery/finance.py
# ... etc.

# Run all 47 examples in one shot
python examples/gallery/run_all.py
```

## What's in the gallery

| File | Examples | Domain |
|------|---------:|--------|
| [`healthcare.py`](healthcare.py) | 6 | EHR · lab panels · claims · clinical trials · ICU vitals · ER admissions |
| [`finance.py`](finance.py) | 6 | bank txns · portfolios · loan apps · card spend · OHLCV · fraud features |
| [`ecommerce.py`](ecommerce.py) | 6 | orders · customer profiles · catalog · reviews · cart events · returns |
| [`iot.py`](iot.py) | 5 | industrial sensors · smart home · weather · vehicle · agriculture |
| [`logistics.py`](logistics.py) | 5 | shipments · warehouse · routes · delivery perf · fleet |
| [`ml.py`](ml.py) | 5 | training-set aug · tabular features · classification · time-series feats · embeddings |
| [`auth_security.py`](auth_security.py) | 5 | sessions · API logs · logins · permissions · incidents |
| [`ops_sre.py`](ops_sre.py) | 5 | latencies · error rates · deploys · service health · incident tickets |
| [`scientific.py`](scientific.py) | 4 | lab experiments · particle physics · astronomy · genomics |
| **Total** | **47** | |

## Variety covered

Across all 47 examples:

- **Scale**: from 200 records (small training sets) to 30,000+ records (industrial sensors). Largest output: 25,000 synthetic API log entries.
- **Width**: from 5 columns (cart events) to 20+ columns (ML tabular features, lab panels with 14 analytes).
- **Types**: numeric (continuous + integer + percentages), categorical (strings + enums + IDs), boolean, mixed.
- **Distributions**: Gaussian, log-normal, exponential, gamma, beta, Poisson, geometric, uniform, multimodal.
- **Temperature**: T=1.0 (faithful, default), T=1.5 (mild augmentation), T=2.0 (DREAM mode), T=4.0 (high novelty).
- **Use cases**: test fixtures · dev DBs · staging · privacy-aware demos · ML training-set augmentation · minority-class oversampling · capacity-planning simulators · ETL test data.

## Limitations visible across the gallery

The examples honestly surface gigi-dream v0's known limitations:

- **No correlation preservation** — see `ml.example_embedding_like_data` (unit-norm constraint lost), `finance.example_market_data` (OHLC ordering lost), `ops_sre.example_latency_percentiles` (percentile ordering lost).
- **Per-column independence** — see `ecommerce.example_cart_events` (event/device shape preserved per-column but no joint event-device patterns).
- **No structural constraints** — `scientific.example_genomic_variants` notes that synthetic positions can fall outside real chromosome lengths.

For all of these, the comment in the example points at `GigiBackend` as the higher-fidelity path — GIGI's `/brain/dream` endpoint uses the engine's full Kähler-aware Welford fit with L13.3 diagonal-Gaussian support and L13.7 denominator floor for proper anisotropic and constrained sampling.

## Status

All 47 examples are pure-Python, depend only on numpy + gigi-dream, and run in a few seconds total against `LocalBackend`. Useful both as a guided tour of gigi-dream and as a smoke-test of the LocalBackend across diverse data shapes.
