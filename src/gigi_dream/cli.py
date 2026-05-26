"""gigi-dream CLI — generate synthetic data from a tabular file.

Usage examples::

    # Generate 1000 synthetic records from a CSV
    gigi-dream customers.csv -n 1000 > test_customers.csv

    # Higher temperature = more novel records
    gigi-dream customers.csv -n 1000 -T 3.0 -o diverse.csv

    # Explicit output format
    gigi-dream data.parquet -n 500 --format jsonl -o synthetic.jsonl

    # Reproducible (with seed)
    gigi-dream data.csv -n 100 --seed 42 -o output.csv

    # Just inspect the fitted distribution (no sampling)
    gigi-dream customers.csv --inspect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .algorithm import DreamResult, fit_columns
from .core import dream
from .io import read_records, write_records


def _format_inspect(records, fits) -> str:
    """Pretty-print a fit summary without sampling."""
    lines = [
        f"  input records: {len(records)}",
        f"  columns:       {len(fits)}",
        "",
        "  per-column fit:",
    ]
    for f in fits:
        if f.kind == "numeric":
            lines.append(
                f"    {f.name:24}  numeric     mean={f.mean:10.4f}  sigma={f.sigma:10.4f}  n={f.n_observed}"
            )
        else:
            distinct = len(f.values) if f.values else 0
            lines.append(
                f"    {f.name:24}  categorical distinct={distinct}  n={f.n_observed}"
            )
            if f.values:
                top = sorted(zip(f.values, f.weights), key=lambda kv: -kv[1])[:5]
                for val, w in top:
                    lines.append(f"      • {str(val)[:30]:30}  {w:6.2%}")
    return "\n".join(lines)


def _format_summary(result: DreamResult, source: str, output: str) -> str:
    """Pretty-print a sampling-run summary."""
    lines = [
        f"  source:      {source}",
        f"  output:      {output}",
        f"  backend:     {result.backend}",
        f"  temperature: {result.temperature}",
        f"  n_samples:   {result.n_samples}",
        f"  columns:     {len(result.columns)}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] = None) -> int:
    """Entry point for the ``gigi-dream`` console script."""
    parser = argparse.ArgumentParser(
        prog="gigi-dream",
        description="Synthetic data generation via GIGI's DREAM brain primitive. "
        "Built on https://davisgeometric.com",
    )
    parser.add_argument(
        "input", help="Path to input file (.csv, .json, .jsonl, .parquet)"
    )
    parser.add_argument(
        "-n", "--num", type=int, default=100,
        help="Number of synthetic records to generate. Default: 100",
    )
    parser.add_argument(
        "-T", "--temperature", type=float, default=1.0,
        help="Sampling temperature. 1.0 = faithful; >1.0 = DREAM (novel). Default: 1.0",
    )
    parser.add_argument(
        "-o", "--output", default="-",
        help="Output path, or '-' for stdout. Default: '-'",
    )
    parser.add_argument(
        "--format", choices=["csv", "json", "jsonl", "parquet"],
        help="Output format override. If not given, inferred from output extension.",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Only print the fitted column distributions — don't generate samples.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the summary text printed to stderr.",
    )

    args = parser.parse_args(argv)

    # Load input
    try:
        records = read_records(args.input)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not records:
        print("error: input file has no records", file=sys.stderr)
        return 3

    # Inspect mode: no sampling, just print fit
    if args.inspect:
        fits = fit_columns(records)
        print(_format_inspect(records, fits))
        return 0

    # Sample
    result = dream(
        records,
        n_samples=args.num,
        temperature=args.temperature,
        seed=args.seed,
    )

    # Write
    try:
        write_records(result.records, args.output, format=args.format)
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 4

    if not args.quiet:
        print(_format_summary(result, args.input, args.output), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
