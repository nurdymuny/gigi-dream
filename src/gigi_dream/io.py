"""I/O helpers for gigi-dream CLI.

Read/write tabular files in CSV, JSON, JSONL formats using only the standard
library. Parquet support requires pandas; falls back gracefully if missing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


def read_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read tabular records from a file. Format auto-detected from extension.

    Parameters
    ----------
    path : str or Path
        Path to a ``.csv``, ``.json``, ``.jsonl`` (or ``.ndjson``), or
        ``.parquet`` file.

    Returns
    -------
    list of dict
        Records as a list of dicts (one per row).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {path}")

    ext = p.suffix.lower()
    if ext == ".csv":
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [_coerce_numeric_values(row) for row in reader]

    if ext == ".json":
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [dict(r) for r in data]
        if isinstance(data, dict) and "records" in data:
            return [dict(r) for r in data["records"]]
        raise ValueError(
            f"unsupported JSON shape; expected list of objects or "
            f"{{'records': [...]}}, got top-level {type(data).__name__}"
        )

    if ext in (".jsonl", ".ndjson"):
        records = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if ext == ".parquet":
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(
                "Parquet support requires pandas. Install with: pip install gigi-dream[parquet]"
            ) from e
        df = pd.read_parquet(p)
        return df.to_dict(orient="records")

    raise ValueError(f"unsupported file extension: {ext!r}")


def write_records(
    records: Sequence[Dict[str, Any]],
    path: Union[str, Path],
    *,
    format: str = None,
) -> None:
    """Write records to a file. Format auto-detected from extension, or override.

    Parameters
    ----------
    records : sequence of dict
        Records to write.
    path : str or Path
        Output path. If ``-``, writes to stdout (CSV format unless ``format=``
        is given explicitly).
    format : str, optional
        Override the format. One of ``"csv"``, ``"json"``, ``"jsonl"``,
        ``"parquet"``. If not given, inferred from the file extension.
    """
    p = Path(path) if str(path) != "-" else None

    if format is None:
        if p is None:
            format = "csv"
        else:
            ext = p.suffix.lower().lstrip(".")
            format = ext if ext else "csv"
            if format == "ndjson":
                format = "jsonl"

    if format == "csv":
        _write_csv(records, p)
    elif format == "json":
        _write_json(records, p)
    elif format == "jsonl":
        _write_jsonl(records, p)
    elif format == "parquet":
        _write_parquet(records, p)
    else:
        raise ValueError(f"unsupported format: {format!r}")


# ── Format-specific writers ─────────────────────────────────────────────────


def _open_or_stdout(path):
    """Open a Path for write, or return sys.stdout if path is None."""
    import sys

    if path is None:
        return sys.stdout
    return open(path, "w", newline="", encoding="utf-8")


def _write_csv(records, path) -> None:
    if not records:
        return
    columns = list(records[0].keys())
    f = _open_or_stdout(path)
    try:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    finally:
        if path is not None:
            f.close()


def _write_json(records, path) -> None:
    f = _open_or_stdout(path)
    try:
        json.dump(list(records), f, indent=2, default=str)
    finally:
        if path is not None:
            f.close()


def _write_jsonl(records, path) -> None:
    f = _open_or_stdout(path)
    try:
        for r in records:
            f.write(json.dumps(r, default=str))
            f.write("\n")
    finally:
        if path is not None:
            f.close()


def _write_parquet(records, path) -> None:
    if path is None:
        raise ValueError("parquet output requires a file path (cannot write to stdout)")
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "Parquet output requires pandas. Install with: pip install gigi-dream[parquet]"
        ) from e
    df = pd.DataFrame(list(records))
    df.to_parquet(path, index=False)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _coerce_numeric_values(row: Dict[str, Any]) -> Dict[str, Any]:
    """Try to coerce CSV string values to numeric. Leave non-numeric as strings."""
    out = {}
    for k, v in row.items():
        if v is None or v == "":
            out[k] = v
            continue
        if isinstance(v, str):
            try:
                if "." in v or "e" in v or "E" in v:
                    out[k] = float(v)
                else:
                    out[k] = int(v)
                continue
            except (ValueError, TypeError):
                pass
        out[k] = v
    return out
