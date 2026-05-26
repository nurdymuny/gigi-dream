"""Tests for the gigi_dream CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gigi_dream.cli import main


def _write_csv(path: Path, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def _make_input_csv(tmp_path):
    """Create a small mixed-column CSV for CLI tests."""
    records = []
    for i in range(50):
        records.append({
            "age": 30 + i % 20,
            "country": ["US", "CA", "UK"][i % 3],
            "salary": 50000 + (i * 1000),
        })
    path = tmp_path / "input.csv"
    _write_csv(path, records)
    return path


def test_cli_basic_csv_to_csv(tmp_path, capsys):
    """Basic CSV in, CSV out."""
    input_path = _make_input_csv(tmp_path)
    output_path = tmp_path / "out.csv"
    rc = main([str(input_path), "-n", "20", "-o", str(output_path), "--seed", "42"])
    assert rc == 0
    assert output_path.exists()
    with output_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    assert set(rows[0].keys()) == {"age", "country", "salary"}


def test_cli_json_output(tmp_path):
    """CSV in, JSON out via --format."""
    input_path = _make_input_csv(tmp_path)
    output_path = tmp_path / "out.json"
    rc = main([str(input_path), "-n", "10", "--format", "json", "-o", str(output_path)])
    assert rc == 0
    data = json.loads(output_path.read_text())
    assert len(data) == 10


def test_cli_jsonl_output(tmp_path):
    """JSONL format produces one JSON object per line."""
    input_path = _make_input_csv(tmp_path)
    output_path = tmp_path / "out.jsonl"
    rc = main([str(input_path), "-n", "8", "--format", "jsonl", "-o", str(output_path)])
    assert rc == 0
    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == 8
    for line in lines:
        json.loads(line)  # each line is valid JSON


def test_cli_inspect_mode(tmp_path, capsys):
    """--inspect prints the fit summary without sampling."""
    input_path = _make_input_csv(tmp_path)
    rc = main([str(input_path), "--inspect"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "numeric" in out
    assert "categorical" in out
    assert "country" in out
    assert "salary" in out


def test_cli_missing_input_errors(capsys):
    """Specifying a nonexistent file produces a clear error and nonzero exit."""
    rc = main(["/no/such/file.csv", "-n", "10"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "file" in err.lower() or "not found" in err.lower()


def test_cli_temperature_flag(tmp_path):
    """The -T flag passes temperature through."""
    input_path = _make_input_csv(tmp_path)
    output_path = tmp_path / "out.csv"
    rc = main([str(input_path), "-n", "30", "-T", "3.0", "-o", str(output_path), "--seed", "7"])
    assert rc == 0
    assert output_path.exists()


def test_cli_seed_makes_output_reproducible(tmp_path):
    """Same --seed → identical output."""
    input_path = _make_input_csv(tmp_path)
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    rc_a = main([str(input_path), "-n", "20", "-o", str(out_a), "--seed", "999"])
    rc_b = main([str(input_path), "-n", "20", "-o", str(out_b), "--seed", "999"])
    assert rc_a == 0 and rc_b == 0
    assert out_a.read_text() == out_b.read_text()
