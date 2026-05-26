"""Run every example in the gigi-dream gallery (all 47)."""

from __future__ import annotations

import sys
import time

# Make sure unicode arrows print on Windows consoles (cp1252 default).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import healthcare
import finance
import ecommerce
import iot
import logistics
import ml
import auth_security
import ops_sre
import scientific


SECTIONS = [
    ("HEALTHCARE",     healthcare,     6),
    ("FINANCE",        finance,        6),
    ("E-COMMERCE",     ecommerce,      6),
    ("IOT / SENSORS",  iot,            5),
    ("LOGISTICS",      logistics,      5),
    ("ML / DATA SCI",  ml,             5),
    ("AUTH / SEC",     auth_security,  5),
    ("OPS / SRE",      ops_sre,        5),
    ("SCIENTIFIC",     scientific,     4),
]


def main():
    start = time.time()
    total = 0
    for name, module, count in SECTIONS:
        print()
        print("#" * 72)
        print(f"#  {name}  ({count} examples)")
        print("#" * 72)
        module.main()
        total += count

    elapsed = time.time() - start
    print()
    print("=" * 72)
    print(f"  ran {total} examples in {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
    main()
