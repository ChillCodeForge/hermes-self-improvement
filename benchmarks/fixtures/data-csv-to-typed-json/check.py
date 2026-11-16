import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import json

EXPECTED = [
    {"sku": "A-100", "name": "Hex bolt M8", "quantity": 1200, "price": 0.15, "discontinued": False},
    {"sku": "A-105", "name": "Washer M8", "quantity": 800, "price": 0.05, "discontinued": False},
    {"sku": "B-200", "name": "Steel bracket", "quantity": 40, "price": 3.25, "discontinued": False},
    {"sku": "C-310", "name": "Hinge, brass", "quantity": 0, "price": 7.5, "discontinued": True},
]


def body(checks):
    data = json.loads(read("inventory.json"))
    checks.check("is_array", isinstance(data, list))
    checks.check("records_equal", data == EXPECTED)
    checks.check(
        "types_exact",
        all(type(row["quantity"]) is int and type(row["price"]) is float
            and type(row["discontinued"]) is bool for row in data),
    )
    checks.check("csv_unchanged", read("inventory.csv").startswith("sku,name,quantity,price,discontinued\n"))


run(body)
