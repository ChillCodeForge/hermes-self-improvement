import csv
from datetime import datetime

FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%Y.%m.%d")


def normalize(value):
    for pattern in FORMATS:
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unsupported date: {value}")


with open("orders.csv", newline="", encoding="utf-8") as source:
    reader = csv.DictReader(source)
    rows = list(reader)
    fields = reader.fieldnames
with open("orders_clean.csv", "w", newline="", encoding="utf-8") as target:
    writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row | {"order_date": normalize(row["order_date"])})
