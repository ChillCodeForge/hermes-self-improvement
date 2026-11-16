import csv
import json

with open("inventory.csv", newline="", encoding="utf-8") as source:
    rows = [
        {
            "sku": row["sku"].strip(),
            "name": row["name"].strip(),
            "quantity": int(row["quantity"]),
            "price": float(row["price"]),
            "discontinued": row["discontinued"].strip() == "yes",
        }
        for row in csv.DictReader(source)
    ]
rows.sort(key=lambda row: row["sku"])
with open("inventory.json", "w", encoding="utf-8") as target:
    json.dump(rows, target, indent=2)
    target.write("\n")
