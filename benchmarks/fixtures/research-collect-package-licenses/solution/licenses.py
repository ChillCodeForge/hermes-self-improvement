import json
from pathlib import Path

result = {}
for metadata in sorted(Path("vendor").glob("*/METADATA")):
    fields = {"Classifier": []}
    for line in metadata.read_text().splitlines():
        key, _, value = line.partition(": ")
        if key == "Classifier":
            fields["Classifier"].append(value)
        else:
            fields[key] = value
    license_name = fields.get("License")
    if not license_name or license_name == "UNKNOWN":
        classifiers = [item for item in fields["Classifier"] if item.startswith("License ::")]
        license_name = classifiers[0].split(" :: ")[-1]
    result[fields["Name"]] = license_name
Path("licenses.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
