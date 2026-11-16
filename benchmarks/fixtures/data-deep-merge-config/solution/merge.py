import json


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


with open("base.json", encoding="utf-8") as base, open("override.json", encoding="utf-8") as override:
    merged = merge(json.load(base), json.load(override))
with open("merged.json", "w", encoding="utf-8") as target:
    json.dump(merged, target, indent=2, sort_keys=True)
    target.write("\n")
