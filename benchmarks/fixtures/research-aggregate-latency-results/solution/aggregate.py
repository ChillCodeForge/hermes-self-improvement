import json
from collections import defaultdict
from pathlib import Path

latencies, totals = defaultdict(list), defaultdict(int)
for path in sorted(Path("results").glob("*.json")):
    for item in json.loads(path.read_text())["measurements"]:
        totals[item["model"]] += 1
        if item["success"]:
            latencies[item["model"]].append(item["latency_ms"])
models = {
    model: {
        "mean_latency_ms": round(sum(latencies[model]) / len(latencies[model]), 1),
        "success_rate": round(len(latencies[model]) / totals[model], 3),
    }
    for model in sorted(totals)
}
eligible = [model for model, row in models.items() if row["success_rate"] >= 0.6]
fastest = min(eligible, key=lambda model: models[model]["mean_latency_ms"])
Path("report.json").write_text(json.dumps({"models": models, "fastest_model": fastest}, indent=2) + "\n")
