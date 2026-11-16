# Latency comparison

Using every file in `results/`, write `report.json` with this shape:

    {"models": {"<model>": {"mean_latency_ms": <float>, "success_rate": <float>}},
     "fastest_model": "<model>"}

- `mean_latency_ms`: mean latency of the model's *successful* measurements,
  rounded to one decimal place.
- `success_rate`: successful / total measurements, rounded to three decimals.
- `fastest_model`: lowest `mean_latency_ms` among models whose
  `success_rate` is at least 0.6.
