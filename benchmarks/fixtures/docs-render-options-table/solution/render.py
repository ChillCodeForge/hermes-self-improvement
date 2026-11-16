import json

with open("cli-options.json", encoding="utf-8") as source:
    options = json.load(source)
lines = ["| Option | Default | Description |", "|---|---|---|"]
for option in options:
    default = f"`{option['default']}`" if option["default"] is not None else "-"
    lines.append(f"| `{option['name']}` | {default} | {option['description']} |")
with open("README.md", encoding="utf-8") as source:
    text = source.read()
start, end = "<!-- options:start -->\n", "<!-- options:end -->"
head, rest = text.split(start, 1)
_old, tail = rest.split(end, 1)
with open("README.md", "w", encoding="utf-8") as target:
    target.write(head + start + "\n".join(lines) + "\n" + end + tail)
