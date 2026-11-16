from pathlib import Path

POLICY = {"api": ("250m", "256Mi", "500m", "512Mi"), "log-shipper": ("50m", "64Mi", "100m", "128Mi")}
output, current = [], None
for line in Path("deployment.yaml").read_text().splitlines():
    output.append(line)
    stripped = line.strip()
    if stripped.startswith("- name:"):
        current = stripped.split(":", 1)[1].strip()
    elif stripped.startswith("image:") and current in POLICY:
        indent = " " * (len(line) - len(line.lstrip()))
        rc, rm, lc, lm = POLICY[current]
        output += [
            f"{indent}resources:", f"{indent}  requests:", f"{indent}    cpu: {rc}",
            f"{indent}    memory: {rm}", f"{indent}  limits:", f"{indent}    cpu: {lc}",
            f"{indent}    memory: {lm}",
        ]
Path("deployment.yaml").write_text("\n".join(output) + "\n")
