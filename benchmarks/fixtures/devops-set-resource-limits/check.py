import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

EXPECTED = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: catalog
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: api
          image: registry.example.com/catalog/api:2.1.0
          resources:
            requests:
              cpu: 250m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          ports:
            - containerPort: 8080
        - name: log-shipper
          image: registry.example.com/infra/log-shipper:0.9.3
          resources:
            requests:
              cpu: 50m
              memory: 64Mi
            limits:
              cpu: 100m
              memory: 128Mi
"""


def body(checks):
    text = read("deployment.yaml")
    checks.check("two_resource_blocks", text.count("resources:") == 2)
    checks.check("images_unchanged", "catalog/api:2.1.0" in text and "log-shipper:0.9.3" in text)
    checks.check("deployment_exact", text == EXPECTED)


run(body)
