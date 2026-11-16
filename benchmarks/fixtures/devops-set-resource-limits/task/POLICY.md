# Resource policy

Every container needs a `resources` block placed directly after its `image:`
line, indented like the other container keys, in this exact shape:

    resources:
      requests:
        cpu: <request-cpu>
        memory: <request-memory>
      limits:
        cpu: <limit-cpu>
        memory: <limit-memory>

| Container   | Request CPU | Request memory | Limit CPU | Limit memory |
|-------------|-------------|----------------|-----------|--------------|
| api         | 250m        | 256Mi          | 500m      | 512Mi        |
| log-shipper | 50m         | 64Mi           | 100m      | 128Mi        |
