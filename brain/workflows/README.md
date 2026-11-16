# Workflows

`WORKFLOW` candidates change exactly one file `brain/workflows/<name>.yaml`.
The YAML must parse (safe loader, no duplicate keys). Manual gate by default;
may be unlocked to auto-promotion in the protected `config/promotion.yaml`.
