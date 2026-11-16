set -eu
sed 's|registry.example.com/shop/api:1.4.2|registry.example.com/shop/api:1.5.0|' deploy.yaml > deploy.yaml.new
mv deploy.yaml.new deploy.yaml
