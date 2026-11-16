set -eu
for file in docs/index.md docs/installation.md docs/guide/using-the-cli.md; do
  sed -e 's|(setup\.md)|(installation.md)|g' -e 's|(\.\./setup\.md)|(../installation.md)|g' \
      -e 's|(guide/usage\.md|(guide/using-the-cli.md|g' "$file" > "$file.new"
  mv "$file.new" "$file"
done
