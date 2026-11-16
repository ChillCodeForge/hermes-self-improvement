set -eu
for path in inbox/IMG_* inbox/VID_*; do
  [ -f "$path" ] || continue
  name=$(basename "$path")
  stamp=$(printf '%s' "$name" | cut -d_ -f2)
  dir="photos/$(printf '%s' "$stamp" | cut -c1-4)/$(printf '%s' "$stamp" | cut -c5-6)/$(printf '%s' "$stamp" | cut -c7-8)"
  mkdir -p "$dir"
  mv "$path" "$dir/$name"
done
