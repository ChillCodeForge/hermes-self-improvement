#!/bin/sh
# Usage: backup.sh SOURCE_DIR DEST_DIR
# Copies every regular file in SOURCE_DIR to DEST_DIR/<name>.bak
src=$1
dest=$2
mkdir -p "$dest"
for path in "$src"/*; do
  [ -f "$path" ] || continue
  file=$(basename "$path")
  cp "$path" "$dest/$file.bak"
done
