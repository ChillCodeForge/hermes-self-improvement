#!/bin/sh
# Usage: backup.sh SOURCE_DIR DEST_DIR
# Copies every regular file in SOURCE_DIR to DEST_DIR/<name>.bak
src=$1
dest=$2
mkdir -p $dest
for file in $(ls $src); do
  cp $src/$file $dest/$file.bak
done
