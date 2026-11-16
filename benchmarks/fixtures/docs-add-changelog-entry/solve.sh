set -eu
awk '
  { print }
  /^- Export reports as CSV\.$/ && !done {
    print ""
    print "### Fixed"
    print ""
    print "- Retry uploads when the storage API returns HTTP 503."
    done = 1
  }
' CHANGELOG.md > CHANGELOG.md.new
mv CHANGELOG.md.new CHANGELOG.md
