set -eu
grep ' ERROR ' app.log | sed -n 's/.* code=\([A-Z0-9]*\).*/\1/p' | sort | uniq -c \
  | awk '{print $1, $2}' | sort -k1,1nr -k2,2 > error-summary.txt
