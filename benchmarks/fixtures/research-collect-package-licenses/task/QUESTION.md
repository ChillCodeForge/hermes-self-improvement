# License inventory

For every vendored package under `vendor/*/METADATA`, record its license in
`licenses.json` as `{"<Name>": "<license>"}`. Use the `License:` field; if it
is missing or `UNKNOWN`, use the last ` :: ` segment of the
`Classifier: License :: ...` line instead.
