# Tools

`TOOL_CREATE` (new directory) and `TOOL_PATCH` (existing directory) change
files under `brain/tools/<name>/` (`.py` and text files). Tool code is the
highest risk class and always needs a signed operator approval by default.
Candidate code is syntax-checked when the candidate is built; it is never
executed by the candidate builder or the promoter.
