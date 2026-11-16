# Skills

One directory per skill: `brain/skills/<slug>/SKILL.md`, optionally with
supporting text files (`.md`, `.txt`, `.yaml`, `.yml`, `.json`) up to three
levels below the slug. The layout is flat: a skill directory sits directly
under `brain/skills/`. The entry file must be named exactly `SKILL.md`.

Candidate types: `SKILL_CREATE` (new slug), `SKILL_PATCH` (existing slug),
`SKILL_MERGE` (two or more slugs; at least one is removed and one survives),
`SKILL_SPLIT` (one existing slug plus at least one new slug). Skills are the
only types that may auto-promote by default.
