# Hindsight leak-type knowledge

One file per leak type in `docs/taxonomy.md`, read at runtime by
`hindsight_core.skills` as plain text. These are **not** Claude Code plugin
skills and nothing in the runtime depends on Claude Code being installed.

Each file: what the leak looks like, the code pattern that signals it, the
mechanical fix, and the near-miss that must not be flagged.
