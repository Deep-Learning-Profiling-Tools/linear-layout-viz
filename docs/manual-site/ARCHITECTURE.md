The `docs/manual-site/` directory contains the static manual page served by project documentation.

Treat this directory as generated or curated output, not as the primary design record. The root `MANUAL.md` should explain user-facing behavior first. Regenerate or refresh this site output only after the source manual text is correct.

Do not add runtime code here. Generic viewer implementation lives in the
upstream tensor-viz packages, LL-viz runtime code lives under `src/`, and the
project-level manual exists only to document how to use the app.
