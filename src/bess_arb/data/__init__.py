"""Data acquisition — run once, then frozen (CLAUDE.md invariant 6).

``esios.py`` pulls the snapshot from Red Eléctrica's API, ``omie.py`` reads
OMIE's public ``marginalpdbc`` files for the independent cross-check, and
``snapshot.py`` orchestrates the freeze and writes the manifest that makes it
evidence rather than an assertion.

After the snapshot commit this package is not modified. The project's risk
was never the MILP — a 192-period problem solves in milliseconds — it is
losing a week to file plumbing, and the way that week gets lost is by
re-pulling "just one more series".

No solver is imported here, and nothing here imports the modelling layer:
data acquisition sits below both.
"""

from __future__ import annotations

__all__: list[str] = []
