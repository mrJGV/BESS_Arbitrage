"""The rolling-horizon loop, its accounting, and the annual-window bound.

``runner.py`` walks the delivery days, ``metrics.py`` turns a run into the
three numbers ``docs/DECISIONS.md`` §4 asks for, and ``bound.py`` runs the
one solve of §2.4 that is too large for a window.

No solver is imported here. The loop asks
:func:`bess_arb.model.get_backend` for a backend by the name in the config
and holds it as a :class:`~bess_arb.model.base.BatteryMILP`; the solver lives
inside that object, which is the whole reason the Protocol is a stateful
object rather than a function.
"""

from __future__ import annotations

__all__: list[str] = []
