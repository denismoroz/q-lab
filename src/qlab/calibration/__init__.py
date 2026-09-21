"""Ruleset calibration against noise (docs/TASKS.md, T16; docs/PLAN.md, M3).

Answers one question: fed strategies known to have no edge, how often does
`qlab evaluate`'s ruleset say yes, and what is the best `sharpe_net` noise
reaches? Until that number exists, any admitted strategy's numbers are
unreadable on their own -- see `qlab.calibration.noise` for how the noise
books are built (structurally matched to a real strategy, never free-riding
on zero turnover), `qlab.calibration.run` for how they are pushed through
the real `evaluate_spec`, and `qlab.calibration.report` for how the result
is written to `docs/CALIBRATION_<rules-version>.md`.
"""

from __future__ import annotations
