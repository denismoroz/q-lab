"""Strategies that CALL funding-rate-arbitrage's own live code, not a transcription.

See `qlab.strategies.live._loader` for how the source is located and
restricted, and `docs/TASKS.md` T29 ("Дополнение 2026-09-22") for why this
package exists: a hand-written port of a live strategy is an interpretation
of it, not a measurement of it. Modules here import the live signal code
through `_loader.import_frab` and drive it bar by bar; they never import
`hyperliquid`/`eth_account` (the exchange/signing layer) directly or
transitively — see `_loader`'s allowlist and
`qlab.strategies.live.test_loader` for the structural proof that this
package cannot place an order.
"""

from __future__ import annotations
