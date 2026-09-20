"""Free, no-API-key market data source fetchers (qlab.data.sources.*).

Each source module exposes a ``fetch_universe(instruments, start, end,
interval) -> dict[str, InstrumentHistory]`` function with the same shape, so
``qlab.data.snapshot`` can assemble a ``MarketPanel`` without caring which
venue the data came from.
"""

from __future__ import annotations
