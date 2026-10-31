"""A small bounded cache that is safe for steps running together (docs/PIPELINE.md, «Дорожки»).

The stand keeps a few results in memory so that the same panel and parameters are not computed
twice (the inner strategy of an overlay, a retuned book, the reference weights of the noise books,
a live Bv2 run). They were plain dicts that dropped their oldest item with
`pop(next(iter(cache)))`; with two steps using one, that line fails ("dictionary changed size
during iteration", a missing key) or returns a key another thread has just dropped. Here every
read and write takes one lock and nothing is computed under it: two threads that miss the same key
both compute it (the answer is the same) and the later one overwrites the earlier."""

from __future__ import annotations

import threading
from typing import Any


class BoundedCache[V]:
    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("a cache keeps at least one item")
        self.size = size
        self._items: dict[Any, V] = {}
        self._lock = threading.Lock()

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return self._items.get(key, default)

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def put(self, key: Any, value: V) -> V:
        """Keep `value` under `key`, dropping the oldest items beyond the size; returns `value`."""
        with self._lock:
            self._items.pop(key, None)
            while len(self._items) >= self.size:
                self._items.pop(next(iter(self._items)))
            self._items[key] = value
        return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
