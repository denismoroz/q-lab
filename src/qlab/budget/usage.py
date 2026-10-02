"""What a Claude Code call reports about itself (docs/BUDGET.md).

`claude -p --output-format stream-json --verbose` emits, among its events:

- `rate_limit_event` with `rate_limit_info.unifiedWindows`: the
  subscription's five-hour and seven-day windows, each with `utilization`
  (a fraction, two decimals) and `resetsAt` (Unix seconds), plus a `status`
  ("allowed" while calls are served). This is the SHARED quota -- it includes
  the owner's own interactive work, which q-lab's ledger never sees.
- `result` with `usage` (input, output, cache-write and cache-read tokens),
  `total_cost_usd` and `modelUsage`.

Observed on 2026-10-02 with a one-word haiku call: utilization 0.42 / 0.17,
and 39,857 cache-write tokens for the CLI's own system prompt -- even an empty
call costs ~40k tokens.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class Window:
    utilization: float
    resets_at: datetime


@dataclass(frozen=True)
class UsageReading:
    status: str
    five_hour: Window | None
    seven_day: Window | None


@dataclass(frozen=True)
class CallUsage:
    tokens_in: int
    tokens_out: int
    cache_write: int
    cache_read: int
    cost_usd: float
    model: str | None

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out + self.cache_write + self.cache_read


@dataclass(frozen=True)
class CallReport:
    usage: CallUsage | None
    reading: UsageReading | None
    text: str | None
    is_error: bool


def _window(raw: dict | None) -> Window | None:
    if not raw or "utilization" not in raw or "resetsAt" not in raw:
        return None
    return Window(float(raw["utilization"]), datetime.fromtimestamp(int(raw["resetsAt"]), UTC))


def parse_stream(lines: Iterable[str]) -> CallReport:
    """The usage and the LAST window reading of one call's stream-json output.
    Lines that are not JSON are ignored; missing parts are None, never
    guessed."""
    usage = reading = text = None
    is_error = False
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if event.get("type") == "rate_limit_event":
            info = event.get("rate_limit_info") or {}
            windows = info.get("unifiedWindows") or {}
            reading = UsageReading(
                status=str(info.get("status", "unknown")),
                five_hour=_window(windows.get("five_hour")),
                seven_day=_window(windows.get("seven_day")),
            )
        elif event.get("type") == "result":
            u = event.get("usage") or {}
            models = list((event.get("modelUsage") or {}).keys())
            usage = CallUsage(
                tokens_in=int(u.get("input_tokens", 0)),
                tokens_out=int(u.get("output_tokens", 0)),
                cache_write=int(u.get("cache_creation_input_tokens", 0)),
                cache_read=int(u.get("cache_read_input_tokens", 0)),
                cost_usd=float(event.get("total_cost_usd", 0.0)),
                model=models[0] if len(models) == 1 else (",".join(models) or None),
            )
            text = event.get("result")
            is_error = bool(event.get("is_error", False))
    return CallReport(usage=usage, reading=reading, text=text, is_error=is_error)


__all__ = ["CallReport", "CallUsage", "UsageReading", "Window", "parse_stream"]
