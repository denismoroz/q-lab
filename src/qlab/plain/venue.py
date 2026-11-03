"""«Где проверена»: the venue and the dates of the data the idea is judged on, with the one line
that says what the two-venue split means (docs/VENUE_SPLIT.md)."""

from __future__ import annotations

from qlab.plain.specs import SpecView
from qlab.plain.words import venue_name

SPLIT_LINE = (
    "У каждой идеи два вердикта, по одному на площадку: по истории Binance (она длинная, "
    "с 2019 года) и по истории Hyperliquid (короче, с 2025 года, зато там лаборатория умеет "
    "торговать); итог — лучший из двух (docs/VENUE_SPLIT.md)."
)

_VENUE_NOTES = {
    "binance": ("это длинная история; торговать на Binance лаборатория пока не умеет: программы, "
                "которая ставила бы там заявки, у неё нет, поэтому прошедшая проверка здесь "
                "значит «нужна инфраструктура»"),
    "hyperliquid": "здесь лаборатория умеет торговать, но история короче, чем у Binance",
    "hyperliquid-xyz": "история на этом рынке короткая",
}


_HISTORY_OF = {"hyperliquid-xyz": "раздела xyz на Hyperliquid (акции, индексы, сырьё)"}
"""«на истории <этого>»: the venues whose name does not decline by itself."""


def describe(view: SpecView, *, parent_id: str | None = None, child_id: str | None = None,
             is_second_stage: bool = False) -> dict:
    """The page's «Где проверена»: `lines` (strings) and the venue's id."""
    source = view.data.get("source")
    lines = []
    start, end = view.data.get("start"), view.data.get("end")
    window = f"с {start} по {end}" if start and end else (f"с {start}" if start else "")
    note = _VENUE_NOTES.get(str(source))
    lines.append(f"Проверена на истории {_HISTORY_OF.get(str(source), venue_name(source))}"
                 + (f" {window}" if window else "") + (f": {note}." if note else "."))
    judged_from = view.extras.get("judged_from")
    if judged_from:
        lines.append(f"Оценка идёт с {judged_from}: раньше шёл разгон, на котором стратегии "
                     "копят историю для сигнала, и он в оценку не входит.")
    if is_second_stage and parent_id:
        lines.append("Это вторая проверка той же идеи: тот же механизм и те же числа (или те же "
                     "списки, из которых выбирает код), но на данных Hyperliquid; "
                     f"первая — {parent_id}.")
    elif child_id:
        lines.append(f"Вторая проверка этой идеи, на Hyperliquid, записана отдельно: {child_id}.")
    lines.append(SPLIT_LINE)
    return {"venue": source, "lines": lines}
