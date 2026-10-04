"""Tests for `qlab.venue_watch`."""

from __future__ import annotations

from qlab.venue_watch import diff_binance, diff_hyperliquid


def _e(name, lev=10, dead=False, funding="0.00001", vol="1000", oi="100"):
    return {"name": name, "maxLeverage": lev, "marginTableId": 1, "szDecimals": 2,
            "isDelisted": dead, "ctx": {"funding": funding, "dayNtlVlm": vol, "openInterest": oi}}


def test_listings_delistings_and_changes_are_reported() -> None:
    prev = {"A": _e("A"), "B": _e("B"), "C": _e("C")}
    cur = {"A": _e("A", lev=20), "B": _e("B", dead=True), "D": _e("D", vol="5000")}
    kinds = {(e.kind, e.instrument) for e in diff_hyperliquid("hyperliquid", prev, cur)}
    assert {("changed", "A"), ("delisted", "B"), ("listed", "D"), ("removed", "C")} <= kinds


def test_extremes_are_ranked_not_thresholded() -> None:
    prev = {n: _e(n) for n in "ABC"}
    cur = {"A": _e("A", funding="0.001", vol="9000"), "B": _e("B", funding="-0.0005"),
           "C": _e("C")}
    events = diff_hyperliquid("hyperliquid", prev, cur)
    top_funding = [e.instrument for e in events if e.kind == "funding-extreme"]
    assert top_funding[0] == "A"
    jumps = [e for e in events if e.kind == "dayNtlVlm-jump"]
    assert jumps[0].instrument == "A" and "x9.0" in jumps[0].detail


def test_binance_status_changes() -> None:
    prev = {"XUSDT": {"status": "TRADING"}, "YUSDT": {"status": "TRADING"}}
    cur = {"XUSDT": {"status": "SETTLING"}, "YUSDT": {"status": "TRADING"},
           "ZUSDT": {"status": "PENDING_TRADING"}}
    kinds = {(e.kind, e.instrument) for e in diff_binance(prev, cur)}
    assert kinds == {("delisted", "XUSDT"), ("listed", "ZUSDT")}


def test_an_instrument_first_seen_already_delisted_is_not_a_listing() -> None:
    from qlab.agents.scout import STRUCTURAL
    from qlab.venue_watch import diff_hyperliquid

    events = diff_hyperliquid("hyperliquid-xyz", {}, {
        "xyz:ACN": {"maxLeverage": 10, "isDelisted": True},
        "xyz:NEW": {"maxLeverage": 10}})
    kinds = {e.instrument: e.kind for e in events}
    assert kinds == {"xyz:ACN": "first-seen-delisted", "xyz:NEW": "listed"}
    assert "first-seen-delisted" not in STRUCTURAL  # does not wake the scout
