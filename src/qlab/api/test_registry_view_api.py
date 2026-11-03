"""The registry page's extras (`qlab.api.registry_view`, owner 2026-11-03): the three bench labels,
the strategy family, the plain title and the 24-hour mark on `GET /api/ideas`, and the bench split
on the funnel and the idea page. Reuses the fixture registry of `test_api`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from qlab import test_bench_label as tb
from qlab.api import test_api
from qlab.api.registry_view import RECENT, recent_changes
from qlab.registry.models import (
    Idea,
    IdeaStatus,
    StageTransition,
    TrialRoute,
)
from qlab.registry.strategies import FAMILY_ORDER, family_of_root

client, engine = test_api.client, test_api.engine
populated, session_factory = test_api.populated, test_api.session_factory

R = TrialRoute


def _ideas(client) -> dict[str, dict]:
    return {i["id"]: i for i in client.get("/api/ideas").json()["items"]}


def _transition(session, idea_id, old, new, at) -> None:
    session.add(StageTransition(idea_id=idea_id, from_status=old, to_status=new, at=at,
                                reason="test"))
    session.flush()


# --------------------------------------------------------------------------
# Family


@pytest.mark.parametrize(("root", "family"), [
    ("strategy-b-v2", "bv2"), ("bv2-fast-release-by-own-trend", "bv2"), ("bv3-income", "bv3"),
    ("trend-tsmom-crypto", "trend"), ("frab", "frab"), ("frab-lot-leftover", "frab"),
    ("frab-binance-spot", "frab-spot"), ("xsmom", "xsmom"), ("xsmom-overlays", "xsmom"),
    ("gen-carry-koijen-funding", "generator"), ("fx-carry", "other"), ("alpha", "other"),
])
def test_a_root_belongs_to_one_family(root, family) -> None:
    assert family_of_root(root) == family
    assert family in FAMILY_ORDER


def test_every_idea_carries_its_root_and_family_and_a_variant_follows_its_root(
        client, session_factory) -> None:
    s = session_factory()
    tb._idea(s, "strategy-b-v2", IdeaStatus.PAPER)
    tb._idea(s, "bv2-variant", IdeaStatus.BENCH, parent="strategy-b-v2")
    tb._idea(s, "bv2-variant-twin", IdeaStatus.BENCH, parent="bv2-variant")
    tb._idea(s, "gen-thing", IdeaStatus.CANDIDATE)
    s.commit()
    s.close()

    rows = _ideas(client)

    assert rows["strategy-b-v2"]["family"] == "bv2" and rows["strategy-b-v2"]["root_id"] == (
        "strategy-b-v2")
    # a variant of a variant is the same strategy: the family is the root's
    assert rows["bv2-variant-twin"]["root_id"] == "strategy-b-v2"
    assert rows["bv2-variant-twin"]["family"] == "bv2"
    assert rows["gen-thing"]["family"] == "generator"
    assert rows["alpha"]["family"] == "other" and rows["omega"]["family"] == "other"


def test_the_list_counts_ideas_per_family_in_the_fixed_order_without_noise(
        client, session_factory) -> None:
    s = session_factory()
    tb._idea(s, "trend-tsmom-crypto", IdeaStatus.PAPER)
    tb._idea(s, "trend-x", IdeaStatus.BENCH, parent="trend-tsmom-crypto")
    tb._idea(s, "xsmom", IdeaStatus.BENCH)
    tb._idea(s, "noise-1", IdeaStatus.CANDIDATE)
    s.commit()
    s.close()

    body = client.get("/api/ideas").json()

    counts = {f["family"]: f["count"] for f in body["family_counts"]}
    assert counts == {"trend": 2, "xsmom": 1, "other": 2}  # `other`: alpha and omega
    keys = [f["family"] for f in body["family_counts"]]
    assert keys == [k for k in FAMILY_ORDER if k in counts]
    real = [i for i in body["items"] if not i["id"].startswith("noise-")]
    assert sum(counts.values()) == len(real)


# --------------------------------------------------------------------------
# Plain title


def test_a_row_carries_the_plain_title_the_idea_page_opens_with(client) -> None:
    row = _ideas(client)["alpha"]
    page = client.get("/api/ideas/alpha").json()

    assert row["plain_title"] and row["plain_title"] == page["plain"]["title"]


# --------------------------------------------------------------------------
# Bench labels


def _bench_registry(session_factory) -> None:
    s = session_factory()
    tb._pair(s, "early", R.NEEDS_FORWARD, R.NEEDS_FORWARD)
    tb._pair(s, "executionless", R.NEEDS_INFRASTRUCTURE, R.NEEDS_FORWARD)
    tb._pair(s, "waits", R.REJECT, R.NEEDS_FORWARD)
    tb._pair(s, "crashed", R.ERROR, None)
    s.commit()
    s.close()


def test_a_bench_row_carries_its_label_and_the_filter_key(client, session_factory) -> None:
    _bench_registry(session_factory)

    rows = _ideas(client)

    for idea_id, reason, words in (
            ("early", "too-early", "рано судить"),
            ("executionless", "passed-no-execution", "прошла, нет исполнения"),
            ("waits", "awaits-second-venue", "ждёт вторую площадку"),
            ("crashed", "undetermined", "скамейка — причина не определена")):
        row = rows[idea_id]
        assert row["status"] == "bench"  # the stored status is untouched
        assert row["bench"]["reason"] == reason and row["bench"]["label_ru"] == words
        assert row["status_key"] == f"bench-{reason}"
    # off the bench there is no label and the key is the status itself
    assert rows["alpha"]["bench"] is None and rows["alpha"]["status_key"] == "validated"


def test_the_labels_follow_a_new_run(client, session_factory) -> None:
    _bench_registry(session_factory)
    assert _ideas(client)["early"]["bench"]["reason"] == "too-early"

    s = session_factory()
    tb._run(s, "early", R.REJECT, "binance")  # Binance now says no; Hyperliquid still waits
    s.commit()
    s.close()

    assert _ideas(client)["early"]["bench"]["reason"] == "awaits-second-venue"


def test_the_funnel_splits_the_bench_in_four_counts(client, session_factory) -> None:
    _bench_registry(session_factory)

    body = client.get("/api/funnel").json()

    got = {b["reason"]: b["count"] for b in body["bench_by_label"]}
    assert [b["reason"] for b in body["bench_by_label"]] == [
        "too-early", "passed-no-execution", "awaits-second-venue", "undetermined"]
    assert sum(got.values()) == body["ideas_by_status"]["bench"]
    # the idea and its second stage are two ideas, each read on its own ladder: both show the
    # Binance pass next to Hyperliquid's waiting
    assert got["passed-no-execution"] == 2 and got["undetermined"] == 1


def test_the_idea_page_and_its_ladder_carry_the_label(client, session_factory) -> None:
    _bench_registry(session_factory)

    page = client.get("/api/ideas/executionless").json()

    assert page["bench"]["label_ru"] == "прошла, нет исполнения"
    assert page["ladder"]["bench"] == page["bench"]
    assert page["family"] == "other"
    assert client.get("/api/ideas/alpha").json()["bench"] is None


# --------------------------------------------------------------------------
# The 24-hour mark

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


def test_a_status_changed_in_the_last_day_is_marked_old_to_new(session_factory) -> None:
    s = session_factory()
    tb._idea(s, "fresh")
    tb._idea(s, "old")
    tb._idea(s, "chain")
    tb._idea(s, "back-and-forth")
    tb._idea(s, "brand-new")
    tb._idea(s, "edge")
    B, V, C, J = IdeaStatus.BENCH, IdeaStatus.VALIDATED, IdeaStatus.CANDIDATE, IdeaStatus.REJECTED
    _transition(s, "fresh", V, B, NOW - timedelta(hours=2))
    _transition(s, "old", V, B, NOW - timedelta(hours=30))
    # one night, two steps: the owner sees one move, candidate -> validated
    _transition(s, "chain", C, B, NOW - timedelta(hours=5))
    _transition(s, "chain", B, V, NOW - timedelta(hours=4))
    # ended where it began: nothing changed for the owner
    _transition(s, "back-and-forth", B, J, NOW - timedelta(hours=6))
    _transition(s, "back-and-forth", J, B, NOW - timedelta(hours=3))
    _transition(s, "brand-new", None, C, NOW - timedelta(hours=1))
    _transition(s, "edge", B, V, NOW - RECENT + timedelta(minutes=1))
    s.commit()

    got = recent_changes(s, now=NOW)
    s.close()

    assert set(got) == {"fresh", "chain", "brand-new", "edge"}
    assert (got["fresh"]["from"], got["fresh"]["to"]) == ("validated", "bench")
    assert (got["chain"]["from"], got["chain"]["to"], got["chain"]["count"]) == (
        "candidate", "validated", 2)
    assert got["brand-new"]["from"] is None and got["brand-new"]["to"] == "candidate"


def test_a_change_just_past_the_day_is_not_marked(session_factory) -> None:
    s = session_factory()
    tb._idea(s, "late")
    _transition(s, "late", IdeaStatus.VALIDATED, IdeaStatus.BENCH,
                NOW - RECENT - timedelta(minutes=1))
    s.commit()

    assert recent_changes(s, now=NOW) == {}
    s.close()


def test_the_list_marks_the_ideas_that_changed_in_the_last_day(client, session_factory) -> None:
    s = session_factory()
    now = datetime.now(UTC)
    tb._idea(s, "moved")
    tb._idea(s, "still")
    _transition(s, "moved", IdeaStatus.VALIDATED, IdeaStatus.BENCH, now - timedelta(hours=1))
    _transition(s, "still", IdeaStatus.VALIDATED, IdeaStatus.BENCH, now - timedelta(days=3))
    s.commit()
    s.close()

    rows = _ideas(client)

    assert rows["moved"]["changed_24h"]["from"] == "validated"
    assert rows["moved"]["changed_24h"]["to"] == "bench"
    assert rows["still"]["changed_24h"] is None and rows["alpha"]["changed_24h"] is None


def test_the_idea_table_is_not_changed_by_reading_it(client, session_factory) -> None:
    _bench_registry(session_factory)
    client.get("/api/ideas")
    client.get("/api/funnel")

    s = session_factory()
    assert s.query(StageTransition).count() == 0
    assert s.get(Idea, "early").status == IdeaStatus.BENCH
    s.close()
