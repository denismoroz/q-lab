"""`GET /api/ideas/{id}` carries the idea's anomaly block: the detector's flags with their
sentences, and the investigator's dossier if there is one (docs/ANOMALIES.md). Reuses the fixture
registry of `test_api`."""

from __future__ import annotations

from qlab.anomalies import flags as fl
from qlab.anomalies import store
from qlab.anomalies.scan import ScanResult
from qlab.api import test_api
from qlab.registry.models import Spec, Trial

# The fixture registry and the client of `test_api`, as fixtures of this module.
client, engine = test_api.client, test_api.engine
populated, session_factory = test_api.populated, test_api.session_factory


def _scan(session, flags: list[fl.FlagResult]) -> int:
    trial = session.query(Trial).join(Spec, Trial.spec_id == Spec.id).filter(
        Spec.idea_id == "alpha").order_by(Trial.id.desc()).first()
    store.save_scan(session, ScanResult(trial.id, "alpha", "paper", flags), "test.1")
    return trial.id


FIRED = fl.FlagResult(
    "implausible_day", fl.FIRES, 1.8, 1.0,
    detail={"days_over": 14, "days": 2262, "top": [{
        "day": "2025-11-02", "return": 0.194, "day_line": 0.107, "positions": 20, "gross": 1.0,
        "net": 0.0, "times_line": 1.8}]})
QUIET = fl.FlagResult("concentrated_instruments", fl.QUIET, 0.16, 0.25,
                      detail={"top1_share": 0.16, "top3_share": 0.34, "instruments": 548,
                              "top": [{"instrument": "LAB", "net": 0.4, "share": 0.16}]})
UNDEFINED = fl.FlagResult("stand_vs_paper", fl.UNDEFINED, reason="у идеи нет бумажной книги")


def test_an_idea_that_was_never_scanned_has_no_block(client):
    assert client.get("/api/ideas/alpha").json()["anomalies"] is None


def test_the_block_holds_every_flag_in_plain_words(client, session_factory):
    session = session_factory()
    trial_id = _scan(session, [FIRED, QUIET, UNDEFINED])
    session.commit()
    session.close()

    block = client.get("/api/ideas/alpha").json()["anomalies"]

    scan = block["scan"]
    assert (scan["trial_id"], scan["route"], scan["config_version"]) == (
        trial_id, "paper", "test.1")
    assert scan["fired"] == ["implausible_day"]
    first, second, third = scan["flags"]
    assert first["state"] == "fires" and first["state_ru"] == "сработал"
    assert "Самый большой день 2025-11-02: +19.4% при 20 позициях" in first["sentence"]
    assert second["state_ru"] == "молчит" and "LAB" in second["sentence"]
    assert third["state"] == "not defined" and "нет бумажной книги" in third["sentence"]
    assert block["investigation"] is None


def test_the_dossier_is_shown_with_its_cause_and_its_check(client, session_factory):
    session = session_factory()
    trial_id = _scan(session, [FIRED])
    store.save_investigation(session, store.NewInvestigation(
        idea_id="alpha", trial_id=trial_id, model="opus",
        flags_fingerprint=store.fingerprint(["implausible_day"], "test.1"),
        cause="unrealistic-tradability", summary="Прибыль взята на монетах, которых нет.",
        dossier="# Разбор аномалии: alpha", evidence=[{"text": "t", "call": 1, "quote": "q"}],
        rejected=[], proposals=[{"kind": "new_check", "text": "p", "code_request": None}],
        caught_by=[], new_check="Сверять списки монет.", programmer_requests=["2026-10-28-1"],
        tokens=61_000, calls=3))
    session.commit()
    session.close()

    investigation = client.get("/api/ideas/alpha").json()["anomalies"]["investigation"]

    assert investigation["cause"] == "unrealistic-tradability"
    assert investigation["cause_ru"] == store.CAUSES_RU["unrealistic-tradability"]
    assert investigation["new_check"] == "Сверять списки монет."
    assert investigation["new_check_status"] == "proposed"
    assert investigation["programmer_requests"] == ["2026-10-28-1"]
    assert investigation["dossier"].startswith("# Разбор аномалии")
    assert investigation["model"] == "opus" and investigation["tokens"] == 61_000


def test_the_api_stays_read_only_with_the_new_block(client):
    # Every route is a GET; the block reads two tables and writes none.
    assert client.post("/api/ideas/alpha").status_code == 405
    assert client.get("/api/ideas/alpha").status_code == 200
