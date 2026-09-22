"""Tests for `qlab.strategies.live._loader`."""

from __future__ import annotations

import pytest

from qlab.strategies.live._loader import ALLOWED_MODULES, FrabImportError, import_frab


def test_allowed_modules_import_cleanly() -> None:
    """Every module on the allowlist actually imports, with no new
    dependency beyond frab's own params/constants and the stdlib -- this is
    the automated re-check of the manual verification T29's scope already
    did (see module docstring)."""
    for module_name in sorted(ALLOWED_MODULES):
        import_frab(module_name)  # must not raise


def test_disallowed_module_is_refused() -> None:
    """A module not on the allowlist -- even a real, importable frab module
    -- must be refused, not silently imported. This is the mechanism, not
    the exchange layer itself; `test_exchange_layer_is_unusable` below is
    the belt-and-suspenders structural guard."""
    with pytest.raises(FrabImportError, match="not on q-lab's frab import allowlist"):
        import_frab("frab.strategy.trend.engine")


def test_exchange_layer_is_unusable() -> None:
    """q-lab never places orders (CLAUDE.md: "Ордеров не ставит никогда").

    This is the structural form of that rule, not a policy check: the
    Hyperliquid SDK and `eth_account` (the signing library any order
    submission needs) are simply not installed in q-lab's venv. Even a bug
    that widened `ALLOWED_MODULES` to frab's exchange/signing layer could
    not make q-lab place an order, because the dependency required to do so
    does not exist in this environment. If this test ever fails, it means
    someone added `hyperliquid`/`eth_account` to q-lab's dependencies --
    which would silently remove the safety property this whole package
    depends on -- not that the loader's allowlist has a bug.
    """
    with pytest.raises(ImportError):
        import hyperliquid  # noqa: F401

    with pytest.raises(ImportError):
        import eth_account  # noqa: F401
