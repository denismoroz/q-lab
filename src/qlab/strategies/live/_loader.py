"""Import a fixed allowlist of pure modules from the `funding-rate-arbitrage`
("frab") checkout, and nothing else.

Why this exists (`docs/TASKS.md` T29, "Дополнение 2026-09-22"): the owner
decided the bench must CALL the live engine's own code rather than
transcribe it, because a transcription is an interpretation and q-lab had
been measuring its interpretation, not the strategy that actually runs.
Calling the live code means q-lab's Python process imports modules that
live in a DIFFERENT repository (`funding-rate-arbitrage`, which places real
orders on Hyperliquid) — that is a new coupling this project did not have
before, and it needs its own guard rail: q-lab may read frab's pure
signal/parameter code, and NOTHING ELSE. In particular it must never be
able to reach frab's exchange/signing layer, because "q-lab ордеров не
ставит никогда" (`CLAUDE.md`) has to remain true structurally, not just by
convention, once q-lab can import code from the repo that DOES place
orders.

Two independent guards enforce this:

  1. `import_frab` only imports modules on `ALLOWED_MODULES` below — a
     fixed allowlist of the seven pure modules verified (by hand, importing
     each one in q-lab's own venv) to depend on nothing beyond stdlib and
     frab's own params/constants. Anything else raises `FrabImportError`
     with a message explaining why, rather than importing it.
  2. Structurally, `hyperliquid` and `eth_account` (the SDK and signing
     library the exchange layer needs) are simply NOT INSTALLED in q-lab's
     venv — verified when this task was scoped, and re-verified by
     `qlab.strategies.live.test_loader.test_exchange_layer_is_unusable`.
     Even a bug that widened the allowlist could not make q-lab place an
     order: the dependency required to do so does not exist here. That
     test's docstring says so explicitly — it is the structural form of
     "q-lab never places orders", not just a dependency check.

Where the frab checkout lives is NOT hardcoded to one path: it defaults to
`/Users/d/prj/funding-rate-arbitrage/src` (this machine's checkout) but is
overridable with the `QLAB_FRAB_SRC` environment variable, so a CI runner or
a different machine can point it elsewhere without editing code.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

#: Environment variable overriding the frab checkout's `src` directory.
#: Unset -> `DEFAULT_FRAB_SRC`.
FRAB_SRC_ENV_VAR = "QLAB_FRAB_SRC"

#: Default frab checkout location on this machine.
DEFAULT_FRAB_SRC = Path("/Users/d/prj/funding-rate-arbitrage/src")

#: The ONLY modules q-lab is allowed to import from frab. Each one was
#: verified importable in q-lab's own venv (with `frab_src_path()` on
#: `sys.path`) with no new dependency beyond frab's own params/constants
#: modules and the stdlib -- see `docs/TASKS.md` T29's "Дополнение
#: 2026-09-22" table. Adding a module here is a decision that needs the
#: same verification, not a convenience edit.
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "frab.strategy.trend.signals",
        "frab.strategy.trend.params",
        "frab.strategy.b2.book",
        "frab.strategy.b2.params",
        "frab.strategy.xsmom.evaluators.signal",
        "frab.strategy.xsmom.params",
        "frab.constants",
    }
)

_sys_path_lock = threading.Lock()


class FrabImportError(ImportError):
    """Raised instead of importing anything not on `ALLOWED_MODULES`, or
    when the frab checkout cannot be found at all.

    This is a safety boundary, not a convenience check: q-lab must never
    import frab's exchange/signing layer (`CLAUDE.md`: "Ордеров не ставит
    никогда"), and the allowlist is how that stays true even as frab grows
    modules q-lab has never heard of.
    """


def frab_src_path() -> Path:
    """Resolve the frab checkout's `src` directory.

    `QLAB_FRAB_SRC`, if set, overrides `DEFAULT_FRAB_SRC` — see module
    docstring.

    Raises:
        FrabImportError: the resolved path does not exist / is not a
            directory. This is reported as a q-lab-side configuration
            problem, not a generic `FileNotFoundError`, because the caller
            (a strategy's `target_weights`) has no other way to explain
            what went wrong.
    """
    raw = os.environ.get(FRAB_SRC_ENV_VAR)
    path = Path(raw).expanduser() if raw else DEFAULT_FRAB_SRC
    if not path.is_dir():
        raise FrabImportError(
            f"frab source directory not found at {path} -- set {FRAB_SRC_ENV_VAR} "
            "to the `src` directory of a funding-rate-arbitrage checkout"
        )
    return path


def frab_code_sha() -> str:
    """Best-effort git commit of the frab checkout `import_frab` reads from.

    Mirrors `qlab.pipeline.evaluate._code_sha`'s own contract exactly:
    falls back to `"unknown"` rather than raising, because a trial's
    reproducibility record should not be the reason a live-code evaluation
    fails to complete. A trial's `code_sha` column already exists
    (`docs/TASKS.md` T29: "в `trial` уже пишется `code_sha`"); this is what
    a live-strategy caller should pass there so a trial can say which
    version of the LIVE code it ran, distinct from q-lab's own
    `code_sha` for the same trial's harness/pipeline code.
    """
    try:
        src = frab_src_path()
    except FrabImportError:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(src.parent),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def import_frab(module_name: str) -> ModuleType:
    """Import `module_name` from the frab checkout, if and only if it is on
    `ALLOWED_MODULES`.

    Inserts `frab_src_path()` onto `sys.path` (once; idempotent) so the
    import resolves, exactly like the manual verification this task's scope
    already did: "with `/Users/d/prj/funding-rate-arbitrage/src` inserted on
    `sys.path`, q-lab's own venv imports [...] with no new dependencies".

    Callers should keep the returned module object (not `from module import
    name`) and call through it (`module.some_function(...)`) — this is also
    what makes the module monkeypatchable in tests, see
    `qlab.strategies.live.test_trend`'s look-ahead guard test.

    Raises:
        FrabImportError: `module_name` is not on `ALLOWED_MODULES` (message
            explains why — see class docstring), or the frab checkout is
            not where `QLAB_FRAB_SRC`/`DEFAULT_FRAB_SRC` says it is.
        ImportError: the module IS allowed but frab's own code fails to
            import (a real break in frab, not a q-lab policy decision) --
            propagated as-is rather than wrapped, so it is not confused
            with a policy refusal.
    """
    if module_name not in ALLOWED_MODULES:
        raise FrabImportError(
            f"{module_name!r} is not on q-lab's frab import allowlist "
            f"({sorted(ALLOWED_MODULES)}). q-lab may only read frab's pure "
            "signal/parameter modules -- never its exchange/signing layer -- "
            "because q-lab never places orders (CLAUDE.md). If this module "
            "is genuinely a pure signal/params module, verify it has no new "
            "dependency beyond frab's own params/constants and the stdlib, "
            "then add it to ALLOWED_MODULES explicitly."
        )
    src = str(frab_src_path())
    with _sys_path_lock:
        if src not in sys.path:
            sys.path.insert(0, src)
        return importlib.import_module(module_name)
