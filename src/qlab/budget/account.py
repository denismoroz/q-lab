"""The Claude account q-lab's agents run under (night/account.yaml, docs/BUDGET.md).

Owner, 2026-10-28: «мне нужно чтобы все запускалось под moroz.denis@gmail.com». The `claude`
CLI of this machine is logged in to the owner's work account (a Team plan), and the night
spent THAT account's quota for weeks. q-lab therefore has a Claude config directory of its
own (`CLAUDE_CONFIG_DIR`), an independent login, and checks before every agent call that the
account found there is the expected one. This module is the configuration and the refusal;
the spawning and the check itself live in `qlab.budget.agent`, the only module that starts
the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from qlab.budget.guard import WrongAccount

ACCOUNT_FILE = Path(__file__).resolve().parents[3] / "night" / "account.yaml"
CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"


@dataclass(frozen=True)
class Account:
    email: str
    config_dir: Path  # expanded
    config_dir_as_written: str  # for the login command shown to the owner

    @property
    def login_command(self) -> str:
        return f"{CONFIG_DIR_ENV}={self.config_dir_as_written} claude auth login"


def load_account(path: Path | None = None) -> Account:
    data = yaml.safe_load((path or ACCOUNT_FILE).read_text(encoding="utf-8")) or {}
    email, written = str(data["email"]).strip(), str(data["config_dir"]).strip()
    return Account(email=email, config_dir=Path(written).expanduser(),
                   config_dir_as_written=written)


def same_email(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and a.strip().lower() == b.strip().lower()


def wrong_account(expected: Account, found: str | None, *, detail: str | None = None
                  ) -> WrongAccount:
    """The refusal, in plain Russian: who was found, who is expected, how to log in."""
    if detail is not None:
        who = detail
    elif found is None:
        who = "в этом каталоге настроек никто не вошёл"
    else:
        who = f"найден аккаунт {found}"
    return WrongAccount(
        f"Агенты не запущены, токены не потрачены: {who}; нужен {expected.email} "
        f"(night/account.yaml). Войти один раз: {expected.login_command}",
        found=found, expected=expected.email,
    )
