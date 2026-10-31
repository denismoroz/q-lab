"""Starting and guarding a run: the lock, the preview, the detached start (docs/PIPELINE.md).

**One run at a time.** The runner (`qlab night run`) holds an advisory lock on `data/night/run.lock`
for as long as it lives, whoever started it: launchd at 03:20, the console's button, or a hand. The
lock is the operating system's (`flock`): it is released when the process ends however it ends, so a
crashed night cannot leave it stuck, and the API only has to TRY the lock to know whether a night
is running. The holder writes who it is into the file (pid, trigger, run token) for the refusal's
message.

**A run on its own process.** The console's button starts `ops/night.sh` in a session of its own
(`start_new_session`), so the run lives when the API restarts or the console is closed, and the
script wraps it in `caffeinate` and appends to `data/night/logs/<date>.log` like the 03:20 night.
This is the console's one way to change anything, and it is local: the API listens on 127.0.0.1
only and refuses a request that does not come from the loopback with the console's own header.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
import uuid
from collections.abc import Collection
from datetime import UTC, date, datetime
from pathlib import Path

from qlab.nightflow.journal import pid_alive
from qlab.nightflow.plan import make_plan
from qlab.nightflow.table import Table

ROOT = Path(__file__).resolve().parents[3]
LOCK_FILE = ROOT / "data" / "night" / "run.lock"
STARTING_FILE = ROOT / "data" / "night" / "run.starting"
NIGHT_SCRIPT = ROOT / "ops" / "night.sh"
LOCK_ENV = "QLAB_NIGHT_LOCK"
EXIT_BUSY = 75


def lock_path() -> Path:
    return Path(os.environ[LOCK_ENV]) if os.environ.get(LOCK_ENV) else LOCK_FILE


class RunLock:
    """The runner's hold on the night. `acquire` retries for a moment: a probe of the API holds the
    lock for microseconds and must not make a real start fail."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or lock_path()
        self._fd: int | None = None

    def acquire(self, info: dict, *, wait: float = 1.5) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(0.1)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({**info, "pid": os.getpid(),
                                 "since": datetime.now(UTC).isoformat()}).encode())
        os.fsync(fd)
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> RunLock:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def lock_holder(path: Path | None = None) -> dict | None:
    """Who holds the night's lock: its info (empty for an instant), or None when it is free."""
    target = path or lock_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raw = os.pread(fd, 4096, 0)
            try:
                return json.loads(raw) if raw else {}
            except ValueError:
                return {}
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def refusal() -> str | None:
    """Why a run cannot start now, in plain words; None when it can."""
    holder = lock_holder()
    if holder is not None:
        who = {"schedule": "ночь по расписанию", "button": "запуск с кнопки",
               "hand": "запуск вручную"}.get(holder.get("trigger"), "запуск")
        since = holder.get("since")
        when = ""
        if since:
            when = f" с {datetime.fromisoformat(since).astimezone():%H:%M}"
        return f"Уже идёт запуск ({who}{when}, процесс {holder.get('pid', '?')}). Дождитесь конца."
    try:
        starting = json.loads(STARTING_FILE.read_text())
    except (OSError, ValueError):
        return None
    age = time.time() - float(starting.get("at", 0))
    if pid_alive(starting.get("pid")) and age < 3600:
        return ("Запуск принят и ждёт своей очереди (скрипт ночи ждёт, пока допишет суточный "
                "рекордер). Второй запуск не нужен.")
    return None


def preview(table: Table, *, done_today: Collection[str], off: Collection[str] = (),
            restart: str | None = None, skips: dict[str, str] | None = None) -> dict:
    """What a run would do, for the console's confirmation. `restart` runs the step, every step
    that depends on it by `data`, and the steps that only read."""
    only = table.restart_set(restart) if restart else None
    plan = make_plan(table, off=off, only=only, done_today=done_today)
    skips = skips or {}
    rows = []
    for item in plan:
        step = table.by_id[item.step]
        row = item.to_json()
        row.update(title=step.title, kind=step.kind, tokens=step.tokens,
                   done_today=step.id in done_today)
        if item.action == "run" and step.id in skips:
            row["note"] = skips[step.id]
        rows.append(row)
    running = [r for r in rows if r["action"] == "run"]
    spends = [r for r in running if r["tokens"] and r["step"] not in skips]
    soft = []
    if restart:
        for child in table.descendants(restart, ("fresh", "exclusive", "budget")):
            if child not in (only or []):
                soft.append(child)
    return {
        "steps": rows,
        "run": [r["step"] for r in running],
        "token_steps": [r["step"] for r in spends],
        "off": [r["step"] for r in rows if r["action"] == "off"],
        "blocked": [{"step": r["step"], "blocked_by": r["blocked_by"], "reason": r["reason"]}
                    for r in rows if r["action"] == "blocked"],
        "outside": [r["step"] for r in rows if r["action"] == "outside"],
        "reads_after": soft,
        "restart": restart,
    }


def start_detached(args: list[str], *, trigger: str = "button", wait: float = 25.0) -> dict:
    """Start `ops/night.sh` on a process of its own and wait until it holds the lock (the run is
    then real) or ends. Returns {started, pid, token} or {started: False, reason}."""
    why = refusal()
    if why:
        return {"started": False, "reason": why}
    token = uuid.uuid4().hex
    env = {**os.environ, "QLAB_TRIGGER": trigger, "QLAB_RUN_TOKEN": token}
    proc = subprocess.Popen(
        [str(NIGHT_SCRIPT), *args], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    STARTING_FILE.parent.mkdir(parents=True, exist_ok=True)
    STARTING_FILE.write_text(json.dumps({"pid": proc.pid, "at": time.time(), "token": token}))
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        holder = lock_holder()
        if holder and holder.get("token") == token:
            return {"started": True, "pid": holder.get("pid"), "token": token}
        if proc.poll() is not None:
            if proc.returncode == EXIT_BUSY:
                return {"started": False, "reason": refusal() or "Уже идёт другой запуск."}
            return {"started": False, "reason": (
                f"Запуск завершился сразу (код {proc.returncode}); смотрите data/night/logs.")}
        time.sleep(0.25)
    return {"started": True, "pid": proc.pid, "token": token, "pending": True,
            "note": "Запуск принят, но ещё не начался (ждёт запись суточных данных)."}


def run_arguments(*, off: Collection[str] = (), restart: str | None = None,
                  only: Collection[str] | None = None) -> list[str]:
    """The command line `qlab night run` gets from the button (through `ops/night.sh`)."""
    args: list[str] = []
    for step in sorted(off):
        args += ["--off", step]
    if restart:
        args += ["--restart", restart]
    for step in sorted(only or ()):
        args += ["--only", step]
    return args


def today() -> date:
    return datetime.now(UTC).date()
