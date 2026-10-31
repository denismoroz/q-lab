"""Engine/session setup for the registry.

DB location comes from the ``QLAB_DB`` environment variable, defaulting to
``data/qlab.db`` (relative to the current working directory, per
docs/REGISTRY.md). SQLite foreign-key enforcement is off by default per
connection, so it is turned on explicitly on every new DBAPI connection.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def get_db_path() -> str:
    """Resolve the SQLite file path from ``QLAB_DB``, defaulting to data/qlab.db."""
    return os.environ.get("QLAB_DB", "data/qlab.db")


def get_database_url() -> str:
    """Build the sqlite+pysqlite URL used by both the app and Alembic."""
    db_path = get_db_path()
    if db_path == ":memory:":
        return "sqlite+pysqlite:///:memory:"
    return f"sqlite+pysqlite:///{db_path}"


def _enable_foreign_keys(dbapi_connection: object, _connection_record: object) -> None:
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    # The console reads the registry while a long evaluation writes it. In
    # SQLite's default rollback journal a reader blocks a writer's commit,
    # and runs failed with "database is locked" (2026-10-02, Binance runs).
    # WAL lets readers and one writer work at once (an in-memory database
    # ignores it); a busy connection waits up to a minute instead of failing
    # at once. Connection plumbing, not research parameters.
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


#: A file named here gets one line for every write transaction that held SQLite's one write lock
#: for `TRACE_MIN_SECONDS` or more (the thread, the seconds, the first statement). Off by default;
#: the proof of the lanes turns it on (docs/PIPELINE.md): a transaction that holds the lock across
#: a fetch or an agent call makes every other writer wait out its busy timeout.
TRACE_ENV = "QLAB_DB_TRACE"
TRACE_MIN_SECONDS = 1.0
_TRACE_LOCK = threading.Lock()


def _trace_write_locks(engine: Engine) -> None:
    path = os.environ.get(TRACE_ENV)
    if not path:
        return

    def before(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        if statement.lstrip()[:6].upper() in ("INSERT", "UPDATE", "DELETE", "REPLAC") \
                and "write_since" not in conn.info:
            conn.info["write_since"] = (time.monotonic(), threading.current_thread().name,
                                        " ".join(statement.split())[:90])

    def done(conn):  # noqa: ANN001
        started = conn.info.pop("write_since", None)
        if started and (held := time.monotonic() - started[0]) >= TRACE_MIN_SECONDS:
            with _TRACE_LOCK, open(path, "a", encoding="utf-8") as out:
                out.write(f"{held:.2f}\t{os.getpid()}\t{started[1]}\t{started[2]}\n")

    event.listen(engine, "before_cursor_execute", before)
    event.listen(engine, "commit", done)
    event.listen(engine, "rollback", done)


def make_engine(url: str | None = None, **kwargs: object) -> Engine:
    """Create a SQLite engine with foreign-key enforcement turned on."""
    engine = create_engine(url or get_database_url(), **kwargs)
    event.listen(engine, "connect", _enable_foreign_keys)
    _trace_write_locks(engine)
    return engine


# Module-level default engine/session factory, built lazily from QLAB_DB so
# importing this module has no side effects on disk and honours env changes
# made before first use (e.g. in tests).
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


_INIT_LOCK = threading.Lock()


def get_engine() -> Engine:
    global _engine
    with _INIT_LOCK:  # steps running together must not each make an engine on first use
        if _engine is None:
            _engine = make_engine()
        return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _SessionLocal
    engine = get_engine()
    with _INIT_LOCK:
        if _SessionLocal is None:
            _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope: commits on success, rolls back on error."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
