"""Engine/session setup for the registry.

DB location comes from the ``QLAB_DB`` environment variable, defaulting to
``data/qlab.db`` (relative to the current working directory, per
docs/REGISTRY.md). SQLite foreign-key enforcement is off by default per
connection, so it is turned on explicitly on every new DBAPI connection.
"""

from __future__ import annotations

import os
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
    cursor.close()


def make_engine(url: str | None = None, **kwargs: object) -> Engine:
    """Create a SQLite engine with foreign-key enforcement turned on."""
    engine = create_engine(url or get_database_url(), **kwargs)
    event.listen(engine, "connect", _enable_foreign_keys)
    return engine


# Module-level default engine/session factory, built lazily from QLAB_DB so
# importing this module has no side effects on disk and honours env changes
# made before first use (e.g. in tests).
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
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
