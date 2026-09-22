"""Request-scoped dependencies for the read-only registry API.

The session here is deliberately *not* `db.session_scope()`: that context
manager commits on exit, and this API never writes. The registry's
integrity rules (docs/REGISTRY.md) are enforced in the write path — the
UI must not have one. A read session is opened, handed to the route, and
rolled back unconditionally.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from qlab.registry.db import get_sessionmaker


def get_session() -> Iterator[Session]:
    """Yield a read-only registry session; always rolls back, never commits."""
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
