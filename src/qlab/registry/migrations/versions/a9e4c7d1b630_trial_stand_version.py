"""trial.stand_version

Revision ID: a9e4c7d1b630
Revises: c5d8a1f3e742
Create Date: 2026-10-28

Which way of measuring a run each new trial was made under (qlab.harness.version, docs/RESTAMP.md).
Old trials keep NULL: older than the first version. A plain ADD COLUMN, not a batch rewrite: SQLite
adds a nullable column without copying the table, and the trial table is a gigabyte.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'a9e4c7d1b630'
down_revision: str | Sequence[str] | None = 'c5d8a1f3e742'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('trial', sa.Column('stand_version', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('trial', 'stand_version')
