"""token_spend: budget guard columns

Revision ID: b81d4e2a9c55
Revises: a3c9e1f20b47
Create Date: 2026-10-02

docs/BUDGET.md: every agent call records its cache traffic, model and the
subscription windows' utilization it reported. All new columns are
nullable; no row changes.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'b81d4e2a9c55'
down_revision: str | Sequence[str] | None = 'a3c9e1f20b47'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = (
    ('model', sa.String()),
    ('tokens_cache_write', sa.Integer()),
    ('tokens_cache_read', sa.Integer()),
    ('five_hour_util', sa.Float()),
    ('seven_day_util', sa.Float()),
    ('seven_day_resets_at', sa.DateTime(timezone=True)),
    ('status', sa.String()),
)


def upgrade() -> None:
    with op.batch_alter_table('token_spend', schema=None) as batch_op:
        for name, type_ in _COLUMNS:
            batch_op.add_column(sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('token_spend', schema=None) as batch_op:
        for name, _ in reversed(_COLUMNS):
            batch_op.drop_column(name)
