"""needs-infrastructure trial route

Revision ID: cfe1f1bc160d
Revises: 5f732bb8b311
Create Date: 2026-10-01

docs/TASKS.md T35: a run whose strategy rules all pass but whose
infrastructure rules fail is routed `needs-infrastructure`, not `reject`.
The enum column is widened to hold the new, longer value; no row changes.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'cfe1f1bc160d'
down_revision: str | Sequence[str] | None = '5f732bb8b311'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = ('reject', 'shelf', 'paper', 'needs-more-data', 'not-evaluable', 'error')
_NEW = ('reject', 'shelf', 'paper', 'needs-infrastructure', 'needs-more-data',
        'not-evaluable', 'error')


def upgrade() -> None:
    with op.batch_alter_table('trial', schema=None) as batch_op:
        batch_op.alter_column(
            'route',
            existing_type=sa.Enum(*_OLD, name='trialroute'),
            type_=sa.Enum(*_NEW, name='trialroute'),
            existing_nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table('trial', schema=None) as batch_op:
        batch_op.alter_column(
            'route',
            existing_type=sa.Enum(*_NEW, name='trialroute'),
            type_=sa.Enum(*_OLD, name='trialroute'),
            existing_nullable=True,
        )
