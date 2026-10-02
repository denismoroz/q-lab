"""needs-forward trial route

Revision ID: a3c9e1f20b47
Revises: cfe1f1bc160d
Create Date: 2026-10-02

docs/FIT_VS_FORWARD.md: a run that passes everything the selection period
can show, or fails a forward test still too short to tell, is routed
`needs-forward`. The enum column is widened to hold the new value; no row
changes.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'a3c9e1f20b47'
down_revision: str | Sequence[str] | None = 'cfe1f1bc160d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = ('reject', 'shelf', 'paper', 'needs-infrastructure', 'needs-more-data',
        'not-evaluable', 'error')
_NEW = ('reject', 'shelf', 'paper', 'needs-infrastructure', 'needs-forward',
        'needs-more-data', 'not-evaluable', 'error')


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
