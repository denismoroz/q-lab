"""decay_check table

Revision ID: c9d2a7f3b861
Revises: b7e3f1a9c425
Create Date: 2026-10-25

One night's comparison of a running strategy's realised results with the
distribution of same-length windows of its own stand history (qlab.decay).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'c9d2a7f3b861'
down_revision: str | Sequence[str] | None = 'b7e3f1a9c425'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'decay_check',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('idea_id', sa.String(), nullable=False),
        sa.Column('book', sa.String(), nullable=False),
        sa.Column('at', sa.Date(), nullable=False),
        sa.Column('window_start', sa.Date(), nullable=False),
        sa.Column('window_end', sa.Date(), nullable=False),
        sa.Column('realised_return', sa.Float(), nullable=False),
        sa.Column('return_percentile', sa.Float(), nullable=True),
        sa.Column('detail', sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(['idea_id'], ['idea.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('idea_id', 'at', name='uq_decay_check_idea_day'),
    )
    with op.batch_alter_table('decay_check', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_decay_check_idea_id'), ['idea_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('decay_check', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_decay_check_idea_id'))
    op.drop_table('decay_check')
