"""paper_check table

Revision ID: b7e3f1a9c425
Revises: a5d2c8e7b194
Create Date: 2026-10-25

One night's comparison of a production paper book with the stand's forward
test of the same configuration over the same window (qlab.paper_check).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'b7e3f1a9c425'
down_revision: str | Sequence[str] | None = 'a5d2c8e7b194'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'paper_check',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('idea_id', sa.String(), nullable=False),
        sa.Column('book', sa.String(), nullable=False),
        sa.Column('at', sa.Date(), nullable=False),
        sa.Column('window_start', sa.Date(), nullable=False),
        sa.Column('window_end', sa.Date(), nullable=False),
        sa.Column('paper_return', sa.Float(), nullable=False),
        sa.Column('stand_return', sa.Float(), nullable=True),
        sa.Column('stand_trial_id', sa.Integer(), nullable=True),
        sa.Column('replay_return', sa.Float(), nullable=True),
        sa.Column('stand_window_start', sa.Date(), nullable=True),
        sa.Column('stand_window_end', sa.Date(), nullable=True),
        sa.Column('note', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['idea_id'], ['idea.id']),
        sa.ForeignKeyConstraint(['stand_trial_id'], ['trial.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('idea_id', 'at', name='uq_paper_check_idea_day'),
    )
    with op.batch_alter_table('paper_check', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_paper_check_idea_id'), ['idea_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('paper_check', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_paper_check_idea_id'))
    op.drop_table('paper_check')
