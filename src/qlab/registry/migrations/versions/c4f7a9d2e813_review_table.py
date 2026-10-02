"""review table

Revision ID: c4f7a9d2e813
Revises: b81d4e2a9c55
Create Date: 2026-10-02

docs/REVIEWER.md: the reviewer agent's findings, with verified evidence.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'c4f7a9d2e813'
down_revision: str | Sequence[str] | None = 'b81d4e2a9c55'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'review',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('idea_id', sa.String(), nullable=False),
        sa.Column('review_key', sa.String(), nullable=False),
        sa.Column('at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('sources', sa.JSON(), nullable=False),
        sa.Column('accepted', sa.JSON(), nullable=False),
        sa.Column('rejected', sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(['idea_id'], ['idea.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('review', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_review_idea_id'), ['idea_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_review_review_key'), ['review_key'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('review', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_review_review_key'))
        batch_op.drop_index(batch_op.f('ix_review_idea_id'))
    op.drop_table('review')
