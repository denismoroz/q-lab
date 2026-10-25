"""research_note table

Revision ID: a5d2c8e7b194
Revises: f3b9d1c7a206
Create Date: 2026-10-25

The researcher agent's memos kept in the registry
(qlab.agents.researcher, docs/RESEARCHER.md).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'a5d2c8e7b194'
down_revision: str | Sequence[str] | None = 'f3b9d1c7a206'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'research_note',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('idea_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('memo_path', sa.String(), nullable=False),
        sa.Column('memo', sa.String(), nullable=False),
        sa.Column('experiments', sa.JSON(), nullable=False),
        sa.Column('problems', sa.JSON(), nullable=False),
        sa.Column('tokens', sa.Integer(), nullable=True),
        sa.Column('model', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['idea_id'], ['idea.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('idea_id', 'memo_path', name='uq_research_note_idea_memo'),
    )
    with op.batch_alter_table('research_note', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_research_note_idea_id'), ['idea_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('research_note', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_research_note_idea_id'))
    op.drop_table('research_note')
