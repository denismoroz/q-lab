"""assessment table

Revision ID: f3b9d1c7a206
Revises: e7c1a4d9b352
Create Date: 2026-10-24

The assessor agent's second opinion on a trial's verdict
(qlab.agents.assessor, docs/ASSESSOR.md).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'f3b9d1c7a206'
down_revision: str | Sequence[str] | None = 'e7c1a4d9b352'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'assessment',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('idea_id', sa.String(), nullable=False),
        sa.Column('trial_id', sa.Integer(), nullable=False),
        sa.Column('at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('view', sa.String(), nullable=False),
        sa.Column('summary', sa.String(), nullable=False),
        sa.Column('accepted', sa.JSON(), nullable=False),
        sa.Column('rejected', sa.JSON(), nullable=False),
        sa.Column('checks', sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(['idea_id'], ['idea.id']),
        sa.ForeignKeyConstraint(['trial_id'], ['trial.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('assessment', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_assessment_idea_id'), ['idea_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_assessment_trial_id'), ['trial_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('assessment', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_assessment_trial_id'))
        batch_op.drop_index(batch_op.f('ix_assessment_idea_id'))
    op.drop_table('assessment')
