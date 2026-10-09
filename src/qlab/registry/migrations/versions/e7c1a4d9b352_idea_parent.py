"""idea.parent_id

Revision ID: e7c1a4d9b352
Revises: d2a8b6e4f019
Create Date: 2026-10-09

An idea may be a variant of another (qlab.registry.families): the registry
and the console group variants under their parent strategy.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'e7c1a4d9b352'
down_revision: str | Sequence[str] | None = 'd2a8b6e4f019'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('idea', schema=None) as batch_op:
        batch_op.add_column(sa.Column('parent_id', sa.String(), nullable=True))
        batch_op.create_index(batch_op.f('ix_idea_parent_id'), ['parent_id'], unique=False)
        batch_op.create_foreign_key('fk_idea_parent_id_idea', 'idea', ['parent_id'], ['id'])


def downgrade() -> None:
    with op.batch_alter_table('idea', schema=None) as batch_op:
        batch_op.drop_constraint('fk_idea_parent_id_idea', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_idea_parent_id'))
        batch_op.drop_column('parent_id')
