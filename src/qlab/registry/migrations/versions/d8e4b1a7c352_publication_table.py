"""publication table

Revision ID: d8e4b1a7c352
Revises: c9d2a7f3b861
Create Date: 2026-10-25

Papers seen by the publication feed (qlab.publications): what was fetched,
whether the topic rule marks it a candidate, and what became of it.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'd8e4b1a7c352'
down_revision: str | Sequence[str] | None = 'c9d2a7f3b861'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'publication',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('title_key', sa.String(), nullable=False),
        sa.Column('authors', sa.JSON(), nullable=False),
        sa.Column('published', sa.DateTime(timezone=True), nullable=False),
        sa.Column('abstract', sa.String(), nullable=False),
        sa.Column('link', sa.String(), nullable=False),
        sa.Column('categories', sa.JSON(), nullable=False),
        sa.Column('first_seen', sa.DateTime(timezone=True), nullable=False),
        sa.Column('filter_version', sa.String(), nullable=True),
        sa.Column('candidate', sa.Boolean(), nullable=False),
        sa.Column('score', sa.Integer(), nullable=False),
        sa.Column('matched', sa.JSON(), nullable=False),
        sa.Column('handed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('scout_note', sa.String(), nullable=True),
        sa.Column('draft', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('publication', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_publication_source'), ['source'], unique=False)
        batch_op.create_index(batch_op.f('ix_publication_title_key'), ['title_key'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('publication', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_publication_title_key'))
        batch_op.drop_index(batch_op.f('ix_publication_source'))
    op.drop_table('publication')
