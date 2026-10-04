"""detector_trial table

Revision ID: d2a8b6e4f019
Revises: c4f7a9d2e813
Create Date: 2026-10-04

docs/REGIME_DETECT.md: every regime-detector variant measured, on which
period, with its metrics -- the selection among detectors on record.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'd2a8b6e4f019'
down_revision: str | Sequence[str] | None = 'c4f7a9d2e813'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'detector_trial',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('family', sa.String(), nullable=False),
        sa.Column('variant', sa.String(), nullable=False),
        sa.Column('period', sa.String(), nullable=False),
        sa.Column('data_start', sa.Date(), nullable=False),
        sa.Column('data_end', sa.Date(), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.Column('script', sa.String(), nullable=False),
        sa.Column('code_sha', sa.String(), nullable=False),
        sa.Column('chosen', sa.Boolean(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('notes', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('detector_trial', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_detector_trial_family'), ['family'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('detector_trial', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_detector_trial_family'))
    op.drop_table('detector_trial')
