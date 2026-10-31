"""pipeline journal: pipeline_run, pipeline_step_run, pipeline_item

Revision ID: d3a7f1c8e905
Revises: a9e4c7d1b630
Create Date: 2026-10-31

The night as steps (qlab.nightflow, docs/PIPELINE.md): one row per run, one per step in a run,
one per item of work inside a step. Written live by the runner. New tables only.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'd3a7f1c8e905'
down_revision: str | Sequence[str] | None = 'a9e4c7d1b630'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'pipeline_run',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('night_date', sa.Date(), nullable=False),
        sa.Column('trigger', sa.String(), nullable=False),
        sa.Column('state', sa.String(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('pid', sa.Integer(), nullable=True),
        sa.Column('concurrency', sa.Integer(), nullable=False),
        sa.Column('steps_off', sa.JSON(), nullable=False),
        sa.Column('only', sa.JSON(), nullable=True),
        sa.Column('restart', sa.String(), nullable=True),
        sa.Column('command', sa.String(), nullable=True),
        sa.Column('note', sa.String(), nullable=True),
        sa.Column('approximate', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('pipeline_run', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pipeline_run_night_date'), ['night_date'], unique=False)
    op.create_table(
        'pipeline_step_run',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('step_id', sa.String(), nullable=False),
        sa.Column('state', sa.String(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_s', sa.Float(), nullable=True),
        sa.Column('items_done', sa.Integer(), nullable=False),
        sa.Column('items_failed', sa.Integer(), nullable=False),
        sa.Column('tokens', sa.Integer(), nullable=True),
        sa.Column('error', sa.String(), nullable=True),
        sa.Column('blocked_by', sa.String(), nullable=True),
        sa.Column('note', sa.String(), nullable=True),
        sa.Column('approximate', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['pipeline_run.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('pipeline_step_run', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pipeline_step_run_run_id'), ['run_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pipeline_step_run_step_id'), ['step_id'], unique=False)
    op.create_table(
        'pipeline_item',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('step_run_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('outcome', sa.String(), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_s', sa.Float(), nullable=True),
        sa.Column('detail', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['step_run_id'], ['pipeline_step_run.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('pipeline_item', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pipeline_item_step_run_id'), ['step_run_id'],
                              unique=False)


def downgrade() -> None:
    with op.batch_alter_table('pipeline_item', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pipeline_item_step_run_id'))
    op.drop_table('pipeline_item')
    with op.batch_alter_table('pipeline_step_run', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pipeline_step_run_step_id'))
        batch_op.drop_index(batch_op.f('ix_pipeline_step_run_run_id'))
    op.drop_table('pipeline_step_run')
    with op.batch_alter_table('pipeline_run', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pipeline_run_night_date'))
    op.drop_table('pipeline_run')
