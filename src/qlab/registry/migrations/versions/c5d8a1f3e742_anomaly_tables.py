"""anomaly tables

Revision ID: c5d8a1f3e742
Revises: b4c7e1a9d305
Create Date: 2026-10-28

The anomaly detector's flags for a judged trial and the investigator agent's dossier on a
flagged one (qlab.anomalies, docs/ANOMALIES.md).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'c5d8a1f3e742'
down_revision: str | Sequence[str] | None = 'b4c7e1a9d305'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'anomaly_scan',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('idea_id', sa.String(), nullable=False),
        sa.Column('trial_id', sa.Integer(), nullable=False),
        sa.Column('config_version', sa.String(), nullable=False),
        sa.Column('scanned_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('route', sa.String(), nullable=True),
        sa.Column('fired', sa.JSON(), nullable=False),
        sa.Column('flags', sa.JSON(), nullable=False),
        sa.Column('facts', sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(['idea_id'], ['idea.id']),
        sa.ForeignKeyConstraint(['trial_id'], ['trial.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('trial_id', 'config_version', name='uq_anomaly_scan_trial_version'),
    )
    with op.batch_alter_table('anomaly_scan', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_anomaly_scan_idea_id'), ['idea_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_anomaly_scan_trial_id'), ['trial_id'], unique=False)
    op.create_table(
        'anomaly_investigation',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('idea_id', sa.String(), nullable=False),
        sa.Column('trial_id', sa.Integer(), nullable=False),
        sa.Column('at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('flags_fingerprint', sa.String(), nullable=False),
        sa.Column('cause', sa.String(), nullable=False),
        sa.Column('summary', sa.String(), nullable=False),
        sa.Column('dossier', sa.String(), nullable=False),
        sa.Column('evidence', sa.JSON(), nullable=False),
        sa.Column('rejected', sa.JSON(), nullable=False),
        sa.Column('proposals', sa.JSON(), nullable=False),
        sa.Column('caught_by', sa.JSON(), nullable=False),
        sa.Column('new_check', sa.String(), nullable=True),
        sa.Column('new_check_status', sa.String(), nullable=False),
        sa.Column('programmer_requests', sa.JSON(), nullable=False),
        sa.Column('tokens', sa.Integer(), nullable=False),
        sa.Column('calls', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['idea_id'], ['idea.id']),
        sa.ForeignKeyConstraint(['trial_id'], ['trial.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('anomaly_investigation', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_anomaly_investigation_idea_id'), ['idea_id'],
                              unique=False)
        batch_op.create_index(batch_op.f('ix_anomaly_investigation_trial_id'), ['trial_id'],
                              unique=False)


def downgrade() -> None:
    with op.batch_alter_table('anomaly_investigation', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_anomaly_investigation_trial_id'))
        batch_op.drop_index(batch_op.f('ix_anomaly_investigation_idea_id'))
    op.drop_table('anomaly_investigation')
    with op.batch_alter_table('anomaly_scan', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_anomaly_scan_trial_id'))
        batch_op.drop_index(batch_op.f('ix_anomaly_scan_idea_id'))
    op.drop_table('anomaly_scan')
