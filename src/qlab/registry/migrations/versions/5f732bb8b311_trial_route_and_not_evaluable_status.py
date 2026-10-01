"""trial route and not-evaluable status

Revision ID: 5f732bb8b311
Revises: 8d6f4fd31f3c
Create Date: 2026-10-01 15:28:25.279075

docs/TASKS.md T24 and T31: a run's route is persisted on the trial, a run
that could not test its idea's strategy gets its own trial status, and a
status change records the run that caused it.

Existing trials keep `route = NULL`: their routes were printed and never
stored, and rebuilding them would mean guessing the deployable capital each
run was handed.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '5f732bb8b311'
down_revision: str | Sequence[str] | None = '8d6f4fd31f3c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FK_TRANSITION_TRIAL = 'fk_stage_transition_trial_id_trial'
_ROUTES = ('reject', 'shelf', 'paper', 'needs-more-data', 'not-evaluable', 'error')


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('stage_transition', schema=None) as batch_op:
        batch_op.add_column(sa.Column('trial_id', sa.Integer(), nullable=True))
        batch_op.create_index(
            batch_op.f('ix_stage_transition_trial_id'), ['trial_id'], unique=False
        )
        batch_op.create_foreign_key(_FK_TRANSITION_TRIAL, 'trial', ['trial_id'], ['id'])

    with op.batch_alter_table('trial', schema=None) as batch_op:
        batch_op.add_column(sa.Column('route', sa.Enum(*_ROUTES, name='trialroute'), nullable=True))
        batch_op.add_column(sa.Column('route_reason', sa.String(), nullable=True))
        batch_op.alter_column(
            'status',
            existing_type=sa.VARCHAR(length=5),
            type_=sa.Enum('ok', 'error', 'not-evaluable', name='trialstatus'),
            existing_nullable=False,
        )
        batch_op.create_index(batch_op.f('ix_trial_route'), ['route'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('trial', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_trial_route'))
        batch_op.alter_column(
            'status',
            existing_type=sa.Enum('ok', 'error', 'not-evaluable', name='trialstatus'),
            type_=sa.VARCHAR(length=5),
            existing_nullable=False,
        )
        batch_op.drop_column('route_reason')
        batch_op.drop_column('route')

    with op.batch_alter_table('stage_transition', schema=None) as batch_op:
        batch_op.drop_constraint(_FK_TRANSITION_TRIAL, type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_stage_transition_trial_id'))
        batch_op.drop_column('trial_id')
