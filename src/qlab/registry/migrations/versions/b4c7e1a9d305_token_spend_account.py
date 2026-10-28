"""token_spend.account: the Claude account a call ran under

Revision ID: b4c7e1a9d305
Revises: e3f7c1d9a204
Create Date: 2026-10-28

The night had been spending the owner's WORK account (denis.moroz@toptal.com, organisation
Stablepay, a Team plan) since at least 2026-10-08: every ledger row's weekly reset is
Thursday 18:00 UTC, which is that account's, not the owner's own (moroz.denis@gmail.com,
Pro, Thursday 05:00 UTC). The windows a row reports belong to its account, so the guard's
calibration and last reading are read per account (qlab.budget.guard). Rows already written
all came from the work account and are marked so; new rows record the checked email.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'b4c7e1a9d305'
down_revision: str | Sequence[str] | None = 'e3f7c1d9a204'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('token_spend', schema=None) as batch_op:
        batch_op.add_column(sa.Column('account', sa.String(), nullable=True))
        batch_op.create_index(batch_op.f('ix_token_spend_account'), ['account'], unique=False)
    op.execute("UPDATE token_spend SET account = 'denis.moroz@toptal.com' "
               "WHERE account IS NULL")


def downgrade() -> None:
    with op.batch_alter_table('token_spend', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_token_spend_account'))
        batch_op.drop_column('account')
