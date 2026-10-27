"""trial metrics: SQL NULL instead of JSON 'null'

Revision ID: e3f7c1d9a204
Revises: d8e4b1a7c352
Create Date: 2026-10-27

A trial without metrics (not evaluable, or an error) was stored as the JSON text 'null',
which `metrics IS NOT NULL` lets through: the assessor read one as a family member's
latest metrics and the night of 2026-10-27 stopped. The column now writes SQL NULL
(`JSON(none_as_null=True)`); this turns the rows already written into SQL NULL too.
"""
from collections.abc import Sequence

from alembic import op

revision: str = 'e3f7c1d9a204'
down_revision: str | Sequence[str] | None = 'd8e4b1a7c352'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("UPDATE trial SET metrics = NULL WHERE metrics = 'null'")


def downgrade() -> None:
    pass  # SQL NULL reads back as None either way
