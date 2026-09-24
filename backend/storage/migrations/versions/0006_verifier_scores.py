"""add verifier_scores JSON column to query_records

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-24
"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "query_records",
        sa.Column("verifier_scores", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("query_records", "verifier_scores")
