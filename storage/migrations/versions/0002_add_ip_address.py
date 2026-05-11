"""add ip_address to query_records

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-11
"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "query_records",
        sa.Column("ip_address", sa.String(45), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("query_records", "ip_address")
