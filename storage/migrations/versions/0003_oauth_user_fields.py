"""add oauth_provider and oauth_id to users, make hashed_password nullable

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-11
"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("users", "hashed_password", existing_type=sa.String(256), nullable=True)
    op.add_column("users", sa.Column("oauth_provider", sa.String(32), nullable=True))
    op.add_column("users", sa.Column("oauth_id", sa.String(128), nullable=True))
    op.create_index("ix_users_oauth_id", "users", ["oauth_id"])


def downgrade() -> None:
    op.drop_index("ix_users_oauth_id", table_name="users")
    op.drop_column("users", "oauth_id")
    op.drop_column("users", "oauth_provider")
    op.alter_column("users", "hashed_password", existing_type=sa.String(256), nullable=False)
