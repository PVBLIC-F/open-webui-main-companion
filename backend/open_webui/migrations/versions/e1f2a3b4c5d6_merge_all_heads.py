"""Merge all migration heads after upstream sync

Revision ID: e1f2a3b4c5d6
Revises: a1b2c3d4e5f6, d5e6f7a8b9c0, add_gmail_sync_to_users, 544ba9a2b077
Create Date: 2026-02-13 12:00:00.000000

Merges upstream migrations (access_grant, skill, chat_message, prompt_history)
with custom migrations (gmail sync, knowledge drive, jsonb indexes).

Safety: Also ensures critical tables exist even if the migration chain
had version tracking issues due to the previous duplicate revision ID.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from open_webui.migrations.util import get_existing_tables


# revision identifiers, used by Alembic.
revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = (
    "a1b2c3d4e5f6",
    "d5e6f7a8b9c0",
    "add_gmail_sync_to_users",
    "544ba9a2b077",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    existing_tables = set(get_existing_tables())

    # Safety net: Create access_grant table if it doesn't exist.
    # This handles the case where the DB had the old duplicate revision ID
    # (a1b2c3d4e5f6 from knowledge_drive_tables) which caused Alembic to
    # incorrectly think the upstream access_grant migration was already applied.
    if "access_grant" not in existing_tables:
        op.create_table(
            "access_grant",
            sa.Column("id", sa.Text(), nullable=False, primary_key=True),
            sa.Column("resource_type", sa.Text(), nullable=False),
            sa.Column("resource_id", sa.Text(), nullable=False),
            sa.Column("principal_type", sa.Text(), nullable=False),
            sa.Column("principal_id", sa.Text(), nullable=False),
            sa.Column("permission", sa.Text(), nullable=False),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.UniqueConstraint(
                "resource_type",
                "resource_id",
                "principal_type",
                "principal_id",
                "permission",
                name="uq_access_grant_grant",
            ),
        )
        op.create_index(
            "idx_access_grant_resource",
            "access_grant",
            ["resource_type", "resource_id"],
        )
        op.create_index(
            "idx_access_grant_principal",
            "access_grant",
            ["principal_type", "principal_id"],
        )

    # Safety net: Create skill table if it doesn't exist
    if "skill" not in existing_tables:
        op.create_table(
            "skill",
            sa.Column("id", sa.String(), nullable=False, primary_key=True),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("name", sa.Text(), nullable=False, unique=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("content", sa.Text(), nullable=True),
            sa.Column("meta", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=True),
            sa.Column("updated_at", sa.BigInteger(), nullable=True),
        )

    # Safety net: Create chat_message table if it doesn't exist
    if "chat_message" not in existing_tables:
        op.create_table(
            "chat_message",
            sa.Column("id", sa.Text(), nullable=False, primary_key=True),
            sa.Column("chat_id", sa.Text(), nullable=False),
            sa.Column("user_id", sa.Text(), nullable=False),
            sa.Column("data", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=True),
            sa.Column("updated_at", sa.BigInteger(), nullable=True),
        )
        op.create_index(
            "idx_chat_message_chat_id",
            "chat_message",
            ["chat_id"],
        )
        op.create_index(
            "idx_chat_message_user_id",
            "chat_message",
            ["user_id"],
        )

    # Safety net: Create prompt_history table if it doesn't exist
    if "prompt_history" not in existing_tables:
        op.create_table(
            "prompt_history",
            sa.Column("id", sa.Text(), nullable=False, primary_key=True),
            sa.Column("user_id", sa.Text(), nullable=False),
            sa.Column("command", sa.Text(), nullable=False),
            sa.Column("created_at", sa.BigInteger(), nullable=True),
        )
        op.create_index(
            "idx_prompt_history_user_id",
            "prompt_history",
            ["user_id"],
        )


def downgrade() -> None:
    pass
