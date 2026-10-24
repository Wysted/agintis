"""llm_usage: one ledger row per (run, model) with the tokens an LLM run spent

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-24

Tokens only, never USD: reports price them at read time with the versioned price file, so a
corrected or new price re-prices history. `(run_id, model)` is the key, which makes a repeated
write a no-op (`ON CONFLICT DO NOTHING`). `tenant_id` has no foreign key (tenants are not rows).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_usage",
        sa.Column("run_id", sa.Text, nullable=False),
        sa.Column("model", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_tokens", sa.BigInteger, nullable=False),
        sa.Column("output_tokens", sa.BigInteger, nullable=False),
        sa.Column("cached_input_tokens", sa.BigInteger, nullable=False),
        sa.Column("calls", sa.BigInteger, nullable=False),
        sa.Column("purpose", sa.Text, nullable=False),
        sa.Column("tenant_id", sa.Text, nullable=True),
        sa.Column("agent_id", sa.Text, nullable=True),
        sa.Column("thread_id", sa.Text, nullable=True),
        sa.Column("source", sa.Text, nullable=True),
        sa.Column("git_revision", sa.Text, nullable=True),
        sa.PrimaryKeyConstraint("run_id", "model", name=op.f("pk_llm_usage")),
        sa.CheckConstraint(
            "purpose IN ('chat', 'dump', 'judge', 'eval', 'calibrate')",
            name=op.f("ck_llm_usage_purpose_valid"),
        ),
    )
    op.create_index("ix_llm_usage_tenant_created", "llm_usage", ["tenant_id", "created_at"])
    op.create_index("ix_llm_usage_purpose_created", "llm_usage", ["purpose", "created_at"])
    op.create_index(
        "ix_llm_usage_thread",
        "llm_usage",
        ["thread_id"],
        postgresql_where=sa.text("thread_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_table("llm_usage")  # indexes go with the table
