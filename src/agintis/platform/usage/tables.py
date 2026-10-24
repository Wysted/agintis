"""SQLAlchemy Core table of the LLM usage ledger (Alembic autogenerates from it).

Money is NOT stored: rows keep tokens and are priced at read time with the price catalog.
`tenant_id` has no foreign key (tenants are not rows anywhere) and rows outlive their thread.
"""

from sqlalchemy import BigInteger, CheckConstraint, Column, DateTime, Index, Table, Text, text

from agintis.platform.db import metadata
from agintis.platform.usage.rows import Purpose

_PURPOSES = ", ".join(f"'{purpose.value}'" for purpose in Purpose)

llm_usage = Table(
    "llm_usage",
    metadata,
    Column("run_id", Text, primary_key=True),
    Column("model", Text, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("input_tokens", BigInteger, nullable=False),
    Column("output_tokens", BigInteger, nullable=False),
    Column("cached_input_tokens", BigInteger, nullable=False),
    Column("calls", BigInteger, nullable=False),
    Column("purpose", Text, nullable=False),
    Column("tenant_id", Text, nullable=True),
    Column("agent_id", Text, nullable=True),
    Column("thread_id", Text, nullable=True),
    Column("source", Text, nullable=True),
    Column("git_revision", Text, nullable=True),
    CheckConstraint(f"purpose IN ({_PURPOSES})", name="purpose_valid"),
    Index("ix_llm_usage_tenant_created", "tenant_id", "created_at"),
    Index("ix_llm_usage_purpose_created", "purpose", "created_at"),
    Index("ix_llm_usage_thread", "thread_id", postgresql_where=text("thread_id IS NOT NULL")),
)
