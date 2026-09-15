-- nmem-immune: Memory Immune System schema
-- PostgreSQL — run against the same database as nmem

-- ============================================================
-- QUARANTINE: memories held for review
-- ============================================================
CREATE TABLE IF NOT EXISTS immune_quarantine (
    id              BIGSERIAL PRIMARY KEY,

    -- What was quarantined
    source_table    TEXT NOT NULL,
    source_id       BIGINT NOT NULL,
    agent_id        TEXT NOT NULL,
    content_hash    TEXT NOT NULL,

    -- Why
    reason          TEXT NOT NULL,       -- contradiction, poison_pattern, low_trust, drift_detected, tainted_by_antidote
    detail          TEXT,
    skeptic_scores  JSONB NOT NULL DEFAULT '{}',

    -- Lifecycle
    status          TEXT NOT NULL DEFAULT 'quarantined',  -- quarantined | promoted | confirmed_poison | expired
    promoted_at     TIMESTAMPTZ,
    promoted_by     TEXT,               -- corroboration | aging | manual
    reviewed_at     TIMESTAMPTZ,

    -- Provenance snapshot (captured at quarantine time)
    grounding       TEXT,
    importance      INTEGER,
    source_type     TEXT,
    write_agent     TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_immune_quarantine_source
    ON immune_quarantine (source_table, source_id);
CREATE INDEX IF NOT EXISTS idx_immune_quarantine_status
    ON immune_quarantine (status);
CREATE INDEX IF NOT EXISTS idx_immune_quarantine_agent
    ON immune_quarantine (agent_id);
CREATE INDEX IF NOT EXISTS idx_immune_quarantine_hash
    ON immune_quarantine (content_hash);
-- Partial unique: only one active quarantine per source row at a time.
CREATE UNIQUE INDEX IF NOT EXISTS idx_immune_quarantine_active_source
    ON immune_quarantine (source_table, source_id) WHERE status = 'quarantined';

-- ============================================================
-- IMMUNITY RECORDS: learned poison patterns
-- ============================================================
CREATE TABLE IF NOT EXISTS immune_immunity_records (
    id              BIGSERIAL PRIMARY KEY,

    pattern_type    TEXT NOT NULL,       -- embedding_region, source_agent, content_pattern, topic_cluster
    feature_vector  JSONB NOT NULL,
    description     TEXT NOT NULL,

    confidence      FLOAT NOT NULL DEFAULT 0.5,
    hit_count       INTEGER NOT NULL DEFAULT 1,
    false_positive_count INTEGER NOT NULL DEFAULT 0,

    status          TEXT NOT NULL DEFAULT 'active',  -- active | decayed | retired
    last_reinforced TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decay_rate      FLOAT NOT NULL DEFAULT 0.05,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_immune_immunity_status
    ON immune_immunity_records (status);
CREATE INDEX IF NOT EXISTS idx_immune_immunity_type
    ON immune_immunity_records (pattern_type);

-- ============================================================
-- PROVENANCE LINKS: derived-from graph
-- ============================================================
CREATE TABLE IF NOT EXISTS immune_provenance_links (
    id              BIGSERIAL PRIMARY KEY,

    parent_table    TEXT NOT NULL,
    parent_id       BIGINT NOT NULL,
    child_table     TEXT NOT NULL,
    child_id        BIGINT NOT NULL,
    link_type       TEXT NOT NULL DEFAULT 'derived_from',

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_immune_provenance_unique
    ON immune_provenance_links (parent_table, parent_id, child_table, child_id, link_type);
CREATE INDEX IF NOT EXISTS idx_immune_provenance_parent
    ON immune_provenance_links (parent_table, parent_id);
CREATE INDEX IF NOT EXISTS idx_immune_provenance_child
    ON immune_provenance_links (child_table, child_id);

-- ============================================================
-- AUDIT LOG: all immune system actions
-- ============================================================
CREATE TABLE IF NOT EXISTS immune_audit_log (
    id              BIGSERIAL PRIMARY KEY,
    action          TEXT NOT NULL,
    target_table    TEXT,
    target_id       BIGINT,
    agent_id        TEXT,
    detail          JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_immune_audit_action
    ON immune_audit_log (action);
CREATE INDEX IF NOT EXISTS idx_immune_audit_target
    ON immune_audit_log (target_table, target_id);
CREATE INDEX IF NOT EXISTS idx_immune_audit_created
    ON immune_audit_log (created_at);
