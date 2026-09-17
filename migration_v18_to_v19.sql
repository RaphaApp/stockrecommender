-- ---------------------------------------------------------------------------
-- Japan RE Screener — migration, schema_version 18 -> 19
-- ---------------------------------------------------------------------------
-- Paste this whole file into the Supabase SQL Editor and run it once.
--
-- Rerun-safe: every statement is IF NOT EXISTS or an idempotent UPSERT, so
-- running it twice changes nothing the second time. It is wrapped in a single
-- transaction, so a failure part-way leaves the database exactly as it was.
--
-- This migration ADDS the post-screening revision trail. It does not alter or
-- delete any existing row: historical_picks remains the recommendation
-- snapshot, and later edits append to property_revisions instead.

BEGIN;

CREATE TABLE IF NOT EXISTS property_revisions (
    revision_id              BIGSERIAL PRIMARY KEY,
    historical_pick_id       INTEGER NOT NULL,
    revision_number          INTEGER NOT NULL,
    revision_kind            TEXT NOT NULL,
    changed_at               TEXT NOT NULL,
    changed_by               TEXT,
    change_reason            TEXT,
    source_name              TEXT,
    source_url               TEXT,
    source_document_name     TEXT,
    ingestion_channel        TEXT,
    fields_before            TEXT,
    fields_after             TEXT,
    changed_fields           TEXT,
    provenance               TEXT,
    refreshed_score          DOUBLE PRECISION,
    refreshed_analysis       TEXT,
    original_score_preserved INTEGER DEFAULT 1,
    UNIQUE (historical_pick_id, revision_number)
);

-- Revision numbering correctness depends on this constraint, so assert it
-- exists even if the table was created by an earlier partial run.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'property_revisions_historical_pick_id_revision_number_key'
    ) THEN
        ALTER TABLE property_revisions
            ADD CONSTRAINT property_revisions_historical_pick_id_revision_number_key
            UNIQUE (historical_pick_id, revision_number);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_revisions_pick
    ON property_revisions(historical_pick_id, revision_number);
CREATE INDEX IF NOT EXISTS idx_revisions_changed_at
    ON property_revisions(changed_at);

-- v18 columns, repeated defensively: a deployment that skipped a step still
-- ends up correct, and ADD COLUMN IF NOT EXISTS is a no-op where present.
ALTER TABLE historical_picks ADD COLUMN IF NOT EXISTS source_url TEXT;
ALTER TABLE historical_picks ADD COLUMN IF NOT EXISTS rent_evidence_summary TEXT;
ALTER TABLE historical_picks ADD COLUMN IF NOT EXISTS local_median_rent_per_sqm DOUBLE PRECISION;
ALTER TABLE historical_picks ADD COLUMN IF NOT EXISTS rental_comparable_count INTEGER;
ALTER TABLE historical_picks ADD COLUMN IF NOT EXISTS rent_median_similarity DOUBLE PRECISION;
ALTER TABLE historical_picks ADD COLUMN IF NOT EXISTS rent_recency_months DOUBLE PRECISION;
ALTER TABLE historical_picks ADD COLUMN IF NOT EXISTS median_listing_duration_days DOUBLE PRECISION;
ALTER TABLE historical_picks ADD COLUMN IF NOT EXISTS rent_source_kind TEXT;

-- Record the new version. app_meta.key is the primary key, so ON CONFLICT
-- makes this an idempotent upsert rather than a duplicate-key failure.
INSERT INTO app_meta (key, value) VALUES ('schema_version', '19')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

COMMIT;

-- Verify afterwards:
--   SELECT value FROM app_meta WHERE key = 'schema_version';      -- expect 19
--   SELECT COUNT(*) FROM property_revisions;                      -- expect 0 on a fresh migration
--   SELECT COUNT(*) FROM historical_picks;                        -- unchanged
