-- ---------------------------------------------------------------------------
-- Japan RE Screener — Supabase (Postgres) schema, SCHEMA_VERSION 19
-- ---------------------------------------------------------------------------
-- GENERATED FROM the live SQLite schema so the two cannot drift. init_db()
-- creates all of this automatically on first connect. For an EXISTING
-- deployment use migration_v18_to_v19.sql instead of rerunning this file.

CREATE TABLE IF NOT EXISTS app_meta (
    key                          TEXT PRIMARY KEY,
    value                        TEXT
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id                           SERIAL PRIMARY KEY,
    prefecture                   TEXT NOT NULL,
    sqm_rate                     DOUBLE PRECISION,
    sample_n                     INTEGER,
    source                       TEXT,
    updated_at                   TEXT
);

CREATE TABLE IF NOT EXISTS dev_tile_stats (
    z                            INTEGER,
    x                            INTEGER,
    y                            INTEGER,
    kodo_count                   INTEGER,
    road_count                   INTEGER,
    checked_at                   TEXT,
    PRIMARY KEY (z, x, y)
);

CREATE TABLE IF NOT EXISTS historical_picks (
    id                           SERIAL PRIMARY KEY,
    listing_id                   TEXT NOT NULL,
    url                          TEXT NOT NULL,
    portal                       TEXT,
    title                        TEXT,
    ptype                        TEXT,
    prefecture                   TEXT,
    station_min                  DOUBLE PRECISION,
    building_age                 DOUBLE PRECISION,
    area_sqm                     DOUBLE PRECISION,
    price_yen                    DOUBLE PRECISION,
    gross_yield                  DOUBLE PRECISION,
    score                        DOUBLE PRECISION,
    factor_snapshot              TEXT,
    recommendation_date          TEXT NOT NULL,
    current_status               TEXT DEFAULT 'pending',
    current_price                DOUBLE PRECISION,
    days_listed                  INTEGER,
    outcome                      INTEGER,
    evaluated                    INTEGER DEFAULT 0,
    zoning                       TEXT,
    far_pct                      DOUBLE PRECISION,
    lat                          DOUBLE PRECISION,
    lon                          DOUBLE PRECISION,
    hazard_flags                 TEXT,
    dev_flags                    TEXT,
    net_yield                    DOUBLE PRECISION,
    monthly_fees_yen             DOUBLE PRECISION,
    seismic                      TEXT,
    yield_basis                  TEXT,
    net_yield_invested           DOUBLE PRECISION,
    area_tier                    TEXT,
    pop_outlook_pct              DOUBLE PRECISION,
    nickname                     TEXT,
    building_name                TEXT,
    unit_number                  TEXT,
    unit_floor                   TEXT,
    source_document_name         TEXT,
    ingestion_channel            TEXT,
    source_type                  TEXT,
    agency_name                  TEXT,
    agent_name                   TEXT,
    agency_phone                 TEXT,
    agency_address               TEXT,
    agency_role                  TEXT,
    source_document_date         TEXT,
    seller_name                  TEXT,
    management_company           TEXT,
    rent_guarantee_company       TEXT,
    extracted_entities           TEXT,
    provenance                   TEXT,
    monthly_rent_yen             DOUBLE PRECISION,
    market_rent_yen              DOUBLE PRECISION,
    occupancy                    TEXT,
    structure                    TEXT,
    size_resilience              DOUBLE PRECISION,
    size_model_mode              TEXT,
    size_adjustment              DOUBLE PRECISION,
    size_evidence                DOUBLE PRECISION,
    size_breakdown               TEXT,
    rent_source                  TEXT,
    contractual_gross_pct        DOUBLE PRECISION,
    gross_gap_pp                 DOUBLE PRECISION,
    source_url                   TEXT,
    rent_evidence_summary        TEXT,
    local_median_rent_per_sqm    DOUBLE PRECISION,
    rental_comparable_count      INTEGER,
    rent_median_similarity       DOUBLE PRECISION,
    rent_recency_months          DOUBLE PRECISION,
    median_listing_duration_days DOUBLE PRECISION,
    rent_source_kind             TEXT,
    manual                       INTEGER DEFAULT 0,
    city_code                    TEXT
);

CREATE TABLE IF NOT EXISTS municipality_benchmarks (
    city_code                    TEXT PRIMARY KEY,
    pref_code                    TEXT,
    avg_price_sqm                DOUBLE PRECISION,
    sample_count                 INTEGER,
    updated_at                   TEXT
);

CREATE TABLE IF NOT EXISTS municipality_quarterly (
    city_code                    TEXT,
    year                         INTEGER,
    quarter                      INTEGER,
    median_ppsm                  DOUBLE PRECISION,
    sample_count                 INTEGER,
    updated_at                   TEXT,
    PRIMARY KEY (city_code, year, quarter)
);

CREATE TABLE IF NOT EXISTS property_revisions (
    revision_id                  SERIAL PRIMARY KEY,
    historical_pick_id           INTEGER NOT NULL,
    revision_number              INTEGER NOT NULL,
    revision_kind                TEXT NOT NULL,
    changed_at                   TEXT NOT NULL,
    changed_by                   TEXT,
    change_reason                TEXT,
    source_name                  TEXT,
    source_url                   TEXT,
    source_document_name         TEXT,
    ingestion_channel            TEXT,
    fields_before                TEXT,
    fields_after                 TEXT,
    changed_fields               TEXT,
    provenance                   TEXT,
    refreshed_score              DOUBLE PRECISION,
    refreshed_analysis           TEXT,
    original_score_preserved     INTEGER DEFAULT 1,
    UNIQUE (historical_pick_id, revision_number)
);

CREATE TABLE IF NOT EXISTS rental_observations (
    observation_id               SERIAL PRIMARY KEY,
    historical_pick_id           INTEGER,
    source_url                   TEXT,
    observed_at                  TEXT,
    monthly_rent_yen             DOUBLE PRECISION NOT NULL,
    management_fee_yen           DOUBLE PRECISION,
    total_monthly_cost_yen       DOUBLE PRECISION,
    area_sqm                     DOUBLE PRECISION,
    floor_plan                   TEXT,
    unit_floor                   TEXT,
    building_age                 DOUBLE PRECISION,
    listing_status               TEXT,
    evidence_kind                TEXT,
    source_label                 TEXT,
    raw_context                  TEXT,
    selected                     INTEGER DEFAULT 0,
    confidence                   DOUBLE PRECISION,
    created_at                   TEXT
);

CREATE TABLE IF NOT EXISTS system_weights (
    id                           SERIAL PRIMARY KEY,
    timestamp                    TEXT NOT NULL,
    factor_name                  TEXT NOT NULL,
    current_weight               DOUBLE PRECISION,
    note                         TEXT
);

CREATE TABLE IF NOT EXISTS transaction_comparables (
    comp_id                      TEXT PRIMARY KEY,
    city_code                    TEXT NOT NULL,
    year                         INTEGER,
    quarter                      INTEGER,
    property_kind                TEXT,
    floor_plan                   TEXT,
    area_sqm                     DOUBLE PRECISION,
    building_year                INTEGER,
    structure                    TEXT,
    price_yen                    DOUBLE PRECISION,
    price_per_sqm                DOUBLE PRECISION,
    fetched_at                   TEXT,
    station_min                  DOUBLE PRECISION,
    source_type                  TEXT,
    observed_at                  TEXT
);

-- Indexes (also created by init_db()).
CREATE INDEX IF NOT EXISTS idx_comps_city ON transaction_comparables(city_code, year, quarter);
CREATE INDEX IF NOT EXISTS idx_picks_evaluated ON historical_picks(evaluated, manual);
CREATE INDEX IF NOT EXISTS idx_picks_listing ON historical_picks(listing_id);
CREATE INDEX IF NOT EXISTS idx_picks_recdate ON historical_picks(recommendation_date);
CREATE INDEX IF NOT EXISTS idx_picks_status ON historical_picks(current_status);
CREATE INDEX IF NOT EXISTS idx_quarterly_city ON municipality_quarterly(city_code);
CREATE INDEX IF NOT EXISTS idx_rentobs_pick ON rental_observations(historical_pick_id);
CREATE INDEX IF NOT EXISTS idx_rentobs_url ON rental_observations(source_url);
CREATE INDEX IF NOT EXISTS idx_revisions_changed_at ON property_revisions(changed_at);
CREATE INDEX IF NOT EXISTS idx_revisions_pick ON property_revisions(historical_pick_id, revision_number);
CREATE INDEX IF NOT EXISTS idx_weights_ts ON system_weights(factor_name, id);

-- Deliberately NOT applied, with reasons:
--   * benchmarks has no UNIQUE(prefecture): append-only history; get_sqm_rate
--     reads the latest row, so a unique constraint would reject every refresh.
--   * system_weights.timestamp stays TEXT so identical SQL runs on SQLite.
