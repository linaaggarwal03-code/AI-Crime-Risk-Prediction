-- =============================================================================
-- AI-Powered Urban Crime Risk & Hotspot Prediction System
-- Preliminary PostgreSQL Database Schema (Milestone 2 Deliverable)
-- Member 2: Aditi Verma (Geospatial Data Science & Full-Stack Implementation)
-- =============================================================================

-- Enable UUID extension if supported
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- Table A: crime_records
-- Stores raw historical reported crime incidents with validated geographic coordinates
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crime_records (
    id BIGSERIAL PRIMARY KEY,
    source_record_id VARCHAR(64) UNIQUE NOT NULL,      -- Chicago Police Department ID
    case_number VARCHAR(32),                           -- Official docket case number
    occurred_at TIMESTAMP WITH TIME ZONE NOT NULL,     -- Date and time of reported occurrence
    block_address VARCHAR(128),                        -- Anonymized hundred-block address
    crime_type VARCHAR(64) NOT NULL,                   -- Primary Type (e.g. THEFT, BATTERY)
    crime_description VARCHAR(128),                    -- Detailed legal sub-classification
    location_description VARCHAR(64),                  -- Physical setting (e.g. STREET, APARTMENT)
    arrest_made BOOLEAN DEFAULT FALSE,                 -- Arrest outcome
    domestic_incident BOOLEAN DEFAULT FALSE,           -- Domestic violence flag
    community_area_id INTEGER,                         -- Chicago Community Area (1 to 77)
    police_district_id INTEGER,                        -- Chicago Police District
    latitude DOUBLE PRECISION NOT NULL,                -- Validated decimal degrees latitude
    longitude DOUBLE PRECISION NOT NULL,               -- Validated decimal degrees longitude
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Geographic coordinate boundary constraints for City of Chicago
    CONSTRAINT chk_latitude_chicago CHECK (latitude >= 41.60 AND latitude <= 42.10),
    CONSTRAINT chk_longitude_chicago CHECK (longitude >= -88.05 AND longitude <= -87.45)
);

CREATE INDEX IF NOT EXISTS idx_crime_records_coords ON crime_records (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_crime_records_occurred_at ON crime_records (occurred_at);
CREATE INDEX IF NOT EXISTS idx_crime_records_type ON crime_records (crime_type);
CREATE INDEX IF NOT EXISTS idx_crime_records_area ON crime_records (community_area_id);


-- -----------------------------------------------------------------------------
-- Table B: hotspots
-- Stores identified historical geographic density clusters detected by DBSCAN
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hotspots (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER NOT NULL,                       -- Integer cluster identifier (-1 reserved for noise)
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    algorithm_name VARCHAR(64) NOT NULL DEFAULT 'DBSCAN',
    distance_metric VARCHAR(64) NOT NULL DEFAULT 'Haversine',
    eps_meters DOUBLE PRECISION NOT NULL,              -- Neighborhood radius (e.g. 500.0 meters)
    min_samples INTEGER NOT NULL,                      -- Minimum points per cluster (e.g. 50)
    incident_count INTEGER NOT NULL,                   -- Number of incident events assigned to cluster
    percentage_of_total DOUBLE PRECISION NOT NULL,     -- Percentage of total analyzed volume
    representative_latitude DOUBLE PRECISION NOT NULL, -- Geographic centroid latitude
    representative_longitude DOUBLE PRECISION NOT NULL,-- Geographic centroid longitude
    min_latitude DOUBLE PRECISION NOT NULL,            -- Spatial bounding envelope minimum lat
    max_latitude DOUBLE PRECISION NOT NULL,            -- Spatial bounding envelope maximum lat
    min_longitude DOUBLE PRECISION NOT NULL,           -- Spatial bounding envelope minimum lon
    max_longitude DOUBLE PRECISION NOT NULL,           -- Spatial bounding envelope maximum lon
    primary_crime_type VARCHAR(64),                    -- Modal crime classification in cluster
    dominant_community_area VARCHAR(32)                -- Modal community area in cluster
);

CREATE INDEX IF NOT EXISTS idx_hotspots_cluster_id ON hotspots (cluster_id);
CREATE INDEX IF NOT EXISTS idx_hotspots_centroid ON hotspots (representative_latitude, representative_longitude);


-- -----------------------------------------------------------------------------
-- Table C: predictions
-- Stores future prospective crime risk inference outputs produced by ML pipeline
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    query_latitude DOUBLE PRECISION NOT NULL,
    query_longitude DOUBLE PRECISION NOT NULL,
    target_prediction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    horizon_hours INTEGER DEFAULT 24,                  -- Prediction lookahead interval
    predicted_risk_tier VARCHAR(32) NOT NULL,          -- e.g. 'LOW', 'MODERATE', 'HIGH'
    predicted_risk_score DOUBLE PRECISION,             -- Calibrated probability [0.0, 1.0]
    predicted_crime_count DOUBLE PRECISION,            -- Expected count from regression model
    nearest_hotspot_id INTEGER REFERENCES hotspots(id) ON DELETE SET NULL,
    distance_to_hotspot_km DOUBLE PRECISION,
    model_name VARCHAR(64) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (target_prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_coords ON predictions (query_latitude, query_longitude);


-- -----------------------------------------------------------------------------
-- Table D: spatial_features
-- Precomputed point-in-time spatial features for ML training and online inference
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS spatial_features (
    id BIGSERIAL PRIMARY KEY,
    crime_record_id BIGINT REFERENCES crime_records(id) ON DELETE CASCADE,
    feature_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    spatial_density_500m_30d INTEGER NOT NULL,         -- Past-only 30d count within 500m
    dist_to_nearest_hotspot_km DOUBLE PRECISION NOT NULL,
    community_area_crime_rate DOUBLE PRECISION NOT NULL,
    is_hotspot_member BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_spatial_features_timestamp ON spatial_features (feature_timestamp);
