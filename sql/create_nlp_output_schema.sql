-- =====================================================================
-- Amazon VOC Pipeline - BigQuery NLP Output Layer Schema
-- =====================================================================
-- This script creates the schema for NLP-generated output tables.
--
-- These tables are populated by the Python-based NLP pipeline and serve
-- as upstream inputs for dbt transformation models.
--
-- dbt models transform these outputs into analytical dimensions and marts
-- consumed by the Power BI semantic model.
-- =====================================================================
-- Note:
--
-- Tables created in this script represent the NLP Output Layer.
--
-- Analytical models such as dim_review and dim_asinreview
-- are managed separately by dbt transformation models.

-- =====================================================================

-- 1. NLP processing control table
-- Tracks NLP processing status and model version information
-- for each review processed by the NLP pipeline.
--
-- Review_ID is the primary business key at the NLP layer.

CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.review_processing` (
    Review_ID STRING NOT NULL,          -- Unique review identifier
    Model_Version STRING,               -- NLP model version used for processing
    Processed_Time TIMESTAMP,           -- Timestamp when the review was processed
    Processing_Status STRING,           -- SUCCESS or FAILED
    Error_Message STRING                -- Error details if processing failed
);

-- 2. Review-level NLP feature table
-- Stores single-valued NLP features generated from review text.
-- Multi-valued classifications are stored separately
-- in bridge tables.

CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.review_features` (
    Review_ID STRING NOT NULL,          -- Unique review identifier
    Keywords STRING                     -- Semicolon-separated keywords extracted from the review
);

-- 3. NLP bridge tables
-- Bridge tables store multi-valued NLP classifications.
--
-- They preserve review-level relationships and prevent
-- Cartesian product issues in downstream analytical models.

CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.scene_bridge` (
    Review_ID STRING,                   -- Review identifier
    Main_Scene STRING,                  -- Main scene category
    Sub_Scene STRING                    -- Sub scene category
);

-- Friction bridge table
CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.friction_bridge` (
    Review_ID STRING,                   -- Review identifier
    Friction_Main STRING,               -- Main friction category
    Friction_Sub STRING,                -- Sub friction category
    Friction_Source STRING,             -- Classification source: Regex or Embedding
    Friction_Score FLOAT64,             -- Semantic similarity score (for embedding)
    Friction_Margin FLOAT64             -- Similarity margin (for embedding)
);

-- Motivation bridge table
CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.motivation_bridge` (
    Review_ID STRING,                   -- Review identifier
    Motivation_Main STRING,             -- Main motivation category
    Motivation_Sub STRING,              -- Sub motivation category
    Motivation_Source STRING,           -- Classification source: Regex or Embedding
    Motivation_Score FLOAT64,           -- Semantic similarity score (for embedding)
    Motivation_Margin FLOAT64           -- Similarity margin (for embedding)
);

-- Time bridge table
CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.time_bridge` (
    Review_ID STRING,                   -- Review identifier
    Time_of_Day STRING                  -- Time-of-day category extracted from the review
);

-- Location bridge table
CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.location_bridge` (
    Review_ID STRING,                   -- Review identifier
    Location STRING                     -- Location category extracted from the review
);
