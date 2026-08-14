-- =====================================================================
-- Amazon VOC Pipeline - BigQuery Feature Layer Schema
-- =====================================================================
-- This script creates the analytical feature tables used by the NLP
-- pipeline and the Power BI semantic model.
-- =====================================================================

-- 1. Processing control table
-- Tracks the NLP processing status for each unique review.
-- Note: Review_Key has been removed from this table; Review_ID is the
-- primary business key at the NLP processing layer.
CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.review_processing` (
    Review_ID STRING NOT NULL,          -- Unique review identifier
    Model_Version STRING,               -- NLP model version used for processing
    Processed_Time TIMESTAMP,           -- Timestamp when the review was processed
    Processing_Status STRING,           -- SUCCESS or FAILED
    Error_Message STRING                -- Error details if processing failed
);

-- 2. NLP feature table
-- Stores review-level NLP features that are not multi-valued.
-- Currently contains Keywords used for word cloud analysis.
CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.review_features` (
    Review_ID STRING NOT NULL,          -- Unique review identifier
    Keywords STRING                     -- Semicolon-separated keywords extracted from the review
);

-- 3. Bridge tables
-- Bridge tables handle multi-valued NLP dimensions to preserve
-- relationships and avoid Cartesian products.

-- Scene bridge table
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

-- 4. Dimension tables
-- Dimension tables are generated via SQL and provide review-centric
-- analytical structures for the Power BI semantic model.

-- Review dimension table (one row per review)
CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.dim_review` AS
SELECT
    r.Review_ID,
    ANY_VALUE(r.Clean_Text) AS Clean_Text,   -- Cleaned review text
    ANY_VALUE(r.Rating) AS Rating,           -- Review rating
    ANY_VALUE(r.Helpful_Votes) AS Helpful,   -- Number of helpful votes
    ANY_VALUE(r.Review_Date) AS Review_Date, -- Review date
    ANY_VALUE(f.Keywords) AS Keywords        -- Keywords from NLP feature table
FROM `amazon-voc-pipeline.voc_raw.review_raw` r
LEFT JOIN `amazon-voc-pipeline.voc_features.review_features` f
    ON r.Review_ID = f.Review_ID
GROUP BY r.Review_ID;

-- ASIN-review relationship table
-- Represents the association between products and reviews.
-- Designed to support many-to-many relationships in the semantic model.
CREATE OR REPLACE TABLE `amazon-voc-pipeline.voc_features.dim_asinreview` AS
SELECT DISTINCT
    ASIN,
    Review_ID,
    Review_Key
FROM `amazon-voc-pipeline.voc_raw.review_raw`;