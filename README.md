# Amazon VOC Pipeline

A cloud-based Amazon review analytics pipeline that ingests review data into Google BigQuery, applies hybrid rule-based and semantic NLP classification, and prepares normalized analytical data for Power BI.

## Overview

This project is an end-to-end Voice of Customer (VOC) analytics pipeline for Amazon product reviews.

The pipeline combines:

- Cloud-based data ingestion
- Data cleaning and normalization
- Hybrid NLP classification
- Semantic embeddings
- Rule-based pattern matching
- BigQuery data modeling
- Incremental NLP processing
- Bridge-table modeling for multi-label features
- Power BI analytics

The goal is to transform raw Amazon review data into structured VOC features that can be explored and analyzed through BI dashboards.

---

## Architecture


Amazon Review Data
        │
        ▼
Google Cloud Storage
        │
        ▼
┌─────────────────────┐
│   Cloud Run ETL Job │
└──────────┬──────────┘
           │
           ▼
BigQuery: voc_raw.review_raw
           │
           ▼
┌─────────────────────┐
│   Cloud Run NLP Job │
└──────────┬──────────┘
           │
           ▼
   Hybrid NLP Engine
    ┌──────┴──────┐
    │             │
  Regex      Embeddings
    │             │
    └──────┬──────┘
           ▼
      NLP Features
           │
     ┌─────┼───────────────┐
     │     │               │
     ▼     ▼               ▼
Processing Features    Bridge Tables
 Status
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
            Scene     Friction   Motivation
              │
              ├── Time
              └── Location
                         │
                         ▼
                      Power BI