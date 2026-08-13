"""
VOC ETL Job (增加 Review_Key)
重新运行以更新 review_raw 表，添加 Review_Key 字段。
"""

import os
import re
import pandas as pd
from io import BytesIO
from datetime import datetime, timezone
from google.cloud import storage, bigquery

# =========================
# 配置
# =========================
BUCKET_NAME = "amazon-voc-raw"
PREFIX = "reviews/"
PROJECT_ID = "amazon-voc-pipeline"
DATASET = "voc_raw"
TABLE = "review_raw"

# =========================
# BigQuery 表结构 (新增 Review_Key)
# =========================
BQ_SCHEMA = [
    bigquery.SchemaField("Review_Key", "STRING"),   # 新增唯一标识
    bigquery.SchemaField("Review_ID", "STRING"),
    bigquery.SchemaField("ASIN", "STRING"),
    bigquery.SchemaField("Rating", "INTEGER"),
    bigquery.SchemaField("Title", "STRING"),
    bigquery.SchemaField("Content", "STRING"),
    bigquery.SchemaField("Clean_Text", "STRING"),
    bigquery.SchemaField("URL", "STRING"),
    bigquery.SchemaField("Review_Date", "DATE"),
    bigquery.SchemaField("Verified", "STRING"),
    bigquery.SchemaField("Helpful_Votes", "INTEGER"),
    bigquery.SchemaField("Variant", "STRING"),
    bigquery.SchemaField("Vine", "STRING"),
    bigquery.SchemaField("Images", "STRING"),
    bigquery.SchemaField("Videos", "STRING"),
    bigquery.SchemaField("Source_File", "STRING"),
    bigquery.SchemaField("ETL_Time", "TIMESTAMP"),
]

# =========================
# 工具函数 (不变)
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]*>", " ", text)
    html_entities = {
        "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&apos;": "'",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    text = (text.replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u00b4", "'").replace("`", "'"))
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_review_id(url):
    if not isinstance(url, str):
        return ""
    match = re.search(r"/customer-reviews/([^/]+)/ref", url)
    return match.group(1) if match else ""

def extract_asin_from_path(path):
    fname = os.path.basename(path)
    return fname.split("-")[0]

# =========================
# 处理单个文件
# =========================
def process_file(blob):
    content = blob.download_as_bytes()
    df = pd.read_excel(BytesIO(content), engine="openpyxl")

    required = ["Title", "Content", "URL", "Rating"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {blob.name}: {missing}")

    df["ASIN"] = extract_asin_from_path(blob.name)
    df["Review_ID"] = df["URL"].apply(extract_review_id)
    # 新增：生成唯一 Review_Key
    df["Review_Key"] = df["ASIN"].astype(str) + "_" + df["Review_ID"].astype(str)
    df["Source_File"] = blob.name
    df["ETL_Time"] = datetime.now(timezone.utc)

    df["Clean_Text"] = df["Title"].fillna("") + " " + df["Content"].fillna("")
    df["Clean_Text"] = df["Clean_Text"].apply(clean_text)

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").astype("Int64")

    if "Date" in df.columns:
        df["Review_Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    else:
        df["Review_Date"] = None

    if "Verified Purchase" in df.columns:
        df["Verified"] = df["Verified Purchase"]
    else:
        df["Verified"] = None

    if "Helpful" in df.columns:
        df["Helpful_Votes"] = pd.to_numeric(df["Helpful"], errors="coerce").fillna(0).astype("Int64")
    else:
        df["Helpful_Votes"] = 0

    if "Vine Voice" in df.columns:
        df["Vine"] = df["Vine Voice"]
    else:
        df["Vine"] = None

    for col in ["Variant", "Images", "Videos"]:
        if col not in df.columns:
            df[col] = None

    keep_cols = [
        "Review_Key",    # 新增
        "Review_ID", "ASIN", "Rating", "Title", "Content",
        "Clean_Text", "URL", "Review_Date", "Verified",
        "Helpful_Votes", "Variant", "Vine", "Images", "Videos",
        "Source_File", "ETL_Time"
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    return df[keep_cols]

# =========================
# 主入口
# =========================
def run_etl():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=PREFIX))
    excel_blobs = [b for b in blobs if b.name.endswith(".xlsx")]

    if not excel_blobs:
        print("No Excel files found.")
        return

    print(f"Found {len(excel_blobs)} files. Starting processing...")
    all_dfs = []
    for i, blob in enumerate(excel_blobs, 1):
        print(f"[{i}/{len(excel_blobs)}] Processing {blob.name}")
        try:
            df = process_file(blob)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {blob.name}: {e}")

    if not all_dfs:
        print("No data extracted.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows: {len(final_df)}")

    # 统一 STRING 字段类型
    string_columns = [
        "Review_Key", "Review_ID", "ASIN", "Title", "Content",
        "Clean_Text", "URL", "Verified", "Variant", "Vine",
        "Images", "Videos", "Source_File"
    ]
    for col in string_columns:
        final_df[col] = final_df[col].astype("string")

    print("Data types after standardization:\n", final_df.dtypes)
    print("Missing values:\n", final_df.isnull().sum())
    print("Sample data:\n", final_df.head(3))

    bq_client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    job_config = bigquery.LoadJobConfig(
        schema=BQ_SCHEMA,
        write_disposition="WRITE_TRUNCATE",  # 覆盖旧表
    )
    job = bq_client.load_table_from_dataframe(final_df, table_id, job_config=job_config)
    job.result()
    print(f"Successfully loaded {len(final_df)} rows into {table_id}")

if __name__ == "__main__":
    run_etl()