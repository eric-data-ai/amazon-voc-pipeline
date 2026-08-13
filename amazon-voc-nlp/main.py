import os
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timezone
from classifier import classify_review, empty_feature, encode_reviews

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "amazon-voc-pipeline")
DATASET = "voc_features"
MODEL_VERSION = "nlp_v2.0_rule_v1"


def run_nlp():
    bq_client = bigquery.Client(project=PROJECT_ID)

    # 查询未处理的评论（按 Review_ID 去重，移除 Review_Key）
    query = f"""
        SELECT
            r.Review_ID,
            ANY_VALUE(r.Clean_Text) AS Clean_Text,
            ANY_VALUE(r.Rating) AS Rating
        FROM `{PROJECT_ID}.voc_raw.review_raw` r
        WHERE NOT EXISTS (
            SELECT 1
            FROM `{PROJECT_ID}.{DATASET}.review_processing` p
            WHERE p.Review_ID = r.Review_ID
              AND p.Processing_Status = 'SUCCESS'
              AND p.Model_Version = '{MODEL_VERSION}'
        )
        AND r.Clean_Text IS NOT NULL
        GROUP BY r.Review_ID
    """
    df = bq_client.query(query).to_dataframe()
    if df.empty:
        print("No new reviews to process")
        return

    run_time = datetime.now(timezone.utc)
    print(f"Processing {len(df)} reviews...")

    # 批量编码
    print("Generating embeddings in batch...")
    texts = [
        str(t).strip() if pd.notna(t) and str(t).strip() else " "
        for t in df["Clean_Text"]
    ]
    try:
        all_embeddings = encode_reviews(texts, batch_size=64)
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        raise

    processing_rows = []
    feature_rows = []
    scene_rows = []
    friction_rows = []
    motivation_rows = []
    time_rows = []
    location_rows = []
    processed_rids = set()

    for idx, row in enumerate(df.itertuples()):
        rating = int(row.Rating) if pd.notna(row.Rating) else None
        rid = row.Review_ID
        processed_rids.add(rid)

        # 状态记录（纯控制字段，无 Review_Key）
        status_record = {
            "Review_ID": rid,
            "Model_Version": MODEL_VERSION,
            "Processed_Time": run_time,
            "Processing_Status": "SUCCESS",
            "Error_Message": None,
        }

        try:
            emb = all_embeddings[idx]
            result = classify_review(row.Clean_Text, rating, emb=emb)
            status = "SUCCESS"
            error = None
        except Exception as e:
            result = empty_feature()
            status = "FAILED"
            error = str(e)

        status_record["Processing_Status"] = status
        status_record["Error_Message"] = error
        processing_rows.append(status_record)

        # NLP 特征表
        feature_rows.append(
            {
                "Review_ID": rid,
                "Keywords": result.get("Keywords", ""),
            }
        )

        # 成功时生成桥表行
        if status == "SUCCESS":
            for scene in result.get("Scenes", []):
                scene_rows.append(
                    {
                        "Review_ID": rid,
                        "Main_Scene": scene["Main"],
                        "Sub_Scene": scene["Sub"],
                    }
                )
            for friction in result.get("Frictions", []):
                friction_rows.append(
                    {
                        "Review_ID": rid,
                        "Friction_Main": friction["Main"],
                        "Friction_Sub": friction["Sub"],
                        "Friction_Source": friction["Source"],
                        "Friction_Score": friction["Score"],
                        "Friction_Margin": friction["Margin"],
                    }
                )
            for motivation in result.get("Motivations", []):
                motivation_rows.append(
                    {
                        "Review_ID": rid,
                        "Motivation_Main": motivation["Main"],
                        "Motivation_Sub": motivation["Sub"],
                        "Motivation_Source": motivation["Source"],
                        "Motivation_Score": motivation["Score"],
                        "Motivation_Margin": motivation["Margin"],
                    }
                )
            for time_slot in result.get("Times", []):
                time_rows.append(
                    {
                        "Review_ID": rid,
                        "Time_of_Day": time_slot,
                    }
                )
            for location in result.get("Locations", []):
                location_rows.append(
                    {
                        "Review_ID": rid,
                        "Location": location,
                    }
                )

    # ========== 写入 BigQuery ==========
    # 1. 控制状态表
    df_processing = pd.DataFrame(processing_rows)
    _merge_processing_status(
        bq_client, df_processing, f"{PROJECT_ID}.{DATASET}.review_processing"
    )

    # 2. NLP 特征表
    df_features = pd.DataFrame(feature_rows)
    _merge_features(bq_client, df_features, f"{PROJECT_ID}.{DATASET}.review_features")

    # 3. 桥表：DELETE + INSERT
    rids = list(processed_rids)
    if rids:
        _delete_old_bridge_rows(
            bq_client, f"{PROJECT_ID}.{DATASET}.scene_bridge", "Review_ID", rids
        )
        _delete_old_bridge_rows(
            bq_client, f"{PROJECT_ID}.{DATASET}.friction_bridge", "Review_ID", rids
        )
        _delete_old_bridge_rows(
            bq_client, f"{PROJECT_ID}.{DATASET}.motivation_bridge", "Review_ID", rids
        )
        _delete_old_bridge_rows(
            bq_client, f"{PROJECT_ID}.{DATASET}.time_bridge", "Review_ID", rids
        )
        _delete_old_bridge_rows(
            bq_client, f"{PROJECT_ID}.{DATASET}.location_bridge", "Review_ID", rids
        )

    if scene_rows:
        _insert_bridge_rows(
            bq_client, pd.DataFrame(scene_rows), f"{PROJECT_ID}.{DATASET}.scene_bridge"
        )
    if friction_rows:
        _insert_bridge_rows(
            bq_client,
            pd.DataFrame(friction_rows),
            f"{PROJECT_ID}.{DATASET}.friction_bridge",
        )
    if motivation_rows:
        _insert_bridge_rows(
            bq_client,
            pd.DataFrame(motivation_rows),
            f"{PROJECT_ID}.{DATASET}.motivation_bridge",
        )
    if time_rows:
        _insert_bridge_rows(
            bq_client, pd.DataFrame(time_rows), f"{PROJECT_ID}.{DATASET}.time_bridge"
        )
    if location_rows:
        _insert_bridge_rows(
            bq_client,
            pd.DataFrame(location_rows),
            f"{PROJECT_ID}.{DATASET}.location_bridge",
        )

    success = sum(1 for r in processing_rows if r["Processing_Status"] == "SUCCESS")
    failed = sum(1 for r in processing_rows if r["Processing_Status"] == "FAILED")
    print(
        f"Processed {len(processing_rows)} reviews. Success: {success}, Failed: {failed}"
    )
    print(
        f"Bridge rows - Scene: {len(scene_rows)}, Friction: {len(friction_rows)}, "
        f"Motivation: {len(motivation_rows)}, Time: {len(time_rows)}, Location: {len(location_rows)}"
    )


def _merge_processing_status(client, df, table_id):
    """MERGE 处理状态表（Review_ID 唯一）"""
    temp_table = f"{table_id}_temp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    schema = [
        bigquery.SchemaField("Review_ID", "STRING"),
        bigquery.SchemaField("Model_Version", "STRING"),
        bigquery.SchemaField("Processed_Time", "TIMESTAMP"),
        bigquery.SchemaField("Processing_Status", "STRING"),
        bigquery.SchemaField("Error_Message", "STRING"),
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE", schema=schema
    )
    client.load_table_from_dataframe(df, temp_table, job_config=job_config).result()

    merge_sql = f"""
    MERGE `{table_id}` T
    USING `{temp_table}` S
    ON T.Review_ID = S.Review_ID
    WHEN MATCHED THEN UPDATE SET
        Model_Version = S.Model_Version,
        Processed_Time = S.Processed_Time,
        Processing_Status = S.Processing_Status,
        Error_Message = S.Error_Message
    WHEN NOT MATCHED THEN INSERT ROW
    """
    client.query(merge_sql).result()
    client.delete_table(temp_table)


def _merge_features(client, df, table_id):
    """MERGE 特征表（Review_ID 唯一）"""
    temp_table = f"{table_id}_temp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    schema = [
        bigquery.SchemaField("Review_ID", "STRING"),
        bigquery.SchemaField("Keywords", "STRING"),
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE", schema=schema
    )
    client.load_table_from_dataframe(df, temp_table, job_config=job_config).result()

    merge_sql = f"""
    MERGE `{table_id}` T
    USING `{temp_table}` S
    ON T.Review_ID = S.Review_ID
    WHEN MATCHED THEN UPDATE SET
        Keywords = S.Keywords
    WHEN NOT MATCHED THEN INSERT ROW
    """
    client.query(merge_sql).result()
    client.delete_table(temp_table)


def _delete_old_bridge_rows(client, table_id, key_col, keys):
    batch_size = 500
    for i in range(0, len(keys), batch_size):
        batch = keys[i : i + batch_size]
        keys_str = ",".join([f"'{k}'" for k in batch])
        client.query(
            f"DELETE FROM `{table_id}` WHERE {key_col} IN ({keys_str})"
        ).result()


def _insert_bridge_rows(client, df, table_id):
    table = client.get_table(table_id)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=table.schema,
    )
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()


if __name__ == "__main__":
    run_nlp()
