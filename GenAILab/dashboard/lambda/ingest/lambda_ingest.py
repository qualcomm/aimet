# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Lambda function to ingest GenAI scorecard results from a staging S3 bucket.

Triggered by a CSV upload to the staging bucket. This function:
1. Moves each model's export zip from staging to the artifacts bucket, and its
   results zip (config, profiling data, analysis-pass report; no model data)
   to the results bucket, both under exports/
2. Reads the CSV, rewrites export_location to the public S3 URL of each export
   zip and sets report to the URL of its results zip
3. Inserts the rows into the RDS PostgreSQL database

S3 layout (uploaded by GitHub Actions):
    <run_id>/<variant>/profiling_data.csv
    <run_id>/<variant>/run_metadata.json
    <run_id>/<variant>/<export_dir>.zip
    <run_id>/<variant>/<export_dir>_results.zip

Environment variables:
    STAGING_BUCKET:   S3 bucket where GitHub Actions uploads results
    ARTIFACTS_BUCKET: Long-term S3 bucket for the export zips
    RESULTS_BUCKET:   Long-term S3 bucket for the results zips (no model data)
    DB_HOST:          RDS endpoint
    DB_NAME:          Database name
    DB_USER:          Database user
    DB_PASSWORD:      Database password (use Secrets Manager in production)
"""

import csv
import sys
import io
import json
import os
import re
import urllib.parse

import logging

import boto3
import pg8000.native

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
sm = boto3.client("secretsmanager", region_name="us-west-1")

STAGING_BUCKET = os.environ["STAGING_BUCKET"]
ARTIFACTS_BUCKET = os.environ["ARTIFACTS_BUCKET"]
RESULTS_BUCKET = os.environ["RESULTS_BUCKET"]
DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_SECRET_ARN = os.environ["DB_SECRET_ARN"]

_secret_resp = sm.get_secret_value(SecretId=DB_SECRET_ARN)
DB_PASSWORD = json.loads(_secret_resp["SecretString"])["password"]


def handler(event, context):
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    csv_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
    logger.info("Triggered by %s/%s", bucket, csv_key)

    # csv_key looks like: <run_id>/<variant>/profiling_data.csv
    parts = csv_key.split("/")
    run_id = parts[0]
    variant = parts[1]
    variant_prefix = f"{run_id}/{variant}/"

    # 1. Read run metadata if available
    run_metadata = _read_run_metadata(bucket, f"{variant_prefix}run_metadata.json")
    logger.info("Run metadata: %s", run_metadata)

    # 2. Move artifact zips from staging to the artifacts bucket
    export_url_map = _move_artifacts(variant_prefix, run_id)
    logger.info(
        "Moved %d artifacts: %s", len(export_url_map), list(export_url_map.keys())
    )

    # 3. Move results zips from staging to the results bucket
    results_url_map = _move_results(variant_prefix, run_id)
    logger.info(
        "Moved %d results: %s", len(results_url_map), list(results_url_map.keys())
    )

    # 4. Read CSV, rewrite export/report URLs, insert into DB
    csv_body = s3.get_object(Bucket=bucket, Key=csv_key)["Body"].read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(csv_body)))
    logger.info("Read %d rows from CSV", len(rows))

    for row in rows:
        row["report"] = ""
        local_export = row.get("export", "")
        if local_export:
            # Match local path to the zip filenames
            # Local export looks like: genai_output/exports/Llama-3.2-1B-Instruct_2026-03-04_143022
            dir_name = local_export.rstrip("/").split("/")[-1]
            row["export"] = export_url_map.get(f"{dir_name}.zip", "")
            row["report"] = results_url_map.get(f"{dir_name}{_RESULTS_SUFFIX}", "")

        # Merge run metadata into the environment column
        if run_metadata and row.get("environment"):
            env = json.loads(_clean_json_field(row["environment"]))
            env["run_type"] = "online"
            env.update(run_metadata)
            row["environment"] = json.dumps(env)

    logger.info("Inserting %d rows into database", len(rows))
    _insert_rows(rows)
    logger.info("Insert complete")

    # 5. Clean up staging
    _delete_staging_objects(run_id, variant)
    logger.info("Staging cleanup complete for %s/%s", run_id, variant)

    return {
        "statusCode": 200,
        "body": f"Ingested {len(rows)} rows from run {run_id}/{variant}",
    }


def _read_run_metadata(bucket, key):
    """Read run_metadata.json from S3. Returns dict or empty dict if not found."""
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        return json.load(io.StringIO(body))
    except s3.exceptions.NoSuchKey:
        return {}


# Suffix CI gives the zip of an export's results (no model data)
_RESULTS_SUFFIX = "_results.zip"


def _date_partitioned_key(filename, run_id, suffix=".zip"):
    """Convert a flat zip filename into a date-partitioned S3 key.

    ``suffix`` is the part after the export dir's timestamp.

    Example: Llama-3.2-1B-Instruct_2026-03-04_143022.zip
          -> exports/2026/03/04/12345678/Llama-3.2-1B-Instruct_143022.zip
    """
    # Matches filenames like: ModelName_2026-03-04_143022<suffix>
    date_pattern = re.compile(
        rf"_(\d{{4}})-(\d{{2}})-(\d{{2}})_\d{{6}}{re.escape(suffix)}$"
    )
    match = date_pattern.search(filename)
    if not match:
        return f"exports/{run_id}/{filename}"
    year, month, day = match.groups()
    # Strip the date from the filename, keep the time
    short_name = filename.replace(f"_{year}-{month}-{day}", "")
    return f"exports/{year}/{month}/{day}/{run_id}/{short_name}"


MULTIPART_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5 GB
MULTIPART_CHUNK = 500 * 1024 * 1024  # 500 MB per part


def _copy_s3_object(source_bucket, source_key, dest_bucket, dest_key, size):
    """Copy an S3 object, using multipart copy for files over 5 GB."""
    copy_source = {"Bucket": source_bucket, "Key": source_key}

    if size < MULTIPART_THRESHOLD:
        s3.copy_object(Bucket=dest_bucket, Key=dest_key, CopySource=copy_source)
        return

    mpu = s3.create_multipart_upload(Bucket=dest_bucket, Key=dest_key)
    upload_id = mpu["UploadId"]
    try:
        parts = []
        part_number = 1
        offset = 0
        while offset < size:
            end = min(offset + MULTIPART_CHUNK, size) - 1
            resp = s3.upload_part_copy(
                Bucket=dest_bucket,
                Key=dest_key,
                CopySource=copy_source,
                CopySourceRange=f"bytes={offset}-{end}",
                PartNumber=part_number,
                UploadId=upload_id,
            )
            parts.append(
                {"ETag": resp["CopyPartResult"]["ETag"], "PartNumber": part_number}
            )
            offset = end + 1
            part_number += 1

        s3.complete_multipart_upload(
            Bucket=dest_bucket,
            Key=dest_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except Exception:
        s3.abort_multipart_upload(Bucket=dest_bucket, Key=dest_key, UploadId=upload_id)
        raise


def _move_artifacts(variant_prefix, run_id):
    """Move artifact zips from staging to long-term bucket. Returns {filename: public_url}.

    Copies are performed in parallel using a thread pool so that multiple
    large zips don't sequentially exhaust the Lambda timeout.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    url_map = {}
    copy_jobs = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=STAGING_BUCKET, Prefix=variant_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".zip") or key.endswith(_RESULTS_SUFFIX):
                continue
            filename = key.split("/")[-1]
            dest_key = _date_partitioned_key(filename, run_id)
            url_map[filename] = (
                f"https://{ARTIFACTS_BUCKET}.s3.amazonaws.com/{dest_key}"
            )
            copy_jobs.append((key, dest_key, obj["Size"]))

    def _do_copy(args):
        src_key, dst_key, size = args
        _copy_s3_object(STAGING_BUCKET, src_key, ARTIFACTS_BUCKET, dst_key, size)

    with ThreadPoolExecutor(max_workers=len(copy_jobs) or 1) as pool:
        futures = [pool.submit(_do_copy, job) for job in copy_jobs]
        for f in as_completed(futures):
            f.result()  # raises if any copy failed

    return url_map


def _move_results(variant_prefix, run_id):
    """Move results zips from staging to the results bucket.

    Returns {filename: public_url}.

    A results zip holds one export's config, profiling data and analysis-pass
    report, without its model data. It lands at the same key as its export zip
    in the artifacts bucket, plus the _results suffix.

    Example: Llama-3.2-1B-Instruct_2026-03-04_143022_results.zip
          -> exports/2026/03/04/12345678/Llama-3.2-1B-Instruct_143022_results.zip
    """
    url_map = {}

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=STAGING_BUCKET, Prefix=variant_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(_RESULTS_SUFFIX):
                continue
            filename = key.split("/")[-1]
            dest_key = _date_partitioned_key(filename, run_id, suffix=_RESULTS_SUFFIX)
            _copy_s3_object(STAGING_BUCKET, key, RESULTS_BUCKET, dest_key, obj["Size"])
            url_map[filename] = f"https://{RESULTS_BUCKET}.s3.amazonaws.com/{dest_key}"

    return url_map


def _clean_json_field(value):
    """Undo the double-quote escaping produced by dict_to_postgres_csv_json_field."""
    if not value:
        return "{}"
    return value.strip().strip('"').replace('""', '"')


def _insert_rows(rows):
    """Insert CSV rows into the RDS database."""
    conn = pg8000.native.Connection(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    try:
        for row in rows:
            run_group = row.get("run_group", "").strip() or None
            # Sparse: only runs with an analysis section carry one.
            analysis = row.get("analysis", "").strip() or None
            conn.run(
                "INSERT INTO genai_laboratory"
                "    (model_type, model_id, model_modifiers, precision,"
                "     components, accuracy_results, export, environment, run_group,"
                "     analysis, report)"
                " VALUES (:model_type, :model_id, :model_modifiers, :precision,"
                "         :components, :accuracy_results, :export, :environment,"
                "         :run_group, :analysis, :report)",
                model_type=row["model_type"],
                model_id=row["model_id"],
                model_modifiers=_clean_json_field(row["model_modifiers"]),
                precision=_clean_json_field(row.get("precision", "{}")),
                components=_clean_json_field(row["components"]),
                accuracy_results=_clean_json_field(row["accuracy_results"]),
                export=row.get("export", ""),
                environment=_clean_json_field(row.get("environment", "{}")),
                run_group=run_group,
                analysis=analysis,
                report=row.get("report", ""),
            )
        conn.run("COMMIT")
    finally:
        conn.close()


def _delete_staging_objects(run_id, variant):
    """Delete all objects under the run_id/variant prefix in the staging bucket."""
    prefix = f"{run_id}/{variant}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=STAGING_BUCKET, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=STAGING_BUCKET, Delete={"Objects": objects})
