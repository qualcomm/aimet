# Ingest Lambda

Source and build tooling for the ingest Lambda function — the pipeline step
that turns an uploaded export into database rows plus a long-term artifact
copy.

- **Trigger:** `s3:ObjectCreated:*` on the staging bucket (a `profiling_data.csv`
  upload under `<run_id>/<variant>/`)
- **Runtime:** python3.14, handler `lambda_ingest.handler`
- **What it does:** moves each model's two zips from staging, date-partitioned
  under `exports/` (multipart for >5 GB):
  - `<export>.zip` (model files, `config.yaml`, `profiling_data.*`,
    `analysis/`) to the artifacts bucket
  - `<export>_results.zip` (the same without the model files) to the results
    bucket

  It writes their URLs into the `export` and `report` columns, inserts rows
  into the Postgres analytics DB, then cleans up staging.
- **Config:** env vars `STAGING_BUCKET`, `ARTIFACTS_BUCKET`
  (`genai-laboratory-artifacts`), `RESULTS_BUCKET`, `DB_HOST`, `DB_NAME`,
  `DB_USER` (set on the function). The DB password is read from Secrets Manager
  at runtime.
- **Buckets:** both long-term buckets need public read (the dashboard links are
  plain https URLs), and the function's role needs `s3:PutObject` and
  `s3:AbortMultipartUpload` on each.
- **Schema:** the `genai_laboratory` table needs `analysis jsonb` and
  `report text` columns
  (`ALTER TABLE genai_laboratory ADD COLUMN IF NOT EXISTS analysis jsonb,
  ADD COLUMN IF NOT EXISTS report text;`).

## Build

```bash
bash build.sh          # produces build/function.zip
```

`build.sh` installs `requirements.txt` for the Lambda platform
(`manylinux2014_x86_64`) and zips it with the handler.

## Deploy

Upload `build/function.zip` via the **Lambda web console**:

> Lambda → *(the ingest function)* → **Code** → **Upload from** → **.zip file**

No credentials or local env config are needed to build.

## Files

| File | Purpose |
|------|---------|
| `lambda_ingest.py` | function source |
| `requirements.txt` | runtime deps (`pg8000`, `boto3`) |
| `build.sh` | build `build/function.zip` |
| `build/` | generated bundle output — gitignored |
