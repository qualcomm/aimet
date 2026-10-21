# Ingest Lambda

Source and build tooling for the ingest Lambda function — the pipeline step
that turns an uploaded export into database rows plus a long-term artifact
copy.

- **Trigger:** `s3:ObjectCreated:*` on the staging bucket (a `profiling_data.csv`
  upload under `<run_id>/<variant>/`)
- **Runtime:** python3.14, handler `lambda_ingest.handler`
- **What it does:** moves artifact zips from staging to the long-term artifacts
  bucket (date-partitioned, multipart for >5 GB), rewrites `export` URLs in the
  CSV, inserts rows into the Postgres analytics DB, then cleans up staging.
- **Config:** env vars `STAGING_BUCKET`, `ARTIFACTS_BUCKET`, `DB_HOST`, `DB_NAME`,
  `DB_USER` (set on the function). The DB password is read from Secrets Manager
  at runtime.

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
