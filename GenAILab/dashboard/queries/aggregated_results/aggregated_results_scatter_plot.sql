-- Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
-- SPDX-License-Identifier: BSD-3-Clause

WITH tagged AS (
  SELECT *,
    -- Full precision tag for filtering (includes visual for VLMs)
    CONCAT(
      CASE
        WHEN components->'backbone'->>'recipe_name' = 'RemoveQuantization'
          OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
              FROM jsonb_array_elements(COALESCE(components->'backbone'->'steps', '[]'::jsonb)) AS step)
        THEN 'FP (baseline)'
        WHEN precision IS NULL OR precision = '{}'::jsonb
          OR precision->'blocks' IS NULL THEN '—'
        ELSE CONCAT(
          (SELECT string_agg(
            CONCAT(
              'W=', value->>'qtype',
              CASE
                WHEN COALESCE(value->>'granularity', 'PCQ') <> 'PCQ'
                THEN CONCAT('/', value->>'granularity',
                            '(', COALESCE(value->>'block_size', '?'), ')')
                ELSE ''
              END,
              CASE
                WHEN key <> 'default'
                THEN CONCAT('[', key, ']')
                ELSE ''
              END
            ),
            '  ' ORDER BY key
          ) FROM jsonb_each(precision->'blocks')),
          '  A=', COALESCE(precision->>'activations', '?'),
          '  KV=', COALESCE(precision->>'kv_cache', 'int8'),
          '  LM=', COALESCE(precision->'lm_head'->>'qtype', '?'),
          CASE
            WHEN COALESCE(precision->'lm_head'->>'granularity', 'PCQ') <> 'PCQ'
            THEN CONCAT('/', precision->'lm_head'->>'granularity',
                        '(', COALESCE(precision->'lm_head'->>'block_size', '?'), ')')
            ELSE ''
          END
        )
      END,
      CASE
        WHEN precision->'visual' IS NULL THEN ''
        WHEN components->'visual'->>'recipe_name' = 'RemoveQuantization'
          OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
              FROM jsonb_array_elements(COALESCE(components->'visual'->'steps', '[]'::jsonb)) AS step)
        THEN '  |  Visual: FP (baseline)'
        WHEN components->'backbone'->>'recipe_name' = 'RemoveQuantization'
          OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
              FROM jsonb_array_elements(COALESCE(components->'backbone'->'steps', '[]'::jsonb)) AS step)
        THEN '  |  Visual: FP (baseline)'
        ELSE CONCAT(
          '  |  Visual: W=', COALESCE(precision->'visual'->'weight'->>'qtype', '?'),
          CASE
            WHEN COALESCE(precision->'visual'->'weight'->>'granularity', 'PCQ') <> 'PCQ'
            THEN CONCAT('/', precision->'visual'->'weight'->>'granularity',
                        '(', COALESCE(precision->'visual'->'weight'->>'block_size', '?'), ')')
            ELSE ''
          END,
          '  A=', COALESCE(precision->'visual'->>'activations', '?')
        )
      END
    ) AS precision_tag,

    -- Recipe label for deduplication
    CASE
      WHEN components->'backbone'->'steps' IS NOT NULL THEN
        (SELECT string_agg(
          CONCAT(
            step->>'recipe_name',
            CASE WHEN step->>'blocks' IS NOT NULL
                 THEN CONCAT('[', step->>'blocks', ']') ELSE '' END,
            '(',
            CASE
              WHEN step->>'dataset_name' = 'Interleaved'
                AND step->'dataset_kwargs'->'source_datasets' IS NOT NULL
              THEN (SELECT string_agg(
                      CONCAT(sd->>'name', COALESCE(':' || NULLIF(sd->>'split', ''), '')),
                      '+' ORDER BY ord2)
                    FROM jsonb_array_elements(step->'dataset_kwargs'->'source_datasets')
                         WITH ORDINALITY AS s2(sd, ord2))
              ELSE CONCAT(
                COALESCE(step->>'dataset_name', ''),
                COALESCE(':' || NULLIF(step->'dataset_kwargs'->>'split', ''), '')
              )
            END,
            COALESCE(
              (SELECT ', ' || string_agg(CONCAT(kv.key, '=', kv.value #>> '{}'), ', ')
               FROM jsonb_each(step->'recipe_kwargs') AS kv
               WHERE step->'recipe_kwargs' IS NOT NULL
                     AND step->'recipe_kwargs' <> '{}'::jsonb),
              ''
            ),
            ')'
          ),
          ' → ' ORDER BY ord
        ) FROM jsonb_array_elements(components->'backbone'->'steps')
               WITH ORDINALITY AS s(step, ord))
      ELSE
        CONCAT(
          components->'backbone'->>'recipe_name', '(',
          COALESCE(components->'backbone'->>'dataset_name', ''),
          COALESCE(':' || NULLIF(components->'backbone'->'dataset_kwargs'->>'split', ''), ''),
          COALESCE(
            (SELECT ', ' || string_agg(CONCAT(kv.key, '=', kv.value #>> '{}'), ', ')
             FROM jsonb_each(components->'backbone'->'recipe_kwargs') AS kv
             WHERE components->'backbone'->'recipe_kwargs' IS NOT NULL
                   AND components->'backbone'->'recipe_kwargs' <> '{}'::jsonb),
            ''
          ),
          ')'
        )
    END AS recipe_label,

    -- Scoring version of the metric selected by {{rank_by}} (SCORING_VERSION)
    COALESCE(
      CASE
        WHEN {{rank_by}} = 'MMLU'    THEN (accuracy_results->'MMLU'->>'scoring_version')::int
        WHEN {{rank_by}} = 'MMMU'    THEN (accuracy_results->'MMMU'->>'scoring_version')::int
        WHEN {{rank_by}} = 'AutogradedPrompts' THEN (accuracy_results->'AutogradedPrompts'->>'scoring_version')::int
        WHEN {{rank_by}} = 'Grace'  THEN (accuracy_results->'Grace'->>'scoring_version')::int
        WHEN {{rank_by}} = 'AutogradedMultimodalPrompts' THEN (accuracy_results->'AutogradedMultimodalPrompts'->>'scoring_version')::int
        WHEN {{rank_by}} = 'PPL'     THEN (accuracy_results->'PPL'->>'scoring_version')::int
      END,
      1
    ) AS rank_scoring_version,

    -- Flags metrics computed under a non-default scoring version
    NULLIF(
      (SELECT string_agg(CONCAT(m.key, '=v', COALESCE((m.value->>'scoring_version')::int, 1)), ', ')
       FROM jsonb_each(accuracy_results) AS m
       WHERE COALESCE((m.value->>'scoring_version')::int, 1) <> 1),
      ''
    ) AS scoring_version_flags

  FROM genai_laboratory
  WHERE model_id = {{model_id}}
    AND environment->>'variant' = 'onnx'
    [[AND model_modifiers->'adaptations' ? {{adaptation}}]]
    [[AND environment->>'actor' = {{actor}}]]
    [[AND environment->>'branch' = {{branch}}]]
    [[AND (
      components->'backbone'->>'recipe_name' = {{recipe}}
      OR EXISTS (
        SELECT 1 FROM jsonb_array_elements(COALESCE(components->'backbone'->'steps', '[]'::jsonb)) AS step
        WHERE step->>'recipe_name' = {{recipe}}
      )
      OR components->'visual'->>'recipe_name' = {{recipe}}
      OR EXISTS (
        SELECT 1 FROM jsonb_array_elements(COALESCE(components->'visual'->'steps', '[]'::jsonb)) AS step
        WHERE step->>'recipe_name' = {{recipe}}
      )
    )]]
),
filtered AS (
  SELECT *
  FROM tagged
  WHERE precision_tag = {{precision_tag}}
    AND (accuracy_results->'MMLU'->>'result') IS NOT NULL
    AND (accuracy_results->'PPL'->>'result') IS NOT NULL
    AND (accuracy_results->'AutogradedPrompts'->>'result') IS NOT NULL
),
-- Never rank/select across scoring versions within a bucket
versioned AS (
  SELECT *,
    MAX(rank_scoring_version) OVER (
      PARTITION BY recipe_label
    ) AS bucket_current_scoring_version
  FROM filtered
),
current_version_only AS (
  SELECT * FROM versioned
  WHERE rank_scoring_version = bucket_current_scoring_version
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY recipe_label
      ORDER BY
        CASE
          WHEN {{rank_by}} = 'MMLU'    THEN (accuracy_results->'MMLU'->>'result')::numeric
          WHEN {{rank_by}} = 'MMMU'    THEN (accuracy_results->'MMMU'->>'result')::numeric
          WHEN {{rank_by}} = 'AutogradedPrompts' THEN (accuracy_results->'AutogradedPrompts'->>'result')::numeric
          WHEN {{rank_by}} = 'Grace'  THEN (accuracy_results->'Grace'->>'result')::numeric
          WHEN {{rank_by}} = 'AutogradedMultimodalPrompts' THEN (accuracy_results->'AutogradedMultimodalPrompts'->>'result')::numeric
          WHEN {{rank_by}} = 'PPL'     THEN -(accuracy_results->'PPL'->>'result')::numeric
        END DESC NULLS LAST,
        created_at DESC
    ) AS rn
  FROM current_version_only
)
SELECT
    recipe_label AS "Recipe",

    -- Y-axis: ranking metric
    CASE
      WHEN {{rank_by}} = 'MMLU'    THEN ROUND((accuracy_results->'MMLU'->>'result')::numeric, 4)
      WHEN {{rank_by}} = 'MMMU'    THEN ROUND((accuracy_results->'MMMU'->>'result')::numeric, 4)
      WHEN {{rank_by}} = 'AutogradedPrompts' THEN ROUND((accuracy_results->'AutogradedPrompts'->>'result')::numeric, 4)
      WHEN {{rank_by}} = 'Grace'  THEN ROUND((accuracy_results->'Grace'->>'result')::numeric, 4)
      WHEN {{rank_by}} = 'AutogradedMultimodalPrompts' THEN ROUND((accuracy_results->'AutogradedMultimodalPrompts'->>'result')::numeric, 4)
      WHEN {{rank_by}} = 'PPL'     THEN ROUND((accuracy_results->'PPL'->>'result')::numeric, 2)
    END AS "Metric",

    -- X-axis: runtime in minutes
    ROUND((components->'backbone'->'resource_utilization'->>'elapsed_ms')::numeric / 60000, 1) AS "Runtime (min)",

    -- Dot size: CUDA memory
    (components->'backbone'->'resource_utilization'->>'cuda_peak_mb')::numeric AS "GPU Peak (MB)",

    -- All accuracy metrics for hover tooltip
    ROUND((accuracy_results->'PPL'->>'result')::numeric, 2)      AS "PPL",
    ROUND((accuracy_results->'MMLU'->>'result')::numeric, 4)     AS "MMLU",
    ROUND((accuracy_results->'AutogradedPrompts'->>'result')::numeric, 4) AS "AutogradedPrompts",
    ROUND((accuracy_results->'Grace'->>'result')::numeric, 4) AS "Grace",
    ROUND((accuracy_results->'MMMU'->>'result')::numeric, 4)     AS "MMMU",
    ROUND((accuracy_results->'AutogradedMultimodalPrompts'->>'result')::numeric, 4) AS "AutogradedMultimodalPrompts",

    TO_CHAR(created_at, 'YYYY-MM-DD') AS "Date",
    environment->>'actor'             AS "actor",
    environment->>'branch'            AS "branch",
    scoring_version_flags             AS "Scoring Versions"

FROM ranked
WHERE rn = 1
ORDER BY "Runtime (min)" ASC
