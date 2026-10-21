-- Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
-- SPDX-License-Identifier: BSD-3-Clause

WITH tags AS (
  SELECT
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
    ) AS precision_tag
  FROM genai_laboratory
  WHERE model_id = {{model_id}}
    AND environment->>'variant' = 'onnx'
    AND (accuracy_results->'MMLU'->>'result') IS NOT NULL
    AND (accuracy_results->'PPL'->>'result') IS NOT NULL
    AND (accuracy_results->'AutogradedPrompts'->>'result') IS NOT NULL
)
SELECT DISTINCT precision_tag
FROM tags
ORDER BY precision_tag
