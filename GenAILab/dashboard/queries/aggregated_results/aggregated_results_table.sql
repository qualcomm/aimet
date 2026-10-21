-- Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
-- SPDX-License-Identifier: BSD-3-Clause

WITH tagged AS (
  SELECT *,
    -- Categorize: 0=FP (no quant), 1=BQ, 2=LPBQ, 3=PCQ
    CASE
      WHEN components->'backbone'->>'recipe_name' = 'RemoveQuantization'
        OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
            FROM jsonb_array_elements(COALESCE(components->'backbone'->'steps', '[]'::jsonb)) AS step)
      THEN 0
      WHEN COALESCE(precision->'blocks'->'default'->>'granularity', 'PCQ') = 'BQ' THEN 1
      WHEN COALESCE(precision->'blocks'->'default'->>'granularity', 'PCQ') = 'LPBQ' THEN 2
      ELSE 3
    END AS category,

    -- Numeric bit-width from default block qtype for intra-category sorting
    COALESCE(
      NULLIF(regexp_replace(precision->'blocks'->'default'->>'qtype', '[^0-9]', '', 'g'), '')::int,
      16
    ) AS weight_bits,

    -- Rendered backbone weight tag
    CASE
      WHEN components->'backbone'->>'recipe_name' = 'RemoveQuantization'
        OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
            FROM jsonb_array_elements(COALESCE(components->'backbone'->'steps', '[]'::jsonb)) AS step)
      THEN 'FP (baseline)'
      WHEN precision IS NULL OR precision = '{}'::jsonb
        OR precision->'blocks' IS NULL THEN '—'
      ELSE
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
        ) FROM jsonb_each(precision->'blocks'))
    END AS backbone_weight_tag,

    -- Rendered backbone activation tag
    CASE
      WHEN components->'backbone'->>'recipe_name' = 'RemoveQuantization'
        OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
            FROM jsonb_array_elements(COALESCE(components->'backbone'->'steps', '[]'::jsonb)) AS step)
      THEN 'FP (baseline)'
      WHEN precision IS NULL OR precision = '{}'::jsonb
        OR precision->'blocks' IS NULL THEN '—'
      ELSE CONCAT(
        'A=', COALESCE(precision->>'activations', '?'),
        '  KV=', COALESCE(precision->>'kv_cache', 'int8'),
        '  LM=', COALESCE(precision->'lm_head'->>'qtype', '?'),
        CASE
          WHEN COALESCE(precision->'lm_head'->>'granularity', 'PCQ') <> 'PCQ'
          THEN CONCAT('/', precision->'lm_head'->>'granularity',
                      '(', COALESCE(precision->'lm_head'->>'block_size', '?'), ')')
          ELSE ''
        END
      )
    END AS backbone_activation_tag,

    -- Rendered visual precision tag (used as partition key)
    CASE
      WHEN precision->'visual' IS NULL THEN '—'
      WHEN components->'visual'->>'recipe_name' = 'RemoveQuantization'
        OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
            FROM jsonb_array_elements(COALESCE(components->'visual'->'steps', '[]'::jsonb)) AS step)
      THEN 'FP (baseline)'
      WHEN components->'backbone'->>'recipe_name' = 'RemoveQuantization'
        OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
            FROM jsonb_array_elements(COALESCE(components->'backbone'->'steps', '[]'::jsonb)) AS step)
      THEN 'FP (baseline)'
      ELSE CONCAT(
        'W=', COALESCE(precision->'visual'->'weight'->>'qtype', '?'),
        CASE
          WHEN COALESCE(precision->'visual'->'weight'->>'granularity', 'PCQ') <> 'PCQ'
          THEN CONCAT('/', precision->'visual'->'weight'->>'granularity',
                      '(', COALESCE(precision->'visual'->'weight'->>'block_size', '?'), ')')
          ELSE ''
        END,
        '  A=', COALESCE(precision->'visual'->>'activations', '?')
      )
    END AS visual_precision_tag,

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
-- Never rank/select across scoring versions within a bucket
versioned AS (
  SELECT *,
    MAX(rank_scoring_version) OVER (
      PARTITION BY backbone_weight_tag, backbone_activation_tag, visual_precision_tag
    ) AS bucket_current_scoring_version
  FROM tagged
),
current_version_only AS (
  SELECT * FROM versioned
  WHERE rank_scoring_version = bucket_current_scoring_version
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY backbone_weight_tag, backbone_activation_tag, visual_precision_tag
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
  WHERE (accuracy_results->'MMLU'->>'result') IS NOT NULL
    AND (accuracy_results->'PPL'->>'result') IS NOT NULL
	AND (accuracy_results->'AutogradedPrompts'->>'result') IS NOT NULL

)
SELECT
     CONCAT(
       model_modifiers->>'sequence_length',
       '/',
       (model_modifiers->>'context_length')::int
     ) AS "SL/CL",

     COALESCE(
       NULLIF(
         array_to_string(ARRAY(SELECT jsonb_array_elements_text(model_modifiers->'adaptations')), ', '),
         ''
       ),
       '-'
     ) AS "Adaptations",

     backbone_weight_tag     AS "Backbone Weights",
     backbone_activation_tag AS "Backbone Activations",
     visual_precision_tag    AS "Visual Precision",

     -- Backbone recipe details
     CASE
       WHEN category = 0 THEN '—'
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
     END AS "Backbone Recipe",

     -- Visual recipe details
     CASE
       WHEN components->'visual' IS NULL THEN '—'
       WHEN category = 0 THEN '—'
       WHEN components->'visual'->'steps' IS NOT NULL THEN
         (SELECT string_agg(
           CONCAT(
             step->>'recipe_name', '(',
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
         ) FROM jsonb_array_elements(components->'visual'->'steps')
                WITH ORDINALITY AS s(step, ord))
       ELSE
         CONCAT(
           components->'visual'->>'recipe_name', '(',
           COALESCE(components->'visual'->>'dataset_name', ''),
           COALESCE(':' || NULLIF(components->'visual'->'dataset_kwargs'->>'split', ''), ''),
           COALESCE(
             (SELECT ', ' || string_agg(CONCAT(kv.key, '=', kv.value #>> '{}'), ', ')
              FROM jsonb_each(components->'visual'->'recipe_kwargs') AS kv
              WHERE components->'visual'->'recipe_kwargs' IS NOT NULL
                    AND components->'visual'->'recipe_kwargs' <> '{}'::jsonb),
             ''
           ),
           ')'
         )
     END AS "Visual Recipe",

     COALESCE(model_modifiers->>'dtype', 'float32') AS "Dtype",

     -- Accuracy columns
     ROUND((accuracy_results->'PPL'->>'result')::numeric, 2)      AS "PPL",
     ROUND((accuracy_results->'TinyMMLU'->>'result')::numeric, 4) AS "TinyMMLU",
     ROUND((accuracy_results->'MMLU'->>'result')::numeric, 4)     AS "MMLU",
     ROUND((accuracy_results->'MMLU1000'->>'result')::numeric, 4) AS "MMLU1000",
     ROUND((accuracy_results->'AutogradedPrompts'->>'result')::numeric, 4) AS "AutogradedPrompts",
     ROUND((accuracy_results->'Grace'->>'result')::numeric, 4) AS "Grace",
     ROUND((accuracy_results->'MMMU'->>'result')::numeric, 4)     AS "MMMU",
     ROUND((accuracy_results->'AutogradedMultimodalPrompts'->>'result')::numeric, 4) AS "AutogradedMultimodalPrompts",

     -- Metadata
     TO_CHAR(created_at, 'YYYY-MM-DD') AS "Date",
     environment->>'actor'             AS "actor",
     environment->>'branch'            AS "branch",
     scoring_version_flags             AS "Scoring Versions"

FROM ranked
WHERE rn = 1
ORDER BY
  category ASC,
  weight_bits DESC
