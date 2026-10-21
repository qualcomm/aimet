-- Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
-- SPDX-License-Identifier: BSD-3-Clause

SELECT
       TO_CHAR(t.created_at, 'YYYY-MM-DD') AS "Date",

	   CONCAT(
		 CASE
	      WHEN jsonb_typeof(t.model_modifiers->'sequence_length') = 'array'
	      THEN array_to_string(ARRAY(SELECT jsonb_array_elements_text(t.model_modifiers->'sequence_length')), ',')
	      ELSE t.model_modifiers->>'sequence_length'
	     END,
		 '/',
		 (t.model_modifiers->>'context_length')::int
	   ) AS "SL/CL",

       COALESCE(
         NULLIF(
           array_to_string(ARRAY(SELECT jsonb_array_elements_text(t.model_modifiers->'adaptations')), ', '),
           ''
         ),
         '-'
       ) AS "Adaptations",

       -- Compact precision tag (from torch row, since it drives the quantization)
       CASE
         WHEN t.components->'backbone'->>'recipe_name' = 'RemoveQuantization'
           OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
               FROM jsonb_array_elements(COALESCE(t.components->'backbone'->'steps', '[]'::jsonb)) AS step)
         THEN '—'
         WHEN t.precision IS NULL OR t.precision = '{}'::jsonb
           OR t.precision->'blocks' IS NULL THEN '—'
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
           ) FROM jsonb_each(t.precision->'blocks')),
           '  A=', COALESCE(t.precision->>'activations', '?'),
           '  KV=', COALESCE(t.precision->>'kv_cache', 'int8'),
           '  LM=', COALESCE(t.precision->'lm_head'->>'qtype', '?'),
           CASE
             WHEN COALESCE(t.precision->'lm_head'->>'granularity', 'PCQ') <> 'PCQ'
             THEN CONCAT('/', t.precision->'lm_head'->>'granularity',
                         '(', COALESCE(t.precision->'lm_head'->>'block_size', '?'), ')')
             ELSE ''
           END,
           CASE
             WHEN t.precision->'visual' IS NOT NULL
             THEN CONCAT('  · vis:W=', COALESCE(t.precision->'visual'->'weight'->>'qtype', '?'),
                         '  A=', COALESCE(t.precision->'visual'->>'activations', '—'))
             ELSE ''
           END
         )
       END AS "Precision",

       -- Recipe pipeline: Torch recipe → ONNX recipe
       CONCAT(
         -- Torch side
         CASE
           WHEN t.components->'backbone'->>'recipe_name' = 'RemoveQuantization' THEN 'FP'
           WHEN t.components->'backbone'->'steps' IS NOT NULL THEN
             (SELECT string_agg(
               CONCAT(
                 step->>'recipe_name',
                 CASE
                   WHEN step->>'blocks' IS NOT NULL
                   THEN CONCAT('[', step->>'blocks', ']')
                   ELSE ''
                 END
               ),
               ' → ' ORDER BY ord
             ) FROM jsonb_array_elements(t.components->'backbone'->'steps')
                    WITH ORDINALITY AS s(step, ord))
           ELSE t.components->'backbone'->>'recipe_name'
         END,
         ' → ',
         -- ONNX side
         CASE
           WHEN o.components->'backbone'->>'recipe_name' = 'RemoveQuantization' THEN 'FP'
           WHEN o.components->'backbone'->'steps' IS NOT NULL THEN
             (SELECT string_agg(
               CONCAT(
                 step->>'recipe_name',
                 CASE
                   WHEN step->>'blocks' IS NOT NULL
                   THEN CONCAT('[', step->>'blocks', ']')
                   ELSE ''
                 END
               ),
               ' → ' ORDER BY ord
             ) FROM jsonb_array_elements(o.components->'backbone'->'steps')
                    WITH ORDINALITY AS s(step, ord))
           ELSE o.components->'backbone'->>'recipe_name'
         END
       ) AS "Recipe Pipeline (Torch → ONNX)",

       -- Accuracy columns (from ONNX row)
       ROUND((o.accuracy_results->'PPL'->>'result')::numeric, 2)      AS "PPL",
       ROUND((o.accuracy_results->'TinyMMLU'->>'result')::numeric, 4) AS "TinyMMLU",
       ROUND((o.accuracy_results->'MMLU'->>'result')::numeric, 4)     AS "MMLU",
       ROUND((o.accuracy_results->'MMLU1000'->>'result')::numeric, 4) AS "MMLU1000",

       -- Recipe details: Torch recipe(...) → ONNX recipe(...)
       CONCAT(
         -- Torch detail
         CASE
           WHEN t.components->'backbone'->'steps' IS NOT NULL THEN
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
             ) FROM jsonb_array_elements(t.components->'backbone'->'steps')
                    WITH ORDINALITY AS s(step, ord))
           ELSE
             CONCAT(
               t.components->'backbone'->>'recipe_name', '(',
               COALESCE(t.components->'backbone'->>'dataset_name', ''),
               COALESCE(':' || NULLIF(t.components->'backbone'->'dataset_kwargs'->>'split', ''), ''),
               COALESCE(
                 (SELECT ', ' || string_agg(CONCAT(kv.key, '=', kv.value #>> '{}'), ', ')
                  FROM jsonb_each(t.components->'backbone'->'recipe_kwargs') AS kv
                  WHERE t.components->'backbone'->'recipe_kwargs' IS NOT NULL
                        AND t.components->'backbone'->'recipe_kwargs' <> '{}'::jsonb),
                 ''
               ),
               ')'
             )
         END,
         ' → ',
         -- ONNX detail
         CASE
           WHEN o.components->'backbone'->'steps' IS NOT NULL THEN
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
             ) FROM jsonb_array_elements(o.components->'backbone'->'steps')
                    WITH ORDINALITY AS s(step, ord))
           ELSE
             CONCAT(
               o.components->'backbone'->>'recipe_name', '(',
               COALESCE(o.components->'backbone'->>'dataset_name', ''),
               COALESCE(':' || NULLIF(o.components->'backbone'->'dataset_kwargs'->>'split', ''), ''),
               COALESCE(
                 (SELECT ', ' || string_agg(CONCAT(kv.key, '=', kv.value #>> '{}'), ', ')
                  FROM jsonb_each(o.components->'backbone'->'recipe_kwargs') AS kv
                  WHERE o.components->'backbone'->'recipe_kwargs' IS NOT NULL
                        AND o.components->'backbone'->'recipe_kwargs' <> '{}'::jsonb),
                 ''
               ),
               ')'
             )
         END
       ) AS "Recipe Details (Torch → ONNX)",

       COALESCE(t.model_modifiers->>'dtype', 'float32') AS "Dtype",

       -- Resource utilization (combined: torch recipe time + ONNX eval time)
       ROUND((t.components->'backbone'->'resource_utilization'->>'elapsed_ms')::numeric / 60000, 1) AS "Torch Recipe (min)",
       ROUND((o.components->'backbone'->'resource_utilization'->>'elapsed_ms')::numeric / 60000, 1) AS "ONNX Recipe (min)",
       (o.components->'backbone'->'resource_utilization'->>'cuda_peak_mb')::numeric AS "GPU Peak (MB)",

       -- Flags metrics computed under a non-default scoring version (SCORING_VERSION)
       NULLIF(
         (SELECT string_agg(CONCAT(m.key, '=v', COALESCE((m.value->>'scoring_version')::int, 1)), ', ')
          FROM jsonb_each(t.accuracy_results) AS m
          WHERE COALESCE((m.value->>'scoring_version')::int, 1) <> 1),
         ''
       ) AS "Torch Scoring Versions",
       NULLIF(
         (SELECT string_agg(CONCAT(m.key, '=v', COALESCE((m.value->>'scoring_version')::int, 1)), ', ')
          FROM jsonb_each(o.accuracy_results) AS m
          WHERE COALESCE((m.value->>'scoring_version')::int, 1) <> 1),
         ''
       ) AS "ONNX Scoring Versions",

       -- Environment columns (from torch row)
       t.environment->>'actor'      AS "actor",
       t.environment->>'branch'     AS "branch",
       t.environment->'commit_sha'  AS "commit SHA",
       t.run_group                  AS "run_group",

	   o.export 					AS "export"

    FROM genai_laboratory t
    JOIN genai_laboratory o
      ON t.run_group = o.run_group
     AND t.environment->>'variant' = 'torch'
     AND o.environment->>'variant' = 'onnx'
    WHERE t.model_id = {{model_id}}
      [[AND t.model_modifiers->'adaptations' ? {{adaptation}}]]
      [[AND t.environment->>'actor' = {{actor}}]]
      [[AND t.environment->>'branch' = {{branch}}]]
    ORDER BY t.created_at DESC