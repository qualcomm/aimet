-- Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
-- SPDX-License-Identifier: BSD-3-Clause

SELECT
       TO_CHAR(created_at, 'YYYY-MM-DD') AS "Date",

	   CONCAT(
		 CASE
	      WHEN jsonb_typeof(model_modifiers->'sequence_length') = 'array'
	      THEN array_to_string(ARRAY(SELECT jsonb_array_elements_text(model_modifiers->'sequence_length')), ',')
	      ELSE model_modifiers->>'sequence_length'
	     END,
		 '/',
		 (model_modifiers->>'context_length')::int
	   ) AS "SL/CL",

	   REPLACE(REPLACE(model_modifiers->>'image_size', ', ', '×'), ' ', '') AS "Image Size",

       COALESCE(
         NULLIF(
           array_to_string(ARRAY(SELECT jsonb_array_elements_text(model_modifiers->'adaptations')), ', '),
           ''
         ),
         '-'
       ) AS "Adaptations",

       -- Compact precision tag: backbone
       CASE
         WHEN components->'backbone'->>'recipe_name' = 'RemoveQuantization'
           OR (SELECT bool_or(step->>'recipe_name' = 'RemoveQuantization')
               FROM jsonb_array_elements(COALESCE(components->'backbone'->'steps', '[]'::jsonb)) AS step)
         THEN '—'
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
       END AS "Backbone Precision",

       -- Visual precision tag
       CASE
         WHEN components->'visual' IS NULL THEN '—'
         WHEN components->'visual'->>'recipe_name' = 'RemoveQuantization' THEN '—'
         WHEN precision->'visual' IS NOT NULL
         THEN CONCAT(
           'W=', COALESCE(precision->'visual'->'weight'->>'qtype', '?'),
           CASE
             WHEN COALESCE(precision->'visual'->'weight'->>'granularity', 'PCQ') <> 'PCQ'
             THEN CONCAT('/', precision->'visual'->'weight'->>'granularity',
                         '(', COALESCE(precision->'visual'->'weight'->>'block_size', '?'), ')')
             ELSE ''
           END,
           '  A=', COALESCE(precision->'visual'->>'activations', '?')
         )
         ELSE 'default'
       END AS "Visual Precision",

       -- Backbone recipe pipeline
       CASE
         WHEN components->'backbone'->>'recipe_name' = 'RemoveQuantization' THEN '—'
         WHEN components->'backbone'->'steps' IS NOT NULL THEN
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
           ) FROM jsonb_array_elements(components->'backbone'->'steps')
                  WITH ORDINALITY AS s(step, ord))
         ELSE components->'backbone'->>'recipe_name'
       END AS "Backbone Recipe",

       -- Visual recipe
       CASE
         WHEN components->'visual' IS NULL THEN '—'
         WHEN components->'visual'->>'recipe_name' = 'RemoveQuantization' THEN '—'
         WHEN components->'visual'->'steps' IS NOT NULL THEN
           (SELECT string_agg(step->>'recipe_name', ' → ' ORDER BY ord)
            FROM jsonb_array_elements(components->'visual'->'steps')
                 WITH ORDINALITY AS s(step, ord))
         ELSE components->'visual'->>'recipe_name'
       END AS "Visual Recipe",

       -- Accuracy columns
       ROUND((accuracy_results->'PPL'->>'result')::numeric, 2)      AS "PPL",
       ROUND((accuracy_results->'TinyMMLU'->>'result')::numeric, 4) AS "TinyMMLU",
       ROUND((accuracy_results->'MMMU'->>'result')::numeric, 4)     AS "MMMU",
       ROUND((accuracy_results->'MMLU'->>'result')::numeric, 4)     AS "MMLU",
       ROUND((accuracy_results->'MMLU1000'->>'result')::numeric, 4) AS "MMLU1000",
	   ROUND((accuracy_results->'AutogradedPrompts'->>'result')::numeric, 4) AS "AutogradedPrompts",
	   ROUND((accuracy_results->'AutogradedMultimodalPrompts'->>'result')::numeric, 4) AS "AutogradedMultimodalPrompts",
	   ROUND((accuracy_results->'Grace'->>'result')::numeric, 4) AS "Grace",
	   -- Grace's recurring failure modes, so a regression is triageable from the
	   -- table without opening the run log.
	   NULLIF(
	     (SELECT string_agg(item, '; ' ORDER BY ord)
	      FROM jsonb_array_elements_text(
	             COALESCE(accuracy_results->'Grace'->'details'->'summary_items', '[]'::jsonb)
	           ) WITH ORDINALITY AS s(item, ord)),
	     ''
	   ) AS "Grace Defects",

       -- Flags metrics computed under a non-default scoring version (SCORING_VERSION)
       NULLIF(
         (SELECT string_agg(CONCAT(m.key, '=v', COALESCE((m.value->>'scoring_version')::int, 1)), ', ')
          FROM jsonb_each(accuracy_results) AS m
          WHERE COALESCE((m.value->>'scoring_version')::int, 1) <> 1),
         ''
       ) AS "Scoring Versions",

       -- Backbone recipe details
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
       END AS "Backbone Recipe Details",

       -- Visual recipe details
       CASE
         WHEN components->'visual' IS NULL THEN '—'
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
       END AS "Visual Recipe Details",

       COALESCE(model_modifiers->>'dtype', 'float32') AS "Dtype",

       -- Resource utilization: backbone
       ROUND((components->'backbone'->'resource_utilization'->>'elapsed_ms')::numeric / 60000, 1) AS "Backbone Time (min)",
       (components->'backbone'->'resource_utilization'->>'cuda_peak_mb')::numeric AS "Backbone GPU Peak (MB)",

       -- Resource utilization: visual
       ROUND((components->'visual'->'resource_utilization'->>'elapsed_ms')::numeric / 60000, 1) AS "Visual Time (min)",
       (components->'visual'->'resource_utilization'->>'cuda_peak_mb')::numeric AS "Visual GPU Peak (MB)",

       -- Environment columns
       environment->>'actor'                            AS "actor",
       environment->>'branch'                           AS "branch",
       environment->'commit_sha'                        AS "commit SHA",

	   export											AS "export",

	   -- Row identity for a per-run report link. NULL on pre-cutover rows, so a
	   -- link built from it renders inert instead of matching every legacy row.
	   NULLIF(run_group, '')                            AS "run_group"

    FROM genai_laboratory
    WHERE model_id = {{model_id}}
      AND environment->>'variant' = 'onnx'
      [[AND model_modifiers->'adaptations' ? {{adaptation}}]]
      [[AND environment->>'actor' = {{actor}}]]
      [[AND environment->>'branch' = {{branch}}]]
    ORDER BY created_at DESC