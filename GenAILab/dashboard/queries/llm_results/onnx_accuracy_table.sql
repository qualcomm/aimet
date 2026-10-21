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

     COALESCE(
       NULLIF(
         array_to_string(ARRAY(SELECT jsonb_array_elements_text(model_modifiers->'adaptations')), ', '),
         ''
       ),
       '-'
     ) AS "Adaptations",

     -- Compact precision tag
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
         END,
         CASE
           WHEN precision->'visual' IS NOT NULL
           THEN CONCAT('  · vis:W=', COALESCE(precision->'visual'->'weight'->>'qtype', '?'),
                       '  A=', COALESCE(precision->'visual'->>'activations', '—'))
           ELSE ''
         END
       )
     END AS "Precision",

     -- Recipe pipeline (compact names)
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
     END AS "Recipe Pipeline",

	 -- Accuracy columns
     ROUND((accuracy_results->'PPL'->>'result')::numeric, 2)      AS "PPL",
     ROUND((accuracy_results->'TinyMMLU'->>'result')::numeric, 4) AS "TinyMMLU",
     ROUND((accuracy_results->'MMLU'->>'result')::numeric, 4)     AS "MMLU",
     ROUND((accuracy_results->'MMLU1000'->>'result')::numeric, 4) AS "MMLU1000",
	 ROUND((accuracy_results->'AutogradedPrompts'->>'result')::numeric, 4) AS "AutogradedPrompts",
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

     -- Recipe details (with dataset:split and kwargs)
     CASE
       WHEN components->'backbone'->'steps' IS NOT NULL THEN
         (SELECT string_agg(
           CONCAT(
             step->>'recipe_name',
             CASE
               WHEN step->>'blocks' IS NOT NULL
               THEN CONCAT('[', step->>'blocks', ']')
               ELSE ''
             END,
             '(',
             CASE
               WHEN step->>'dataset_name' = 'Interleaved'
                 AND step->'dataset_kwargs'->'source_datasets' IS NOT NULL
               THEN (SELECT string_agg(
                       CONCAT(sd->>'name', COALESCE(':' || NULLIF(sd->>'split', ''), '')),
                       '+' ORDER BY ord)
                     FROM jsonb_array_elements(step->'dataset_kwargs'->'source_datasets')
                          WITH ORDINALITY AS s(sd, ord))
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
           components->'backbone'->>'recipe_name',
           '(',
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
     END AS "Recipe Details",

     COALESCE(model_modifiers->>'dtype', 'float32') AS "Dtype",

     -- Resource utilization columns
     ROUND((components->'backbone'->'resource_utilization'->>'elapsed_ms')::numeric / 60000, 1) AS "Time (min)",
     (components->'backbone'->'resource_utilization'->>'cuda_peak_mb')::numeric AS "GPU Peak (MB)",

     -- Environment columns
     environment->>'actor'                            AS "actor",
     environment->>'branch'                           AS "branch",
     environment->'commit_sha'                        AS "commit SHA",

	 export											  AS "export",

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