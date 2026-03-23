# Run Report (20260323_042119)

- Generated at (UTC): `2026-03-23T03:21:19.180277Z`
- Run ID: `20260323_041229_restaurant`
- Run output root: `output/runs/20260323_041229_restaurant/`
- Validation overall (final): `0.86`
- Validation macro (final): `0.8666666666666668`
- Sealed test overall (final): `0.30037546933667086`
- Sealed test macro (final): `0.30555116783555525`
- Correspondence structurally valid: `True`
- Final route: `finish automatic passes and export for human review`

## Agent Overview
The agent profiles the datasets, optionally reruns normalization, keeps the tested blocking setup, searches or refreshes matching when needed, generates a fusion pipeline, generates and runs an evaluation script, diagnoses the result, and then routes either back to matching, back to normalization, back to pipeline adaptation, or to final human review.

## Run Narrative
- Main problem: n/a
- Next-step advice: The evidence points away from normalization: target formatting already largely matches sources, and the dominant issue is missing fused outputs plus wrong source selection. Routing to normalization would be incorrect because the most impactful fixes are frozen-fusion changes: apply the recommended post-clustering, correct per-attribute resolvers, and use attribute-specific trust orders instead of one global trust map. Blocking/matching are not the bottleneck given high pairwise F1 and non-empty correspondences. Pipeline adaptation is the correct next step because it can change fusion strategy and post-clustering without altering preprocessing.
- Report takeaway: The main failure is in fusion/output construction, not blocking or matching. Matching is already strong (F1 0.89-0.98, no empty correspondence files, blocking recall probe clean), while 93.3% of errors are missing_fused_value and direct ID coverage is only 30%. This indicates many validation records are not being surfaced under the evaluation ID the scorer expects, and for the records that do align, several low-performing attributes use the wrong fusion resolver/trust order. Current fusion wrongly privileges Yelp globally (trust_map yelp>kaggle>uber), but source-vs-validation shows source should prefer uber_eats_small, street should prefer kaggle_small, and map_url/rating are not best served by the current settings. Post-clustering guidance also recommends MaximumBipartiteMatching, which is not applied. Protected high-performing attributes (>85%) should be left unchanged.
- Latest detected problems: low attribute accuracy: _id=30.000%, source=30.000% | low direct ID coverage: 30.000%
- Top planned actions: Enable post-clustering with MaximumBipartiteMatching on the concatenated correspondences before DataFusionEngine.run, then fuse on the post-clustered correspondences. | Change source fuser to prefer_higher_trust with an attribute-specific trust_map: {'uber_eats_small': 3, 'yelp_small': 2, 'kaggle_small': 1}. | Change map_url fuser from prefer_higher_trust to voting. If voting in PyDI supports tie behavior only via trust, use voting alone first rather than Yelp-biased trust.

## Attempt Timeline
| attempt | recorded_at | raw_overall | accepted_overall | guard_rejected | structural_valid | invalid_pairs |
|---|---|---:|---:|---|---|---|
| 1 | 2026-03-23T03:17:05.405388Z | 0.2575 | 0.2575 | False | True |  |
| 2 | 2026-03-23T03:19:46.274421Z | 0.86 | 0.86 | False | True |  |
