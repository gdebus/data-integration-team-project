# Agent VII — Reasoning Traces (Verbatim Investigator Output)

This file contains verbatim quotes from `decision.diagnosis`, `decision.reasoning`,
`decision.ceiling_assessment`, and `decision.recommendations` extracted from
`output/runs/<run_id>/investigation/investigation_*.json` for representative
runs across the Games, Music, and Books domains.

The website's Agent VII Reasoning Case Study links here for the deep traces.
All text in blockquotes is exactly what the investigator wrote — nothing has
been paraphrased.

For aggregate routing statistics across all 28 clean runs, see the *Cross-Domain
Reasoning Analysis* subsection on the project website.

---

## Games — Run A (matching → adaption → gate, val 0.85, test 0.824)

Run ID: `20260423_103054_games`

### Attempt 1 — decision: `run_matching_tester`

**Diagnosis (verbatim).**

> The dominant error is not normalization or fusion resolver choice; it is over-linking/ambiguous entity matching that merges different games into the same cluster. Evidence: worst mismatches are semantic substitutions across title/version/platform/year (e.g. Plants vs Zombies: Garden Warfare 2 fused as Lego Star Wars: The Force Awakens; NBA 2K12 fused as Madden NFL 07; Civilization V fused as Civilization IV). Those are wrong entity pairs, not formatting differences. Cluster diagnostics reinforce this: one-to-many ratios of 62-66%, ambiguity ratio >71%, and extremely large clusters (max_size >1000) indicate many competing candidate links and insufficient disambiguation. The current fusion then propagates wrong matched records, causing catastrophic releaseYear/platform/name/developer errors. Normalization already aligns simple formats, and the worst attributes are explicitly assessed as structurally limited at current matching structure. Fusion changes alone cannot repair clusters that already contain the wrong records.

**Reasoning (verbatim).**

> Matching should be fixed next because blocking/matching artifacts show high ambiguity despite decent pairwise F1, and the observed output errors are classic false-positive matches between related but distinct games/franchises/editions/platform releases. The protected attributes above 85% should be preserved; changing fusion globally risks regressing them while leaving the underlying wrong-entity clusters untouched. Although publisher/criticScore/ESRB have some fusion headroom, the largest current failures come from mislinked entities, and pipeline_adaption is frozen from changing matching/post-link evidence. The cluster analysis already recommends MaximumBipartiteMatching; matching evaluation should tighten thresholds and disambiguate sequel/platform/version collisions before any further fusion tuning. Given the structural ceiling estimate (~57.5%), current 48.8% macro is not far below what this matching structure supports; the remaining headroom is mostly from reducing false positives, not formatting tweaks.

**Ceiling assessment (verbatim).**

> With current source coverage and validation alignment, achievable overall accuracy appears capped around 57.5% unless matching ambiguity is reduced. Tightening/entity-disambiguating matching could likely recover most of the reported 8.7-8.8% headroom; fusion-only changes are unlikely to exceed the low-to-mid 50s because many fused clusters currently contain the wrong games.

**Per-attribute recommendations.**

- **name** (improvable, impact 0.04)
  - *Issue.* False-positive matches between sequel/version/franchise-neighbor titles are causing fused names to come from entirely different games.
  - *Fix.* In run_matching_tester, increase weight on exact/near-exact title agreement and raise thresholds for metacritic_sales and metacritic_dbpedia; add stricter comparator behavior for name (e.g. stronger jaro_winkler requirement, consider exact token overlap gating) and apply MaximumBipartiteMatching after pair scoring.
- **releaseYear** (structurally_limited, impact 0.02)
  - *Issue.* Years are wildly wrong because clusters contain older/different franchise entries; earliest resolver amplifies this by selecting the oldest date in an already bad cluster.
  - *Fix.* In run_matching_tester, use releaseYear as a stronger disambiguator for pair acceptance, especially across sequel/remaster candidates; reduce allowed date tolerance or require stronger agreement when names are similar but not exact.
- **platform** (improvable, impact 0.025)
  - *Issue.* Wrong platforms (e.g. Wii U vs Nintendo 64, PS3 vs PS2) show cross-platform edition mismatches entering the same entity cluster.
  - *Fix.* In run_matching_tester, strengthen platform agreement in the matcher, especially for title-similar records; require higher platform similarity or use platform as a gating feature for ambiguous title matches.
- **developer** (structurally_limited, impact 0.015)
  - *Issue.* Developer values from unrelated titles are being fused because of bad links; additionally, similar franchise records can have different studios across versions/remasters.
  - *Fix.* In run_matching_tester, lower the influence of developer as a soft similarity when it causes broad franchise matching, but use it as a supporting tie-breaker only after strong title/platform evidence; inspect ambiguous many-to-one pairs with high name similarity but divergent developer/platform.

### Attempt 2 — decision: `pipeline_adaption`

**Diagnosis (verbatim).**

> The primary actionable defect is in fusion logic, specifically the name attribute being fused with the wrong resolver. The pipeline currently uses maximum for name, which is a numeric-oriented resolver and is producing an entirely null name column (100% null, 0% accuracy) despite all three sources having 100% non-null names and metacritic matching the validation set exactly 65% of the time. This explains the catastrophic regression of name_accuracy from 40% to 0%. Matching is not the best next fix because current evidence shows blocking recall issues are absent, correspondences are non-empty, matching F1 is already high (~0.86-0.89), and most other attributes improved substantially in the current iteration. Some residual platform/releaseYear/userScore errors are structurally limited and likely due to remaining wrong links, but the largest recoverable loss right now is the broken name fusion rule. Protected attributes developer, publisher, ESRB, and criticScore are already above 90% and should not be changed.

**Reasoning (verbatim).**

> The highest-impact improvable issue is name. This is not a normalization problem because source names are already non-null and in the same target format style as the validation set; the failure is that the fusion stage chooses/produces no value. This is also not primarily a blocking problem because correspondence files are present and blocking probes found no recall issues. Although cluster diagnostics show some ambiguity, changing matching first would be lower leverage than fixing the obviously broken name resolver. In pipeline_adaption we can safely replace the name fuser without touching well-performing attributes. Source-vs-validation comparison already shows the best direction: prefer metacritic over sales over dbpedia for name. Since 16 attributes are above 85%, recommendations should be narrowly targeted to avoid regressions.

**Ceiling assessment (verbatim).**

> With the current matching structure, overall accuracy should be able to rise from 0.75 toward the reported ~0.888 ceiling. The biggest immediate gain is recovering name from 0% toward roughly the metacritic-aligned range; a realistic short-term outcome after fixing only name fusion is an overall improvement of about 0.08-0.13. Platform/releaseYear/userScore appear near their current structural ceiling unless matching is revisited later.

**Per-attribute recommendations.**

- **name** (improvable, impact 0.125)
  - *Issue.* The pipeline uses maximum for a text field, and the fused name column is entirely null even though all sources provide non-null names. This is the direct cause of 0% name accuracy.
  - *Fix.* Replace strategy.add_attribute_fuser("name", maximum) with strategy.add_attribute_fuser("name", prefer_higher_trust, trust_map={"metacritic": 3, "sales": 2, "dbpedia": 1}).
- **platform** (structurally_limited, impact 0.0)
  - *Issue.* Residual wrong platform values remain, but current accuracy is already 70% and diagnostics classify this attribute as structurally limited under current matching structure.
  - *Fix.* Preserve the current platform resolver for now; do not change it in pipeline_adaption until name is fixed and re-evaluated.
- **developer** (structurally_limited, impact 0.0)
  - *Issue.* Developer is already at 90% accuracy and should be protected.
  - *Fix.* Keep strategy.add_attribute_fuser("developer", prefer_higher_trust, trust_map=trust_map) unchanged.
- **publisher** (structurally_limited, impact 0.0)
  - *Issue.* Publisher is already at 90% accuracy and should be protected.
  - *Fix.* No change.
- **ESRB** (structurally_limited, impact 0.0)
  - *Issue.* ESRB is already at 90% accuracy and should be protected.
  - *Fix.* No change.
- **criticScore** (structurally_limited, impact 0.0)
  - *Issue.* criticScore is already at 100% accuracy.
  - *Fix.* No change.

---

## Games — Run B (matching → matching → gate, val 0.85, test 0.832)

Run ID: `20260423_105253_games`

### Attempt 1 — decision: `run_matching_tester`

**Diagnosis (verbatim).**

> The dominant failure is incorrect entity grouping, not attribute-level fusion or simple normalization. The mismatch examples show whole records being fused to the wrong game/platform/year/developer (e.g. Zelda Twilight Princess HD fused as Zelda II; NBA 2K12 fused as NBA with PS2/developer=PlayStation 2). This pattern is consistent across multiple attributes within the same fused IDs, which indicates bad correspondences and ambiguous many-to-many clusters. Cluster analysis confirms this: very high one-to-many ratios (63%+), ambiguity ratio 73.3%, and huge clusters. Because blocking/matching are frozen inside pipeline_adaption, changing resolvers now would only choose among already-wrong source records. Normalization is not the primary issue for the worst attributes; source-vs-validation exact rates near zero across all sources show the evaluated fused rows are usually attached to the wrong source entity rather than suffering mere formatting differences. The next fix should reduce false matches and enforce better 1-to-1 clustering/post-match selection before any further fusion tuning.

**Reasoning (verbatim).**

> Routing to matching is the highest-leverage step because: (1) 12 attributes are already above 85% and should be preserved; broad normalization or fusion changes risk regressing them. (2) Worst attributes are marked structurally_limited with ceilings at or near current performance for several fields, meaning fusion cannot recover correctness when the cluster contains the wrong game. (3) Matching F1 around 0.86-0.89 looks decent globally, but cluster diagnostics reveal pairwise ambiguity severe enough to corrupt downstream entity fusion. (4) Post-clustering recommendation of MaximumBipartiteMatching supports the same conclusion, but the core issue still originates in overly permissive/ambiguous matches; the matching stage must be tightened and re-evaluated. Preserve high-performing attributes and revisit fusion only after correspondence quality improves.

**Ceiling assessment (verbatim).**

> With current source structure and after fixing correspondence quality, overall accuracy likely can rise from 42.5% toward the reported structural ceiling around 57.5%. Most remaining gap is structurally limited in name/platform/releaseYear/developer/criticScore because source coverage or canonical value agreement is weak. Additional gains beyond ~58% would likely require better source data or evaluation-ID alignment assumptions, not just resolver tweaks.

**Per-attribute recommendations.**

- **entity_matching** (improvable, impact 0.12)
  - *Issue.* Pairwise matches are too permissive and ambiguous, creating one-to-many clusters that combine different games/ports/remasters into a single fused entity.
  - *Fix.* In run_matching_tester, tighten thresholds and reduce comparator over-reliance on fuzzy title similarity for sequel/remaster-heavy titles. Re-evaluate pairwise match quality with stricter thresholds, stronger penalties for platform/releaseYear disagreement, and inspect whether Exact/near-exact name matching plus platform/date constraints outperforms current weighted rule-based settings. After matcher outputs improve, apply MaximumBipartiteMatching during clustering as already recommended.
- **name** (improvable, impact 0.045)
  - *Issue.* Title similarity is matching related but distinct games (sequels, remasters, franchise variants, shortened titles like NBA vs NBA 2K12).
  - *Fix.* In run_matching_tester, test stricter name logic: higher threshold for jaro_winkler on name, add token-sensitive comparator behavior if available, and require supporting agreement on platform and releaseYear before accepting close franchise-name matches.
- **platform** (improvable, impact 0.03)
  - *Issue.* Platform is often from a different release/port, showing platform disagreement is not preventing false matches strongly enough.
  - *Fix.* In run_matching_tester, increase platform weight and add stricter acceptance criteria so cross-platform variants are less likely to match unless title/date evidence is exceptionally strong. Keep later normalization for canonical synonyms like Microsoft Windows→PC separate if needed after matching improves.
- **releaseYear** (structurally_limited, impact 0.02)
  - *Issue.* Years are consistently wrong because records from different editions/releases are being linked; source exact rates are 0% across all sources under current alignment.
  - *Fix.* In run_matching_tester, strengthen the date comparator influence and test narrower allowed differences for releaseYear, especially for metacritic-sales and dbpedia-sales pairs. Consider rejecting matches with large year gaps even when names are similar.
- **developer/criticScore** (structurally_limited, impact 0.015)
  - *Issue.* These fields inherit wrong values from wrongly matched entities; examples include developer becoming a platform value and critic scores from unrelated titles.
  - *Fix.* Do not change their fusion resolvers yet. First repair matching/clustering, then reassess whether prefer_higher_trust or favour_sources is best using post-fix source-vs-validation comparisons.

### Attempt 2 — decision: `run_matching_tester`

**Diagnosis (verbatim).**

> The dominant error is still incorrect entity matching/clustering, not fusion or normalization. Multiple mismatch examples show the same fused ID carrying a coherent but wrong game/version across several attributes at once (platform, developer, releaseYear, name), which indicates source records for different editions/ports/sequels were linked together. Cluster diagnostics support this: very high one-to-many ratios and ambiguity, even after post-clustering. The only clearly improvable attribute from source-vs-validation evidence is publisher, but its current 70% accuracy is downstream of the same matching issue and isolated fusion tuning would not address the broader cross-attribute substitutions. High-performing attributes (>85%) should be preserved.

**Reasoning (verbatim).**

> Matching is the highest-leverage next fix because: (1) most remaining bad attributes are marked structurally limited under current correspondences; (2) mismatch samples show wrong record identity, not just wrong resolver choice; (3) source-vs-validation rates for platform are weak across all sources, so fusion cannot reliably recover correctness from bad clusters; (4) publisher does have a clear best source (sales, 84% exact), but publisher alone contributes limited headroom and the current pipeline already uses prefer_higher_trust for publisher. The safer next step is to tighten matcher acceptance for sequel/remaster/cross-platform variants and preserve already-good attributes. After correspondence quality improves, re-check whether publisher needs any additional fusion adjustment.

**Ceiling assessment (verbatim).**

> Given the provided improvability analysis, current structure supports an overall ceiling around 86.2%. Most residual error after a matching fix is likely structural in platform/name/developer/releaseYear because sources often disagree by edition/port or lack the exact validation value. A realistic gain from better matching is moderate rather than dramatic, but it is still the best available next move.

**Per-attribute recommendations.**

- **entity_matching** (improvable, impact 0.08)
  - *Issue.* Current rule-based matching is too permissive for franchise-similar titles and cross-platform editions, causing wrong games/releases to be merged into the same entity.
  - *Fix.* In run_matching_tester, tighten RuleBasedMatcher thresholds and strengthen disagreement penalties using existing comparators: raise thresholds above current 0.75/0.78/0.80 where precision improves, increase weight on platform and releaseYear, and test requiring stronger support beyond name similarity for metacritic-sales and dbpedia-sales. Keep MaximumBipartiteMatching post-clustering.
- **name** (improvable, impact 0.03)
  - *Issue.* Name similarity is over-linking related but distinct titles such as sequels, remasters, and franchise variants.
  - *Fix.* In run_matching_tester, test stricter acceptance around StringComparator(column='name', similarity_function='jaro_winkler'): either raise the effective score threshold for close title matches or reduce name weight unless platform/releaseYear also agree closely.
- **platform** (structurally_limited, impact 0.015)
  - *Issue.* Platform mismatches frequently reflect wrong ports/releases being matched rather than mere platform synonym formatting.
  - *Fix.* In run_matching_tester, increase platform comparator influence and test stricter rejection of pairs with weak platform agreement, especially when titles are franchise-similar. Do not route to normalization first because the main errors are Xbox/PS/Wii substitutions, not simple canonicalization.
- **publisher** (improvable, impact 0.01)
  - *Issue.* Publisher is the only clearly improvable low-accuracy attribute, but remaining misses are mixed between missing fused values and wrong-entity attachment.
  - *Fix.* Preserve current prefer_higher_trust for publisher with sales prioritized. After matching is repaired, re-evaluate publisher; if still needed in a later pipeline_adaption round, keep prefer_higher_trust(trust_map={'sales':3,'metacritic':2,'dbpedia':1}) rather than voting.

---

## Games — Run C (adaption → gate, val 0.90, test 0.832)

Run ID: `20260423_111733_games`

### Attempt 1 — decision: `pipeline_adaption`

**Diagnosis (verbatim).**

> The dominant failure is incorrect entity clustering/matching, not fusion or normalization. The worst attributes (name, platform, developer, publisher, releaseYear) all show semantically wrong values from entirely different games, e.g. expected 'Plants vs Zombies: Garden Warfare 2' fused as 'Lego Star Wars: The Force Awakens' and expected 'NBA 2K12' fused as 'Madden NFL 07'. That pattern means the pipeline is combining records from wrong entities. Fusion choices cannot recover from wrong clusters, and normalization is not the main bottleneck because mismatches are mostly cross-entity value mismatches rather than systematic formatting differences. The cluster analysis strongly supports over-linking/ambiguous matches: very high one-to-many ratios (63%+), ambiguity ratio 73.3%, and recommendation for MaximumBipartiteMatching. Current matching F1 around 0.86-0.89 is not enough given downstream fusion sensitivity, and the current pipeline does not apply the recommended post-clustering strategy. Additionally, releaseYear/platform are being used in blocking/matching despite platform variants and year differences across editions, which likely creates/accepts near-name false positives within franchises.

**Reasoning (verbatim).**

> Because blocking and pairwise matching outputs are already reasonably dense and non-empty, but downstream clusters are many-to-many and highly ambiguous, the most surgical next fix is in frozen-stage-allowed fusion/post-clustering logic: add the recommended MaximumBipartiteMatching post-clustering before fusion. This directly addresses the observed cross-game contamination while preserving the 14 attributes already above 85%. I am explicitly not recommending normalization first because the evidence is not format-driven overall; and not routing to run_matching_tester because the provided cluster analysis already identifies the actionable remedy within pipeline adaptation: post-clustering. Protected attributes such as ESRB and userScore are already at/near ceiling and should be preserved. Attribute-level resolver tweaks are low-value here because source-vs-validation exact rates are near zero for the worst attributes, showing structural limitations under current correspondences; changing fusers without fixing cluster purity risks regressions and will not solve wrong-entity fusion.

**Ceiling assessment (verbatim).**

> With current source coverage and attribute disagreement, the estimated overall ceiling is about 58.6%. Current 52.5% is only ~6.1 points below that ceiling. Applying MaximumBipartiteMatching should recover a meaningful portion of that remaining headroom by reducing false merges, but major gains beyond the high-50s are unlikely without deeper matching redesign or better source data alignment.

**Per-attribute recommendations.**

- **post_clustering** (improvable, impact 0.04)
  - *Issue.* No post-clustering is applied despite correspondence graphs being highly ambiguous and one-to-many; this causes fused records to contain multiple different games from the same source family, contaminating name/platform/developer/publisher/releaseYear simultaneously.
  - *Fix.* In pipeline_adaption, enable the recommended post-clustering strategy MaximumBipartiteMatching on the concatenated correspondences before DataFusionEngine.run. Keep existing resolvers for protected high-performing attributes unchanged.
- **name** (structurally_limited, impact 0.0)
  - *Issue.* Current voting resolver cannot correct wrong clusters; however name itself is structurally at ceiling under current correspondences, so do not change it yet until cluster purity improves.
  - *Fix.* Preserve current resolver during this iteration; reassess after MaximumBipartiteMatching. If cluster purity improves and name remains low, test shortest_string versus favour_sources/prefer_higher_trust using validation-aligned source comparison.
- **platform** (structurally_limited, impact 0.0)
  - *Issue.* Platform mismatches are mostly wrong-game values, not just formatting. There is a minor normalization issue like 'Wii' vs 'Wii U', but the main errors are cross-entity.
  - *Fix.* Preserve resolver for now and reassess after post-clustering. Do not introduce platform normalization in pipeline_adaption; if residual errors after reclustering are mostly alias-format issues, route to normalization_node next for canonical platform labels matching validation format (e.g. PS4, Wii U, Xbox 360).

---

## Games — One-shot matching (val 0.886, test 0.824)

Run ID: `20260323_072905_games`

### Attempt 1 — decision: `run_matching_tester`

**Diagnosis (verbatim).**

> The main error source is entity matching/post-clustering, not fusion or normalization. The mismatch examples show entire records fused onto the wrong metacritic IDs (e.g. FIFA Soccer 13 fused as Madden NFL 07; Plants vs Zombies: Garden Warfare 2 fused as Lego Star Wars), which simultaneously corrupts name, releaseYear, developer, platform, and publisher. This pattern is consistent with over-linked correspondences and ambiguous many-to-many clusters, confirmed by cluster analysis: huge max cluster sizes (~1000), one-to-many ratios above 63%, ambiguity ratio above 73%, and explicit recommendation for MaximumBipartiteMatching. Fusion resolver tweaks cannot recover when the cluster contains the wrong entities, and normalization is not the dominant issue because values are mostly semantically wrong, not merely differently formatted. There is a small secondary fusion opportunity on criticScore and ESRB, but routing there first would not address the dominant failure mode limiting macro accuracy.

**Reasoning (verbatim).**

> Matching F1 around 0.88 looks decent pairwise, but the correspondence graph is too dense/ambiguous for fusion: large erroneous clusters are causing cross-game contamination. The worst protected attributes already above 85% should be preserved, and changing fusion resolvers broadly would risk regression without fixing the core problem. releaseYear/name are at structural ceiling under current clusters; their probe-reported source exact-match rates near zero are a downstream symptom of wrong record alignment. The current pipeline already uses the recommended post-clustering note in guidance, but because the failure signature is pervasive wrong-entity fusion, the next step should be to retune matching thresholds/comparators with cluster quality as the target, then re-evaluate fusion afterward. Normalization should not be first because the validation format already broadly matches source formatting, and the errors are mostly wrong values rather than format mismatches.

**Ceiling assessment (verbatim).**

> With current matching structure, overall accuracy is near the observed ceiling (~63.1%). If matching ambiguity is reduced and one-to-many links are pruned, a meaningful jump beyond that ceiling is plausible; current 56.2% is not a fusion ceiling but a correspondence-quality ceiling.

**Per-attribute recommendations.**

- **matching_graph** (improvable, impact 0.12)
  - *Issue.* Pairwise matcher admits too many plausible near-duplicates, producing massive ambiguous clusters and one-to-many contamination across editions/franchises/platform variants.
  - *Fix.* In run_matching_tester, tighten thresholds and increase discriminative weight on releaseYear/platform while reducing tolerance for franchise-title similarity; then regenerate correspondences and apply MaximumBipartiteMatching or StableMatching to enforce near-1:1 links before fusion.
- **criticScore** (improvable, impact 0.026)
  - *Issue.* This attribute is one of the few genuinely improvable fusion-stage gaps, but should be addressed only after cluster quality is fixed.
  - *Fix.* After matching is corrected, in a later pipeline_adaption step compare median vs prefer_higher_trust using source-vs-validation evidence and likely keep median only if sources are symmetrically noisy.
- **ESRB** (improvable, impact 0.018)
  - *Issue.* Secondary fusion-stage gap; not the dominant source of current macro loss.
  - *Fix.* After matching correction, evaluate prefer_higher_trust versus voting/favour_sources on matched records against validation; preserve current resolver unless evidence shows a clear source winner.
- **name/developer/platform/releaseYear/publisher** (structurally_limited, impact 0.0)
  - *Issue.* Current poor accuracy is largely downstream of wrong entity alignment, not primarily resolvers or formatting.
  - *Fix.* Do not change these resolvers in pipeline_adaption yet; preserve high-performing attributes and revisit only after re-running matching with stricter constraints.

---

## Music — Three-iter adaption loop (val 0.792, test 0.751)

Run ID: `20260322_233450_music`

### Attempt 1 — decision: `pipeline_adaption`

**Diagnosis (verbatim).**

> The main remaining error is in fusion strategy, not blocking or matching. Entity matching is already strong (pairwise F1 about 0.87-0.90, correspondence files non-empty, blocking recall probe found no issues), while the worst residual errors are caused by wrong attribute resolvers and source trust choices after the correct entities are already linked. Specifically: (1) release-date is fused with prefer_higher_trust favoring musicbrainz, but source-vs-validation shows all sources have 0% exact because validation uses coarser dates and discogs has the best fuzzy alignment and full coverage; current resolver also causes missing values when the highest-trust source is null. (2) label is fused with longest_string, but validation comparison shows discogs is the only useful source (64% exact, 100% non-null; musicbrainz 0% exact and 0% non-null), so string-length selection is choosing descriptively longer but wrong labels. (3) Track-list attributes are structurally limited under the frozen pipeline because union fusion combines tracklists from mismatched editions/releases inside already-formed clusters; diagnostics explicitly cap their current ceiling at present scores, so they should not drive the next fix. Protected high-performing attributes such as name, release-country, and most others above 85% should be preserved.

**Reasoning (verbatim).**

> This should go to pipeline_adaption because the highest-impact fixes are resolver/trust-map changes only. No source-format preprocessing is required first: release-country is already aligned to validation format, ID coverage is mapped to 100%, and the key actionable evidence is about which source value should win once records are matched. release-date is marked improvable with ceiling 92.2% and highest impact. label is lower accuracy but structurally limited overall; still, its current resolver is clearly suboptimal and can be improved via source preference without changing normalization. Matching/blocking changes are not justified by the evidence, and list-track errors are mostly structural under current correspondences/post-clustering.

**Ceiling assessment (verbatim).**

> With current matched structure and frozen preprocessing, overall accuracy appears realistically liftable from 67.0% toward about 72-75%. Probe-estimated ceiling is 74.7%. Most unrecoverable gap is in track list attributes, where current cluster composition mixes editions and union inflates lists beyond validation. Fusion changes should recover much of the release-date headroom and some label headroom, but not the structurally limited track-list gap.

**Per-attribute recommendations.**

- **release-date** (improvable, impact 0.035)
  - *Issue.* Current resolver uses prefer_higher_trust with trust_map musicbrainz > discogs > lastfm, but validation comparison shows discogs has the best fuzzy alignment (64%) and 100% non-null coverage, while all sources have 0% exact due to granularity differences. Current choice also yields missing fused dates when the trusted source lacks a value.
  - *Fix.* Change release-date fuser to voting with trust_map={'discogs':3,'lastfm':2,'musicbrainz':1}. If PyDI voting does not accept trust weighting in your setup, use prefer_higher_trust with the same reordered trust map as a fallback.
- **label** (improvable, impact 0.02)
  - *Issue.* Current resolver longest_string selects overly descriptive but incorrect labels (e.g., 'not on label (tapping the vein self-released)') instead of the validation-preferred publisher label. Source comparison shows discogs is the only strong source: 64% exact, 100% non-null; musicbrainz contributes no correct non-null labels.
  - *Fix.* Change label fuser to prefer_higher_trust with trust_map={'discogs':3,'musicbrainz':1,'lastfm':1}. If you want stricter source ordering and PyDI supports it, favour_sources(['discogs','lastfm','musicbrainz']) is an alternative, but prefer_higher_trust is safer and consistent with other attributes.
- **artist** (improvable, impact 0.017)
  - *Issue.* Artist is below the protected threshold but clearly source-quality driven: musicbrainz exact=83%, much higher than discogs/lastfm, while current resolver is longest_string, which can over-append collaborators or variants.
  - *Fix.* Change artist fuser from longest_string to prefer_higher_trust with trust_map={'musicbrainz':3,'discogs':2,'lastfm':1}.
- **duration** (improvable, impact 0.013)
  - *Issue.* Current resolver maximum may overstate album duration when one source includes bonus material or alternate editions; probe flags remaining headroom here.
  - *Fix.* Re-evaluate duration resolver in pipeline_adaption, prioritizing median if PyDI supports it; otherwise keep maximum for now unless a quick validation-based resolver test shows prefer_higher_trust(discogs or musicbrainz) performs better.
- **tracks_track_name** (structurally_limited, impact 0.0)
  - *Issue.* Low accuracy is driven by structurally inflated list unions from mixed release editions inside clusters, not a simple resolver mistake.
  - *Fix.* Do not change in pipeline_adaption for now; preserve current behavior and accept structural limit under frozen matching/cluster structure.
- **tracks_track_duration** (structurally_limited, impact 0.0)
  - *Issue.* Same structural problem as track names: union produces supersets because linked records represent different editions/tracklists.
  - *Fix.* Do not change in pipeline_adaption for now; preserve current behavior.

### Attempt 2 — decision: `pipeline_adaption`

**Diagnosis (verbatim).**

> The remaining errors are primarily caused by fusion configuration, with one critical regression in release-country caused by the wrong trust map. Matching/blocking are already strong and correspondences are populated, so the linked entities are mostly correct. The strongest evidence is source-vs-validation: release-country matches validation exactly from musicbrainz 100% of the time, yet current fusion uses prefer_higher_trust with trust_map {'musicbrainz':3,'discogs':2,'lastfm':1} only in a variable, but the actual fuser call passes the generic trust_map where discogs outranks musicbrainz for some attributes? More importantly, the trajectory shows release-country collapsed from 96% to 0%, so the current pipeline adaptation introduced a harmful regression and that attribute must be restored. Label and release-date are also fusion-choice issues: label should strongly prefer discogs; release-date should not be protected by a sparse higher-trust source because missing_fused remains a major error mode. Track-list attributes are structurally limited under the frozen matching/clustering structure and should not drive the next fix. High-performing attributes like artist and name should be preserved unchanged.

**Reasoning (verbatim).**

> This should be fixed in pipeline_adaption because the actionable issues are resolver/trust-map choices after records are already matched. No new preprocessing is needed first: normalization already targets validation formats, blocking recall probes found no issues, and matching F1 is high across pairs. The most urgent problem is a fusion regression: release-country dropped from 96% to 0% despite probe evidence that musicbrainz is perfect on that field. That indicates a bad resolver/trust choice, not a normalization or matching failure. Label and release-date also show classic resolver problems supported by source-vs-validation evidence. I am preserving protected attributes above 85%, especially artist and name, because they are already stable and changing them risks regression.

**Ceiling assessment (verbatim).**

> With current correspondences and cluster structure, overall accuracy is likely recoverable into roughly the high-70s to about 80%, consistent with the probe ceiling near 80.2%. The main unrecoverable gap is in track-list attributes, where mixed editions inside clusters make list fusion structurally limited under the frozen pipeline.

**Per-attribute recommendations.**

- **release-country** (improvable, impact 0.078)
  - *Issue.* Severe fusion regression: accuracy fell from 96% to 0% even though source comparison shows musicbrainz is 100% exact, 100% fuzzy, and 100% non-null against validation while discogs is much worse. Current fusion is choosing the wrong source for this attribute.
  - *Fix.* Set release-country fuser to prefer_higher_trust with trust_map={'musicbrainz':3,'discogs':2,'lastfm':1} explicitly for that attribute. Do not reuse a generic trust map that ranks another source higher.
- **label** (improvable, impact 0.044)
  - *Issue.* Current label fusion still misses values or picks wrong ones; source comparison shows discogs is the only strong source with 90% exact and 100% non-null, while musicbrainz and lastfm contribute essentially nothing useful.
  - *Fix.* Set label fuser to prefer_higher_trust with trust_map={'discogs':3,'lastfm':1,'musicbrainz':1}.
- **release-date** (improvable, impact 0.026)
  - *Issue.* Release-date still has missing fused values and value mismatches. Source comparison shows no source is exact due to granularity differences, but discogs has the best fuzzy alignment and full non-null coverage; current source preference can still underperform if it favors a sparser source.
  - *Fix.* Change release-date fuser to prefer_higher_trust with trust_map={'discogs':3,'lastfm':2,'musicbrainz':1}. If PyDI voting for dates is reliable in your setup, test voting second only after restoring source priority, but the safer immediate fix is discogs-first trust.
- **tracks_track_name** (structurally_limited, impact 0.0)
  - *Issue.* Low accuracy is driven by mixed-edition tracklist unions inside already-formed clusters, producing supersets and alternate track variants relative to validation.
  - *Fix.* Preserve current behavior for now; do not change this attribute in pipeline_adaption.
- **tracks_track_duration** (structurally_limited, impact 0.0)
  - *Issue.* Zero accuracy is caused by the same structural tracklist contamination problem, not a simple scalar resolver issue.
  - *Fix.* Preserve current behavior for now; do not change this attribute in pipeline_adaption.

### Attempt 3 — decision: `pipeline_adaption`

**Diagnosis (verbatim).**

> The main remaining errors are in fusion configuration, not blocking or matching. Correspondences are populated and matcher F1 is strong across all source pairs, while source-vs-validation analysis already shows clear per-attribute source dominance. The most critical defect is a harmful fusion regression on release-country: validation matches MusicBrainz exactly (100% exact, 100% non-null), yet current fused accuracy is 0%, so the active fusion behavior is effectively selecting or outputting the wrong value for that attribute despite correct linkage. Label is also a pure fusion-choice issue with Discogs clearly dominant. Release-date still has headroom, but it is secondary and may need a safer resolver choice after the country regression is fixed. Track-list fields are structurally limited under the frozen clustering/matching structure because clusters mix editions and union-style list fusion creates supersets and alternate variants; these should not be the next target. Protected high-performing attributes (artist, name, tracks_track_position, duration) should be preserved.

**Reasoning (verbatim).**

> This should be fixed next in pipeline_adaption because the actionable issues are attribute-level resolver/trust-map choices after entities are already matched. No new preprocessing is required for the top improvable attributes: validation already expects full country names and the normalization step explicitly targeted that format; blocking probes found no recall issues; correspondence files are non-empty; matching quality is high. The strongest evidence is release-country collapsing from 96% to 0% across iterations while MusicBrainz remains 100% exact against validation. That points to a bad fusion setup or attribute-specific trust application, not a normalization problem. Label likewise has one clearly superior source (Discogs at 90% exact, 100% non-null), so prefer_higher_trust is the correct resolver. Release-date has improved but still suffers from missing values and granularity mismatch; because exact matches are zero for all sources, it should be adjusted cautiously and only for that attribute. I am explicitly preserving protected attributes above 85%: artist (87%), name (91%), tracks_track_position (91.3%), duration (95.7%).

**Ceiling assessment (verbatim).**

> With current blocking/matching and frozen cluster structure, overall accuracy appears capped around the reported ceiling of 83.4%. The largest unrecoverable gap is in track-list attributes, especially tracks_track_name and tracks_track_duration, where mixed-edition clustering and list fusion create structural errors. A careful fusion fix for release-country, label, and possibly release-date should recover several points, likely bringing the pipeline close to the low-80s, but not fully solving the track-level ceiling without changing earlier stages.

**Per-attribute recommendations.**

- **release-country** (improvable, impact 0.078)
  - *Issue.* Severe fusion regression: accuracy dropped from 96% to 0% although source comparison shows MusicBrainz is 100% exact, 100% fuzzy, and 100% non-null against validation, while Discogs is much weaker.
  - *Fix.* Set the release-country attribute fuser explicitly to prefer_higher_trust with trust_map={'musicbrainz': 3, 'discogs': 2, 'lastfm': 1} and verify that this exact map is the one actually used for the attribute in DataFusionStrategy.
- **label** (improvable, impact 0.044)
  - *Issue.* Current label fusion still misses values or selects the wrong source. Discogs is the only useful source for this attribute (90% exact, 100% non-null), while MusicBrainz and LastFM contribute essentially nothing.
  - *Fix.* Set label to prefer_higher_trust with trust_map={'discogs': 3, 'lastfm': 1, 'musicbrainz': 1}. Keep other attribute fusers unchanged.
- **release-date** (improvable, impact 0.022)
  - *Issue.* Release-date has remaining missing_fused_value errors and some mismatches. No source is exact due to granularity differences, but Discogs has the best fuzzy alignment and full coverage.
  - *Fix.* Adjust release-date to prefer_higher_trust with trust_map={'discogs': 3, 'lastfm': 2, 'musicbrainz': 1}. Do not change date formatting in this stage. If a subsequent iteration still leaves errors, then test voting for this attribute only.
- **tracks_track_name** (structurally_limited, impact 0.0)
  - *Issue.* Low accuracy is driven by structural list-format and mixed-edition tracklist contamination inside clusters, producing supersets, duplicates, and alternate punctuation variants relative to validation.
  - *Fix.* Do not change this attribute in pipeline_adaption for now; preserve current behavior.
- **tracks_track_duration** (structurally_limited, impact 0.0)
  - *Issue.* Zero accuracy is tied to the same structural tracklist contamination problem affecting track names and positions across editions.
  - *Fix.* Do not change this attribute in pipeline_adaption for now; preserve current behavior.

---

## Music — Adaption→Norm→Norm pivot (val 0.782, test 0.662)

Run ID: `20260322_171209_music`

### Attempt 1 — decision: `pipeline_adaption`

**Diagnosis (verbatim).**

> The main error is in fusion strategy/post-clustering, not blocking or source normalization. Matching is reasonably strong (pairwise F1 ≈0.87-0.90, no empty correspondence files, blocking recall probe found no issues), but fusion is choosing the wrong source/resolver and many fused entities end up without the expected MusicBrainz-based eval ID, producing widespread missing_fused_value. Current fusers conflict with the source-vs-validation evidence: label uses most_complete even though only discogs has non-null/correct labels; release-date uses prefer_higher_trust with a trust map favoring musicbrainz even though exact date agreement is 0% for all sources and the probe recommends voting; name uses favour_sources preferring musicbrainz though validation is closest to lastfm/shorter canonical names; artist uses longest_string though validation favors musicbrainz. Track-list attributes are partly structurally limited, but tracks_track_position also shows avoidable missingness likely from cluster/post-cluster selection. Overall, the pipeline is matching many right entities but fusing attributes incorrectly and preserving suboptimal cluster assignments.

**Reasoning (verbatim).**

> This should go to pipeline_adaption because the needed fixes are resolver choice, trust ordering, and post-clustering only. The current normalization already targets validation formats (country names, datetime formatting, list parsing), and the remaining top improvable errors are not primarily source-format mismatches but wrong fused value selection or no selected fused representative. Blocking/matching metrics are already high enough that they are not the first bottleneck. Protected attributes above 85% should be left untouched. We should change only low-performing attributes with explicit source-vs-validation evidence and adopt the recommended MaximumBipartiteMatching post-clustering to reduce ambiguous many-to-many clusters that currently lead to missing fused outputs.

**Ceiling assessment (verbatim).**

> With current source content and unchanged matching/import structure, overall accuracy should plausibly rise from 0.64 toward about 0.80-0.86. Reaching the full estimated 0.881 ceiling is unlikely in one step because tracks_track_duration and tracks_track_name are structurally limited list attributes with source disagreement and formatting/content variation. Biggest safe gains are from label, release-date, tracks_track_position, plus better source preference for name/artist/duration.

**Per-attribute recommendations.**

- **label** (improvable, impact 0.06)
  - *Issue.* Current resolver most_complete returns empty or wrong values because musicbrainz has 0% non-null for label while discogs has 100% non-null and 64% exact to validation.
  - *Fix.* Use prefer_higher_trust for label with trust_map={'discogs': 3, 'musicbrainz': 2, 'lastfm': 1} or favour_sources(source_preferences=['discogs','musicbrainz','lastfm']).
- **release-date** (improvable, impact 0.04)
  - *Issue.* Current resolver prefer_higher_trust with global trust_map favoring musicbrainz chooses a source that never exactly matches validation; validation appears to prefer consensus/coarser dates rather than any single source.
  - *Fix.* Use voting for release-date. If API requires tie handling through trust, use voting with source preference discogs > lastfm > musicbrainz as indicated by probe.
- **artist** (improvable, impact 0.02)
  - *Issue.* Current resolver longest_string is misaligned with validation; probe shows musicbrainz is best by a large margin.
  - *Fix.* Use prefer_higher_trust for artist with trust_map={'musicbrainz': 3, 'discogs': 2, 'lastfm': 1}.
- **name** (improvable, impact 0.02)
  - *Issue.* Current resolver favour_sources preferring musicbrainz conflicts with validation, which is closest to lastfm and tends toward shorter canonical titles.
  - *Fix.* Use shortest_string for name; if implementation needs source bias, pair with source order lastfm > musicbrainz > discogs where available.
- **duration** (improvable, impact 0.02)
  - *Issue.* Current resolver maximum likely overestimates album duration when sources disagree; probe indicates discogs is the best exact source.
  - *Fix.* Use prefer_higher_trust for duration with trust_map={'discogs': 3, 'musicbrainz': 2, 'lastfm': 1}.
- **tracks_track_position** (improvable, impact 0.03)
  - *Issue.* Missing fused values remain despite source non-nullness, and ambiguous many-to-many correspondences likely produce poor cluster representatives or fragmented clusters.
  - *Fix.* Enable/replace post-clustering with MaximumBipartiteMatching as recommended by cluster analysis; keep existing list attribute fuser otherwise for now.
- **tracks_track_name** (structurally_limited, impact 0.01)
  - *Issue.* Low accuracy is driven by structural list disagreement and formatting/content differences such as numbering, punctuation, and partial tracklists.
  - *Fix.* Preserve current resolver for now; do not spend this iteration on it in pipeline_adaption beyond benefiting from MaximumBipartiteMatching. Revisit in normalization later if needed.
- **tracks_track_duration** (structurally_limited, impact 0.01)
  - *Issue.* Low accuracy is driven by structural list disagreement and differing per-track values/track coverage across sources.
  - *Fix.* Preserve current resolver for now; only expect minor indirect gains from MaximumBipartiteMatching. Do not prioritize direct fusion-rule changes in this iteration.

### Attempt 2 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The remaining errors are now primarily caused by pre-fusion formatting/representation mismatches in list-valued track attributes, especially tracks_track_name, while scalar fusion is already aligned for protected attributes. Current pipeline code has already adopted the previously suggested fusion fixes (MaximumBipartiteMatching, shortest_string for name, prefer_higher_trust for label/artist/duration, voting for release-date), and scalar metrics improved accordingly: label rose to 62.5%, release-date to 78.3%, artist/name are protected at 95.7%, release-country at 100%. The biggest regression is tracks_track_name dropping from 48% to 0%, and mismatch samples show mostly case/list-format issues rather than wrong source choice. Because the evaluation compares serialized list values strictly, even visually identical lists are flagged as case_mismatch, indicating the source list strings are not normalized into the exact target representation before fusion/evaluation. Pipeline_adaption cannot fix this because it cannot add preprocessing; blocking/matching are not the bottleneck given strong F1 and no empty correspondence files.

**Reasoning (verbatim).**

> The next fix must happen in normalization, not fusion. Protected attributes already above 85% should be preserved: tracks_track_position (87%), artist (95.7%), name (95.7%), release-country (100%), plus the other 13 high-performing attributes. Changing fusion again risks regressing these while offering little benefit, because the current code already contains the recommended resolver/post-clustering updates. The dominant remaining improvable attribute, tracks_track_name, is explicitly classified as case_mismatch and list_format_mismatch, with examples where apparent equality still fails under evaluation. That is a representation problem: normalize list-valued track names to the exact validation format before fusion, including parsing, element-wise trimming/case handling, and stable serialization. release-date still has some missing_fused cases, but it is already improving under the current fusion strategy and is not the top bottleneck. tracks_track_duration and duration remain structurally limited, so they should not drive this iteration.

**Ceiling assessment (verbatim).**

> With current blocking/matching and the improved scalar fusion already in place, a realistic ceiling remains around the reported 89.7%. Normalizing track list representation should recover a substantial part of the lost headroom, especially the 0% tracks_track_name attribute, likely bringing overall accuracy into the low-to-mid 0.8s. Full ceiling is still constrained by structurally limited track-duration and duration disagreements across sources.

**Per-attribute recommendations.**

- **tracks_track_name** (improvable, impact 0.1)
  - *Issue.* Serialized list values are not normalized to the validation representation; mismatches are dominated by case/list-format issues, and examples suggest even near-identical lists fail strict comparison.
  - *Fix.* In normalization_node, apply element-wise normalization for tracks_track_name across all sources: parse serialized lists to actual lists, strip surrounding whitespace per element, normalize Unicode punctuation/quotes consistently, lowercase elements only if the validation comparison expects lowercase list items, remove empty items, and serialize back with a stable canonical list format matching validation/output conventions. Use PyDI normalization on the column plus the existing list normalization utilities so values remain lists rather than ad hoc strings.
- **tracks_track_duration** (structurally_limited, impact 0.02)
  - *Issue.* Track-duration lists show different lengths, missing elements, and small numeric disagreements across sources; current ceiling is limited, but some errors may still be inflated by inconsistent list parsing/serialization.
  - *Fix.* In normalization_node, ensure tracks_track_duration is parsed as a true list column in every source, coerce each element to a canonical integer-string representation without decimals/whitespace, drop empty placeholders, and serialize consistently. Do not change fusion yet.
- **release-date** (improvable, impact 0.02)
  - *Issue.* Residual errors are mostly missing fused values plus one coarse-vs-specific date mismatch; scalar formatting may still be inconsistent with the validation timestamp style on some rows.
  - *Fix.* In normalization_node, verify all non-null release-date values are emitted in the exact validation datetime format with month/day defaults handled consistently (e.g., YYYY -> YYYY-01-01T00:00:00.000, YYYY-MM -> YYYY-MM-01T00:00:00.000) before fusion.
- **label** (improvable, impact 0.0)
  - *Issue.* Still below ceiling, but current fusion already correctly prefers discogs and the remaining misses are likely from unmatched missing source values rather than wrong resolver choice.
  - *Fix.* Preserve current label fusion; only ensure normalization does not alter label casing/content away from validation. No label-specific fusion change now.

### Attempt 3 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The main remaining fixable errors are pre-fusion normalization/representation issues, not blocking or matching. Blocking/matching are healthy (all correspondence files non-empty, F1≈0.87–0.90, no blocking-recall issues). The clearest improvable attributes are label and release-date, but their error pattern is still dominated by missing fused values caused by source-format/null handling rather than wrong entity pairs. The list-valued track attributes remain largely structurally limited because fused outputs often contain different tracklists than validation, not just different formatting. Current fusion already uses the resolver family previously indicated, and 18 attributes are >85%, so those should be preserved. The next fix should therefore target normalization so source values align exactly with the validation representation before fusion, especially release-date datetime completion/formatting and preservation of label values from discogs, while avoiding regressions on protected attributes.

**Reasoning (verbatim).**

> Protected attributes already performing well and should not be changed in fusion: duration, release-country, tracks_track_position, artist, name, plus the other high-performing attributes above 85%. Matching is not the bottleneck. For label, source-vs-validation evidence is decisive: discogs is 100% exact and non-null, so the remaining misses indicate the discogs label is not consistently surviving in a usable normalized form into fusion/evaluation. For release-date, all sources have 0% exact but high fuzzy agreement, which means validation expects a canonical timestamp representation with default month/day completion; this is a normalization problem first. The worst list attributes show structural disagreement in content/length, so fusion changes there would not recover much and risk harming stable attributes. Because pipeline_adaption cannot add preprocessing and the needed fixes are source transformations before fusion, normalization_node is the correct route.

**Ceiling assessment (verbatim).**

> A realistic ceiling remains around the reported 84.5% with the current structure. Most remaining headroom is in label and release-date. Track-name and track-duration are structurally limited by source disagreement, so even perfect normalization will not fully close the gap. A careful normalization pass should recover several points overall without disturbing protected attributes.

**Per-attribute recommendations.**

- **label** (improvable, impact 0.049)
  - *Issue.* Validation comparison shows discogs matches perfectly (100% exact, 100% non-null), yet fused output is often blank. This indicates label values need to be preserved/standardized pre-fusion rather than changing the resolver.
  - *Fix.* In normalization_node, apply conservative string normalization to discogs.label only: strip surrounding whitespace, normalize Unicode punctuation/spacing, convert empty strings/placeholders to null, but preserve original casing/content. Ensure no transformation drops non-empty labels. Keep fusion as prefer_higher_trust with discogs highest.
- **release-date** (improvable, impact 0.026)
  - *Issue.* All sources are fuzzy-close but 0% exact, showing the validation expects canonical timestamps with imputed month/day/time components. Current misses are mostly blank fused values plus format granularity mismatch.
  - *Fix.* In normalization_node, standardize release-date in discogs and musicbrainz to the exact validation format YYYY-MM-DDT00:00:00.000, with explicit completion rules: YYYY -> YYYY-01-01T00:00:00.000 and YYYY-MM -> YYYY-MM-01T00:00:00.000 where source granularity is partial. Preserve nulls only when truly missing. Keep current fusion resolver for release-date.
- **tracks_track_name** (structurally_limited, impact 0.0)
  - *Issue.* Errors are dominated by list mismatches with differing content/length; examples show many releases have fundamentally different tracklists across sources, not merely casing or delimiter issues.
  - *Fix.* Only apply safe list normalization in normalization_node: parse list-like strings consistently, trim elements, normalize quotes/dashes/whitespace, remove empty items, and serialize stably. Do not change fusion yet.
- **tracks_track_duration** (structurally_limited, impact 0.0)
  - *Issue.* Track-duration lists disagree in both values and list lengths; several mismatches reflect extra/missing tracks rather than formatting alone.
  - *Fix.* In normalization_node, coerce list elements to canonical integer-string tokens, remove blanks, and ensure stable serialization only. Avoid fusion-rule changes.

---

## Music — One-shot adaption (val 0.863, test 0.793)

Run ID: `20260323_095052_music`

### Attempt 1 — decision: `pipeline_adaption`

**Diagnosis (verbatim).**

> The main remaining improvable error is in fusion, specifically the release-date resolver/trust setup. Blocking and matching are already strong (F1 ~0.87-0.90, no empty correspondence files, blocking recall probe found no issues), and the worst track attributes are structurally limited list-combination problems near their current ceiling. For release-date, the current pipeline uses prefer_higher_trust with musicbrainz highest trust, but source-vs-validation shows all exact rates are 0% and musicbrainz only has the best fuzzy rate (31%), while validation often expects coarser normalized dates like YYYY-01-01 or YYYY-MM-01. This means source priority is selecting the wrong source/date granularity rather than recovering the validation-style value. Label is also fused suboptimally: source comparison shows discogs is the only non-null and only exact-matching source (31% exact, 100% non-null), but current resolver is most_complete instead of an explicit source preference. However label is classified structurally limited and should be treated as a secondary fusion improvement, not the primary bottleneck. Protected attributes above 85% (name, release-country, label_count, tracks_track_duration_count, etc.) should not be changed.

**Reasoning (verbatim).**

> This should go to pipeline_adaption because the highest-impact improvable issue is choosing the wrong fusion resolver/trust configuration, not a source-format preprocessing problem. The normalization attempt already converts dates to datetime and countries to names. The evidence does not indicate missing candidate pairs or broadly bad correspondences. The release-date failures are mostly missing_fused/value choice issues under the current resolver, and label also needs a better resolver based on explicit source quality evidence. Track list fields do show format/list mismatch, but those fixes would require pre-fusion list transformation or different list semantics and are marked structurally limited; they are not the best next fix. Since pipeline_adaption can change fusers and trust maps but not preprocessing, it is the correct next stage.

**Ceiling assessment (verbatim).**

> Estimated achievable overall accuracy with the current blocking/matching/import structure is about 74.8%. Current accuracy 62.4% leaves ~13.5 points headroom, but much of the remaining gap is in structurally limited track-list attributes. A fusion-only adaptation should recover a meaningful portion of the release-date headroom and some label headroom, but is unlikely to solve the track attributes. Realistic near-term overall accuracy after fusion changes is likely in the high 60s to low 70s.

**Per-attribute recommendations.**

- **release-date** (improvable, impact 0.035)
  - *Issue.* Current resolver prefer_higher_trust with trust_map musicbrainz>discogs>lastfm is unsupported by validation comparison and likely picks the wrong source/date variant or propagates nulls. Source-vs-validation shows exact=0% for all sources, fuzzy rates musicbrainz 31%, discogs 22%, lastfm 21%, indicating no single source is consistently correct in raw form and source-priority is not appropriate.
  - *Fix.* Change release-date fuser to voting and, if the API requires a tie-break, keep source order discogs > lastfm > musicbrainz as indicated by the probe recommendation: strategy.add_attribute_fuser('release-date', voting) or strategy.add_attribute_fuser('release-date', voting, trust_map={'discogs':3,'lastfm':2,'musicbrainz':1}) depending on PyDI signature.
- **label** (structurally_limited, impact 0.015)
  - *Issue.* Current resolver most_complete is selecting verbose but wrong labels (e.g. self-released/not-on-label variants) and misses the clear source quality difference. Probe evidence shows discogs is 31% exact, 100% non-null, while musicbrainz is 0% exact and 0% non-null.
  - *Fix.* Change label fuser to prefer_higher_trust with a label-specific trust map favoring discogs strongly: strategy.add_attribute_fuser('label', prefer_higher_trust, trust_map={'discogs':3,'lastfm':2,'musicbrainz':1})

---

## Music — Three-norm dead-end (val 0.726, test 0.645)

Run ID: `20260323_120322_music`

### Attempt 1 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The main remaining errors are dominated by two different causes: (1) normalization mismatch for release-country, where sources/pipeline produce shortened country names like 'United States' while the validation expects full canonical names like 'United States of America'; and (2) fusion-strategy mistakes for scalar attributes, especially label and likely release-date/artist/duration, where the current resolvers or trust settings choose inferior source values despite matched entities being reasonably good. The very worst track-level attributes are structurally limited under the frozen schema because the fused output is combining track lists by union at album level, creating supersets/duplicates/order issues relative to validation; these are not the best next target compared with higher-impact improvable scalar fields. Blocking/matching are not the primary problem: correspondence files are present, blocking recall probes show no issue, and matcher F1 is already high.

**Reasoning (verbatim).**

> Normalization must be fixed first because release-country errors are explicitly format-driven and pipeline_adaption cannot add preprocessing. The validation set clearly expects long-form country names, while current normalization only applies generic country_format='name', which still leaves shortened names ('United States', 'United Kingdom') that score fuzzy-high but exact-low. release-country is the top-impact improvable attribute (impact 0.057, ceiling 94.3%). After this normalization pass, the pipeline should be re-evaluated; if label/release-date/artist/duration remain low, then a follow-up pipeline_adaption should change fusion resolvers/trust. Protected attributes above 85% should remain untouched. I am not routing to matching/blocking because evidence shows acceptable candidate generation and pair quality.

**Ceiling assessment (verbatim).**

> With current structural constraints, overall accuracy appears capped around 71.4%. A normalization fix for canonical country naming should recover a meaningful part of the 17.8% improvable headroom, but track list attributes remain structurally limited unless the representation/evaluation handling changes in a later fusion adaptation. Near-term post-normalization overall accuracy should plausibly move into the low 60s, with further gains available afterward from fusion-strategy fixes for label and release-date.

**Per-attribute recommendations.**

- **release-country** (improvable, impact 0.057)
  - *Issue.* Exact-format mismatch against validation: fused/source values use shortened country names ('United States', 'United Kingdom') while validation requires full canonical forms ('United States of America', 'United Kingdom of Great Britain and Northern Ireland'). Current evidence: musicbrainz exact=41% but fuzzy=92%, which is the signature of representation mismatch rather than wrong entities.
  - *Fix.* In normalization_node, add explicit value mapping before fusion to validation-aligned canonical names for all source datasets, e.g. NormalizationSpec or preprocessing rules that map 'United States'->'United States of America', 'United Kingdom'->'United Kingdom of Great Britain and Northern Ireland', while preserving already-correct full names. Keep country output as full names, not codes.
- **label** (improvable, impact 0.039)
  - *Issue.* Current fusion uses most_complete, but probe evidence shows discogs is the only source with signal (exact=32%, non_null=100%) while musicbrainz has 0% exact and 0% non-null. This indicates a source-preference problem, though it should be handled only after normalization because current routing must prioritize pre-fusion format repair.
  - *Fix.* After normalization rerun, in a subsequent pipeline_adaption change label fuser to prefer_higher_trust with trust_map ranking discogs > musicbrainz > lastfm, or favour_sources(['discogs','musicbrainz','lastfm']) if available in your PyDI build. Do not change now in normalization_node.
- **release-date** (improvable, impact 0.035)
  - *Issue.* Current fusion uses prefer_higher_trust with global trust favoring musicbrainz, but source-vs-validation shows all sources have 0% exact and only fuzzy/partial agreement, while current errors include many missing fused values. Dates likely need a different scalar/date resolver after normalization rerun, but this is not the first routing target.
  - *Fix.* After normalization rerun, inspect source date granularity and switch in pipeline_adaption from prefer_higher_trust to earliest if validation reflects earliest known release, otherwise voting/most_complete depending on PyDI date support. Also consider source-specific trust if one source more often matches year/month completeness. Do not alter matching.
- **artist** (improvable, impact 0.017)
  - *Issue.* Artist is already relatively strong (78.3%) and therefore should be changed cautiously, but probe evidence indicates musicbrainz exact=47% is clearly best; current longest_string may over-expand values beyond validation's preferred form.
  - *Fix.* Preserve for this normalization pass. After reevaluation, only if still below target, switch in pipeline_adaption to prefer_higher_trust with musicbrainz > discogs > lastfm rather than longest_string.
- **duration** (improvable, impact 0.017)
  - *Issue.* Album duration is fairly good already (78.3%) and should be preserved during normalization. Current maximum may overshoot if some sources aggregate bonus tracks or alternate editions; probes suggest discogs is best exact source (21%) and musicbrainz only fuzzy-better, indicating source preference may beat aggregation.
  - *Fix.* No change in normalization. After reevaluation, if needed, test prefer_higher_trust with discogs > lastfm > musicbrainz versus maximum in pipeline_adaption.
- **tracks_track_name** (structurally_limited, impact 0.0)
  - *Issue.* Very low accuracy is caused by union-style supersets, duplicates, and serialized-list mismatches versus validation track lists. This is structurally limited under current representation and not the highest-value next fix.
  - *Fix.* Do not prioritize in normalization_node. If later addressed in pipeline_adaption, test intersection or source-preference over union, but expect limited gains without track-level alignment/order-aware fusion.
- **tracks_track_position** (structurally_limited, impact 0.0)
  - *Issue.* Positions are fused as union, causing lexicographic-style supersets like ['1','10','11',...,'2',...]. Accuracy is structurally limited and likely tied to album edition differences plus list evaluation.
  - *Fix.* Do not prioritize now. Later, if pipeline_adaption is attempted, prefer source-based selection rather than union, but gains may be capped.
- **tracks_track_duration** (structurally_limited, impact 0.0)
  - *Issue.* Union fusion creates oversized duration lists from multiple editions/sources, leading to extreme mismatches. This attribute is already assessed at ceiling under current structure.
  - *Fix.* Do not spend the next fix on this. Any later mitigation belongs in pipeline_adaption with non-union list handling, but expected gains are limited without track-level correspondence.

### Attempt 2 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The remaining high-impact error is still a pre-fusion normalization mismatch on release-country. The pipeline now already uses the evidence-supported resolver for this field (prefer_higher_trust with musicbrainz over discogs), yet exact accuracy remains only 47.8% while fuzzy agreement is 100% for musicbrainz. Mismatch examples show the values are semantically correct but not in the validation set's exact canonical form: e.g. 'United States' vs 'United States of America', 'United Kingdom' vs 'United Kingdom of Great Britain and Northern Ireland'. This must be fixed before fusion. The worst track-level attributes are structurally limited by unioning album track lists from multiple editions/sources, so they should not drive the next routing. Blocking and matching are not the main issue: correspondence files exist, recall probes found no issues, and matcher F1 is already high. Protected attributes artist and name are already above 90% and should be preserved.

**Reasoning (verbatim).**

> Normalization is the only correct next stage because the top improvable attribute requires transforming source values to the validation set's exact representation before fusion. pipeline_adaption cannot add inline mappings, and changing the resolver alone will not turn shortened country names into the expected canonical long-form names. The current pipeline already protects strong attributes reasonably well: artist_accuracy and name_accuracy are >91% and should not be changed. Although label and release-date may still benefit from later fusion adjustments, the routing rule requires normalization first when pre-fusion format mismatches are present. This also avoids unnecessary risk to protected attributes from broader fusion changes.

**Ceiling assessment (verbatim).**

> With current frozen structure, overall accuracy appears capped around 68.8%. A release-country canonicalization fix should recover much of the remaining improvable headroom for that attribute, but track-list attributes will likely stay near their current ceiling because union-based album-level list fusion creates supersets, duplicates, and ordering mismatches relative to the validation set. After normalization, a later pipeline_adaption pass may still improve label and possibly release-date, but the structural ceiling remains below perfect accuracy.

**Per-attribute recommendations.**

- **release-country** (improvable, impact 0.058)
  - *Issue.* Sources and fused output use shortened country names while the validation set expects full canonical long-form names. Examples: 'United States' -> 'United States of America', 'United Kingdom' -> 'United Kingdom of Great Britain and Northern Ireland'. Current evidence shows musicbrainz already has 46% exact but 100% fuzzy, confirming representation mismatch rather than entity mismatch.
  - *Fix.* In normalization_node, add explicit pre-fusion canonical value mapping for release-country across all source datasets. Map shortened names to validation-aligned names, e.g. 'United States' -> 'United States of America', 'United Kingdom' -> 'United Kingdom of Great Britain and Northern Ireland', while preserving already-correct full names. Keep output as country names, not ISO codes. Apply this before the existing prefer_higher_trust fusion.
- **artist** (improvable, impact 0.0)
  - *Issue.* Already performing at ceiling (>91%). No evidence of harmful format mismatch requiring normalization.
  - *Fix.* Preserve current normalization and fusion settings for artist; do not modify this attribute in the next pass.
- **name** (improvable, impact 0.0)
  - *Issue.* Already performing at ceiling (>91%). No evidence of normalization problems that align with validation format.
  - *Fix.* Preserve current handling for name; do not lowercase, shorten, or otherwise normalize beyond existing safe trimming.
- **tracks_track_name** (structurally_limited, impact 0.0)
  - *Issue.* Very low accuracy is caused by structural list fusion problems: union creates supersets/duplicates/order mismatches relative to a single validation track list.
  - *Fix.* Do not target in normalization_node. Revisit only later in pipeline_adaption if needed, but expect limited gains without track-level alignment.
- **tracks_track_duration** (structurally_limited, impact 0.0)
  - *Issue.* Unioned list values produce oversized duration lists and duplicates versus validation track durations.
  - *Fix.* Do not target now; leave for potential later fusion-stage mitigation only.
- **tracks_track_position** (structurally_limited, impact 0.0)
  - *Issue.* Unioned positions create supersets and lexicographic-looking serialized outputs like ['1','10',...,'2',...], which disagree with the validation's canonical track order.
  - *Fix.* Do not target in normalization_node.

### Attempt 3 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The main remaining fixable error is still pre-fusion normalization of release-country to the validation set’s exact canonical long-form names. Evidence is decisive: current fused values are semantically correct but textually shorter than the target format (e.g. 'United States' vs 'United States of America', 'United Kingdom' vs 'United Kingdom of Great Britain and Northern Ireland'). Source-vs-validation analysis shows musicbrainz already has 100% fuzzy agreement but only 48% exact, which means entity linkage and source choice are not the problem. Label is also improvable, but that is a fusion-stage source-selection/null-handling issue to tackle after normalization. Blocking and matching are not the next bottleneck: correspondence files are present, blocking probes found no recall issues, and matcher F1 is already high. Track-level name/duration errors are structurally limited under the current album-level list fusion and should not drive routing now.

**Reasoning (verbatim).**

> Routing must go to normalization_node because the highest-impact improvable attribute requires transforming source data before fusion. pipeline_adaption cannot introduce canonical mappings like 'United States' -> 'United States of America'. The current resolver for release-country is already effectively source preference toward musicbrainz, which the source-vs-validation probe supports; changing resolver alone will not create the validation-aligned text form. Protected attributes must be preserved: name_accuracy and artist_accuracy are already above 91%, and tracks_track_position_accuracy is 87%, so no broad normalization or fusion changes should target them. After release-country normalization, a later pipeline_adaption pass can address label with a discogs-favoring resolver/trust setup if needed.

**Ceiling assessment (verbatim).**

> Estimated achievable overall accuracy with current structure remains about 82.8%. A release-country canonicalization fix should recover most of that attribute’s remaining gap and yield a meaningful overall gain. However, tracks_track_name, tracks_track_duration, and duration appear near structural ceiling under current schema/fusion behavior, so perfect accuracy is not attainable without deeper track-level alignment changes outside the current next step.

**Per-attribute recommendations.**

- **release-country** (improvable, impact 0.058)
  - *Issue.* Systematic format mismatch against validation: sources/fused output use shortened country names while validation expects full canonical long-form names.
  - *Fix.* In normalization_node, add explicit value mapping before fusion for release-country across source datasets, preserving existing correct values and mapping at least: 'United States' -> 'United States of America', 'United Kingdom' -> 'United Kingdom of Great Britain and Northern Ireland'. Keep country output as names, not ISO codes or lowercased forms. Apply this normalization before existing fusion.
- **label** (improvable, impact 0.035)
  - *Issue.* Missing fused values despite discogs matching validation perfectly in the probe (100% exact, 100% non-null on evaluated cases).
  - *Fix.* Do not change in normalization_node. Preserve label formatting now and revisit in a later pipeline_adaption pass with prefer_higher_trust or favour_sources strongly prioritizing discogs if post-normalization evaluation still shows misses.
- **release-date** (improvable, impact 0.013)
  - *Issue.* Still some residual gap, but not the highest-impact problem and not clearly a current blocker.
  - *Fix.* Preserve current date normalization for now; reassess after release-country canonicalization.
- **name** (improvable, impact 0.0)
  - *Issue.* Already at ceiling (>91%) with no reported systematic target-format mismatch.
  - *Fix.* Preserve current handling; do not apply additional case folding, shortening, or title normalization.
- **artist** (improvable, impact 0.0)
  - *Issue.* Already at ceiling (>91%) with no evidence of harmful formatting differences.
  - *Fix.* Preserve current handling unchanged.
- **tracks_track_name** (structurally_limited, impact 0.0)
  - *Issue.* Low accuracy is driven by structural list mismatches across editions/sources, including supersets, alternate spellings, numbering prefixes, and edition-specific tracklists.
  - *Fix.* Do not target in normalization_node.
- **tracks_track_duration** (structurally_limited, impact 0.0)
  - *Issue.* Track duration lists are structurally mismatched due to extra/missing tracks and edition-level list differences.
  - *Fix.* Do not target in normalization_node.
- **duration** (structurally_limited, impact 0.0)
  - *Issue.* Album duration mismatches remain, but probe classifies this as structurally limited and current best source is already discogs.
  - *Fix.* Do not target now.

---

## Books — Norm-loop with adaption pivot (val 0.713, test 0.825)

Run ID: `20260322_155058_books`

### Attempt 1 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The main remaining errors are caused by fusion/post-fusion formatting choices, but the first fix must be normalization because the worst attribute with 0% accuracy (genres) is failing due to source/target format mismatch before fusion. The validation set expects genres as a plain comma-separated string/list of individual items, while the current pipeline normalizes sources into list cells and then fuses with union(separator='; '), producing nested/composite list elements like ['historical fiction, fiction, ...'] instead of atomic genres. In addition, title and author mismatches are systematic source-format issues: one source often includes series suffixes or role annotations (e.g. '(The Birmingham Family, 1)', '(Goodreads Author)', '(Illustrator)'), while validation expects cleaned canonical strings. Publisher/title source-vs-validation evidence also shows metabooks_small is the best source, but those fusion changes should wait until normalization fixes the target-alignment issues. Blocking and matching are not the bottleneck: correspondences exist, blocking recall probes show no issue, and pairwise matching F1 is already solid.

**Reasoning (verbatim).**

> Per routing rules, any fix that requires transforming source data before fusion must go to normalization_node. The highest-impact improvable errors are title and author format mismatches, and genres is entirely a pre-fusion representation mismatch. These cannot be safely fixed in pipeline_adaption because fusion is frozen to resolver/trust changes only there. Protected attributes to preserve: language (90%), isbn_clean (100%), publish_year (100%) and the other >85% attributes should not be altered. After normalization, a follow-up pipeline_adaption pass will likely still be needed to change trust/resolvers for title and publisher toward metabooks_small, but normalization is the correct first move because it addresses the target-format incompatibilities directly and removes avoidable mismatches without touching strong attributes.

**Ceiling assessment (verbatim).**

> With current entity matching/blocking structure, overall accuracy appears capped around 78.1%. A normalization pass should recover most of the improvable headroom (roughly +7% to +10% overall) by fixing title/author formatting and genres representation. Publisher and page_count remain partly structurally limited because sources genuinely disagree; even after normalization, perfect accuracy is unlikely without better source-specific fusion choices. A realistic near-term target after normalization plus later fusion adaptation is mid-to-high 70s overall.

**Per-attribute recommendations.**

- **genres** (improvable, impact 0.0)
  - *Issue.* Validation expects atomic genre items in a comma-separated representation, but current data enters fusion as list-like cells containing whole comma-joined strings from each source; union then preserves those composite strings as elements, causing systematic list_format_mismatch and 0% accuracy.
  - *Fix.* In normalization_node, normalize goodreads_small.genres and metabooks_small.genres into true atomic lists by splitting on commas, trimming whitespace, preserving original case, and removing empty items; ensure amazon_small missing values stay null/empty rather than stringified. Target output should align with validation's item granularity, not semicolon-joined composite strings.
- **title** (improvable, impact 0.044)
  - *Issue.* Systematic source formatting noise: titles often include appended series/imprint text in parentheses, while validation usually wants the canonical title string. Current fusion uses longest_string, which actively favors the noisy variant.
  - *Fix.* In normalization_node, add source-specific title cleanup rules that strip trailing parenthetical series/imprint annotations when they are auxiliary metadata rather than core title content, especially for amazon_small and goodreads_small. Preserve meaningful parenthetical content when it is part of the canonical validation title. Keep case/title text otherwise unchanged to match validation.
- **author** (improvable, impact 0.025)
  - *Issue.* Author values contain role suffixes and extra contributors, e.g. '(Goodreads Author)', '(Adaptation), Sharon Ross (Illustrator)', while validation expects the principal author only.
  - *Fix.* In normalization_node, add source-specific author cleanup: remove platform suffixes like '(Goodreads Author)', strip contributor-role annotations in parentheses, and when multiple contributors are present with role labels, keep the main author matching the leading canonical person name used by validation. Preserve mixed/title case as in validation.
- **publisher** (structurally_limited, impact 0.025)
  - *Issue.* Half the errors are source-choice or publisher hierarchy differences (imprint vs parent publisher), with some additional formatting noise like HTML entities and division suffixes.
  - *Fix.* Do not change in normalization first beyond safe decoding/cleanup if supported (e.g. '&amp;' -> '&', trimming corporate suffix noise only when it matches validation conventions). Preserve for now and revisit in pipeline_adaption after normalization rerun.
- **page_count** (structurally_limited, impact 0.013)
  - *Issue.* Sources disagree materially on numeric values, and one source (amazon_small) is often null/poor. Errors are not primarily normalization-driven.
  - *Fix.* No normalization change recommended now; revisit in pipeline_adaption after normalization if needed, likely replacing maximum with prefer_higher_trust or favour_sources toward metabooks_small based on full validation comparison.

### Attempt 2 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The dominant current failure is a regression caused by normalization/format handling, not blocking or entity matching. isbn_clean fell from 100% to 0% because the fused output now preserves 10-digit ISBN strings with leading zeroes while the validation set expects the canonical integer-like string without leading zeroes (e.g. 0618250743 -> 618250743). This is a pre-fusion representation mismatch. The genres errors are also pre-fusion/representation-driven relative to the validation format: validation stores a plain comma-separated string, while the pipeline/list handling is producing Python-list-style serialized values, causing systematic mismatches even when the underlying genre content is mostly right. Blocking/matching are not the bottleneck: correspondences exist, recall probes show no issue, and matching F1 is solid. Some remaining page_count/title/publisher gaps are fusion-choice or structural disagreements, but the highest-impact fix must first undo the normalization-format regressions.

**Reasoning (verbatim).**

> Routing must go to normalization_node because the needed fixes transform source values before fusion to match the validation set’s exact representation. The biggest impact attribute is isbn_clean (0% accuracy, previously 100%), and its errors are clearly systematic formatting differences, not wrong source choice. genres also shows target-format incompatibility: validation examples are comma-separated title-cased strings, while current outputs are serialized list objects. Protected attributes already above 85%—especially language (90%) and publish_year (100%)—should be preserved. Fusion changes for page_count/publisher/title can wait until after these normalization issues are fixed, because pipeline_adaption cannot safely repair leading-zero ISBN formatting or source string/list serialization mismatches.

**Ceiling assessment (verbatim).**

> With current matching/blocking, a realistic ceiling remains around 79% overall. Fixing normalization should recover most of the lost isbn_clean accuracy and improve genres representation materially, likely yielding the largest immediate gain. Some page_count, title, and publisher errors are structurally limited by source disagreement, so perfect accuracy is unlikely without later fusion tuning; near-term post-normalization performance should move into the 70s.

**Per-attribute recommendations.**

- **isbn_clean** (improvable, impact 0.125)
  - *Issue.* Current fused values retain leading-zero 10-digit ISBN strings, but validation expects the same identifier without leading zeroes. This caused a catastrophic regression from 100% to 0% despite sources being otherwise aligned.
  - *Fix.* In normalization_node, standardize isbn_clean in all source datasets to the validation format by treating it as a string identifier, stripping whitespace/non-digit separators, and removing leading zeroes for 10-digit ISBN values so outputs match examples like 0618250743 -> 618250743. Preserve non-null values as strings, not numeric floats.
- **genres** (improvable, impact 0.04)
  - *Issue.* Validation expects a plain comma-separated genre string in title case, but current pipeline output serializes genres in Python-list style, producing systematic representation mismatches. The content is often close, but the format is wrong for evaluation.
  - *Fix.* In normalization_node, ensure source genres remain or are converted to a flat comma-separated string matching validation style rather than list objects/stringified Python lists. Do not over-normalize into list cells before fusion. If source values are comma-delimited, keep atomic items trimmed and preserve title-style capitalization consistent with validation output.
- **page_count** (structurally_limited, impact 0.013)
  - *Issue.* Remaining errors reflect source disagreement and poor trust ordering rather than normalization. Current trust map prefers amazon_small even though probes show it is the weakest/non-null-poor source.
  - *Fix.* Do not change now in normalization. After normalization rerun, revisit in pipeline_adaption and test prefer_higher_trust with metabooks_small > goodreads_small > amazon_small, or favour_sources in that order.
- **title** (structurally_limited, impact 0.01)
  - *Issue.* Some titles include source-specific series/appended text not matching validation’s canonical title style.
  - *Fix.* Preserve for now; revisit after normalization if still needed. If errors persist, likely pipeline_adaption should retain prefer_higher_trust toward metabooks_small rather than any length-based resolver.

### Attempt 3 — decision: `pipeline_adaption`

**Diagnosis (verbatim).**

> The main remaining errors are now in fusion strategy, not normalization, blocking, or matching. Blocking/matching are healthy (all correspondence files non-empty, F1 ~0.80-0.85, no recall issues), and the prior normalization regression on isbn_clean is already fixed/protected at 100%. The current pipeline is selecting the wrong source or wrong resolver for several low-performing attributes: genres uses most_complete even though validation expects complementary genres combined from goodreads+metabooks; publisher uses most_complete even though source-vs-validation shows metabooks is exact 100% and should be trusted; title and page_count trust maps are reversed, currently favoring amazon although amazon is the weakest source in the probe data. There is also a likely source-loading mismatch in code: the pipeline loads normalization/attempt_1 while the current normalized files are in attempt_3, which may explain some regressions, but under the routing rules the next fixable stage is fusion logic because blocking/matching are frozen-good and the dominant configurable errors are resolver/trust choices.

**Reasoning (verbatim).**

> Route to pipeline_adaption because the highest-impact remaining fixes are attribute fusers and trust maps. Do not touch protected attributes already >85%, especially publish_year and isbn_clean at 100%. genres is the top improvable attribute and needs union-style fusion to match validation provenance patterns like goodreads+metabooks. publisher, title, and page_count all show wrong source preference under current trust ordering. Although the code appears to load attempt_1 normalized files instead of attempt_3, the actionable next stage within the allowed routing is still fusion adaptation: change resolvers/trust maps and preserve working attributes. Normalization should not be revisited first because current mismatch examples for the priority attributes are mostly wrong value choice, not simple formatting, and isbn_clean is no longer broken.

**Ceiling assessment (verbatim).**

> With current matching/blocking structure, overall accuracy around 0.80-0.85 appears realistic. The supplied ceiling estimate is 88.5%, but page_count/publisher/title are partly structurally limited by source disagreement, so a practical near-term target after fusion fixes is low 80s rather than perfect accuracy.

**Per-attribute recommendations.**

- **genres** (improvable, impact 0.09)
  - *Issue.* Current resolver most_complete picks a single source's longer genre string, but validation expects complementary genres combined across sources, typically goodreads_small + metabooks_small. This causes systematic under-recall and occasional missing values.
  - *Fix.* Change genres fuser from most_complete to union.
- **publisher** (improvable, impact 0.05)
  - *Issue.* Current resolver most_complete chooses verbose/imprint variants like 'Simon & Schuster (Trade Division)' or wrong imprints like 'Eos', while validation usually wants the canonical publisher name. Probe data shows metabooks_small matches validation exactly in 100% of examined cases.
  - *Fix.* Change publisher fuser from most_complete to prefer_higher_trust with trust_map {'metabooks_small': 3.0, 'goodreads_small': 2.0, 'amazon_small': 1.0}.
- **title** (improvable, impact 0.04)
  - *Issue.* Current title fuser prefer_higher_trust is configured backwards, favoring amazon_small, but probe data shows goodreads_small is the strongest source and amazon_small the weakest. This yields expanded/marketing titles instead of validation's canonical titles.
  - *Fix.* Keep prefer_higher_trust for title but reverse trust_map to {'goodreads_small': 3.0, 'metabooks_small': 2.0, 'amazon_small': 1.0}.
- **page_count** (structurally_limited, impact 0.02)
  - *Issue.* Current page_count fuser prefer_higher_trust is also reversed, favoring amazon_small even though amazon_small has 0% exact and 0% non-null in the probe sample. Validation often matches metabooks_small for page_count provenance.
  - *Fix.* Keep prefer_higher_trust for page_count but reverse trust_map to {'metabooks_small': 3.0, 'goodreads_small': 2.0, 'amazon_small': 1.0}.
- **language** (improvable, impact 0.01)
  - *Issue.* Language has regressed to 80% due to missing fused values, but it is near-protected and should not be over-tuned aggressively.
  - *Fix.* Preserve current resolver for language for now; if revisited later, test most_complete or prefer_higher_trust with metabooks-first only if missingness persists after broader fusion fixes.

---

## Books — Three-norm convergence (val 0.812, test 0.713)

Run ID: `20260323_034342_books`

### Attempt 1 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The matched entities are mostly correct; the main remaining error is fusion strategy, with one clear pre-fusion formatting problem on author. The current pipeline is choosing the wrong source/value for publisher and page_count and an overlong/noisy value for author. Evidence: publisher validation aligns strongly with metabooks_small (88% exact, best source), but the pipeline uses longest_string, which systematically prefers imprints/expanded publisher names like 'Simon & Schuster (Trade Division)' or sibling imprints like 'Eos' over the parent publisher expected by validation. Page_count validation is sourced from metabooks in the sample rows and amazon is often null/wrong; current maximum fuser over-selects inflated counts (e.g. 554 vs expected 512). Author mismatches are format noise such as '(Goodreads Author)' and role suffixes like '(Adaptation), ... (Illustrator)', which require normalization before fusion. Title is already at its structural ceiling (80%) and should not be changed. High-performing attributes (>85%) should be preserved.

**Reasoning (verbatim).**

> Because the only clearly improvable attribute is author, and its errors are due to source text formatting that must be cleaned before fusion, normalization must happen first per routing rules. Although publisher and page_count also need fusion changes, pipeline_adaption cannot fix author formatting and should not be attempted before source cleanup. After normalization removes Goodreads-style author annotations/role suffixes, the pipeline can be re-evaluated; if publisher/page_count still lag, a subsequent pipeline_adaption should switch resolvers/trust. Preserving already-strong attributes is important: genres, language, isbn_clean, publish_year and 16 total attributes above 85% should remain untouched now.

**Ceiling assessment (verbatim).**

> With current matching/blocking, the practical ceiling is about 87.5%. Author can likely move from 80% to near 100% via normalization, lifting overall accuracy by roughly 2.5 points. Publisher and page_count appear structurally capped near 70% under current source disagreement, but later fusion changes could recover part of their remaining gap without exceeding the estimated ceiling.

**Per-attribute recommendations.**

- **author** (improvable, impact 0.025)
  - *Issue.* Validation expects plain canonical author names, but source values include Goodreads annotations and contributor-role suffixes, e.g. 'John Barlow (Goodreads Author)' and 'Dina Anastasio (Adaptation), Sharon Ross (Illustrator)'. Current longest_string then amplifies the noisy variant.
  - *Fix.* In normalization_node, add regex-based cleanup before fusion: strip trailing parenthetical site annotations like '\s*\(Goodreads Author\)\s*$'; remove contributor-role suffix segments such as ',\s*[^,]*\((Illustrator|Adaptation|Editor|Translator|Introduction)\).*' when validation expects primary author only; preserve mixed/title case and comma-separated primary names. Apply with a PyDI NormalizationSpec text transform or preprocessing function on author for goodreads_small (and any other affected source).
- **publisher** (structurally_limited, impact 0.02)
  - *Issue.* Current longest_string resolver systematically prefers expanded imprint/division names or alternate imprints, while validation most often expects the canonical metabooks_small publisher string.
  - *Fix.* After normalization is completed and re-evaluated, in pipeline_adaption change publisher fuser from longest_string to prefer_higher_trust with trust_map={'metabooks_small':3,'goodreads_small':2,'amazon_small':1}. Do not normalize case now beyond existing trimming, since target uses mixed case.
- **page_count** (structurally_limited, impact 0.013)
  - *Issue.* Current maximum resolver over-selects inflated page counts and does not match validation, while amazon_small contributes no exact matches and is often null/wrong.
  - *Fix.* After normalization is completed and re-evaluated, in pipeline_adaption change page_count fuser from maximum to prefer_higher_trust with trust_map={'metabooks_small':3,'goodreads_small':2,'amazon_small':1}; if PyDI requires numeric resolver only, use favour_sources/most_complete equivalent that prefers non-null metabooks_small first. Avoid maximum.

### Attempt 2 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The dominant remaining actionable error is pre-fusion formatting in author values. Matching/blocking are adequate, and most low scores outside author are structurally limited or already near their observed ceiling. Genres collapsed from 90% to 10% because the fused representation no longer matches the validation set’s expected single comma-separated string format; however the probe explicitly classifies genres as structurally limited at the current ceiling and the current issue is fundamentally a pre-fusion/list-format normalization problem rather than matcher failure. Author is the only attribute marked improvable: validation expects plain canonical names, while sources—especially Goodreads—contain annotations like '(Goodreads Author)' and contributor-role suffixes. Since these transformations must occur before fusion, the next fix belongs in normalization, not pipeline adaptation.

**Reasoning (verbatim).**

> Routing must go to normalization first because the fix requires transforming source text before fusion. Author accuracy is plateaued at 80% with a 100% ceiling and concrete examples show pure format noise, not wrong entity linkage. Blocking recall checks found no issues, correspondence files are populated, and matching F1 is healthy, so changing blocking/matching is unlikely to help. Pipeline adaptation alone cannot strip Goodreads annotations or contributor-role suffixes. Protected attributes should be preserved: language (90%), publish_year (90%), isbn_clean (100%), plus the many other >85% attributes. Do not touch title, language, publish_year, or isbn_clean. Note that publisher/page_count may still benefit later from trust-based fusion, but per routing rules normalization must happen first because it is the only clear improvable issue and because genres/list formatting also points to source-format alignment problems.

**Ceiling assessment (verbatim).**

> Current observed structural ceiling is about 81.2% overall under the frozen matching structure. A targeted author normalization should recover roughly the available improvable headroom (~2.5 percentage points overall), pushing overall accuracy toward ~78.5-81.2% depending on side effects. Larger gains likely require a later fusion adjustment for publisher/page_count and possibly safer list normalization for genres, but those should be reconsidered only after author cleanup is re-evaluated.

**Per-attribute recommendations.**

- **author** (improvable, impact 0.025)
  - *Issue.* Source values include site annotations and contributor-role suffixes that do not match validation, e.g. 'John Barlow (Goodreads Author)' and 'Dina Anastasio (Adaptation), Sharon Ross (Illustrator)', while validation expects the primary canonical author only.
  - *Fix.* In normalization_node, add source-specific text cleanup for author, especially on goodreads_small: strip trailing '\s*\(Goodreads Author\)\s*$'; remove parenthetical contributor roles such as '\s*\((Adaptation|Illustrator|Editor|Translator|Introduction|Foreword|Afterword)\)'; when multiple contributors are present, retain the primary author portion before secondary role-tagged contributors (e.g. split on ', ' when the following segment contains a role tag). Preserve original title/mixed case; only trim whitespace after cleanup. Implement via PyDI NormalizationSpec text preprocessing or a preprocessing function applied before fusion.
- **genres** (structurally_limited, impact 0.0)
  - *Issue.* Fused output format diverges sharply from validation: validation expects one comma-separated genre string preserving case, while fused output contains list artifacts / nested chunks from list normalization and union fusion. This caused a severe regression from 90% to 10%.
  - *Fix.* In normalization_node, inspect and standardize source genres into a single consistent list representation before fusion: split embedded multi-genre strings on commas, trim whitespace, preserve title case, and avoid producing nested list elements or semicolon-based output when validation expects comma-separated values. If a source column is scalar text rather than true list, normalize it to tokenized genre items consistently before union fusion.

### Attempt 3 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The current errors are primarily caused by pre-fusion normalization problems, not blocking or entity matching. Matching/blocking look adequate (all correspondence files populated, blocking recall probe clean, pairwise matching F1 ~0.80-0.85). The strongest actionable issues are source-format mismatches against the validation format: (1) genres is being normalized/fused into list artifacts or nested chunks, while the validation expects one flat comma-separated string of atomic genre items; this regression explains the 90% -> 10% -> 0% collapse. (2) author still contains Goodreads/site annotations and contributor-role suffixes that validation does not want. Publisher is also partially format-driven, though later fusion/trust tuning may still be needed. Because these fixes require transforming source values before fusion, the next step must be normalization rather than pipeline adaptation.

**Reasoning (verbatim).**

> Route to normalization first because the required fixes are upstream text/list transformations. pipeline_adaption cannot strip '(Goodreads Author)', remove role suffixes, flatten nested genre strings, or standardize list item boundaries before fusion. Protected attributes should be preserved: language_accuracy, isbn_clean_accuracy, publish_year_accuracy, and the other >85% attributes should not be touched. I am not recommending blocking/matching changes because evidence shows sufficient correspondence coverage and no empty files. I am also not recommending immediate fusion-only changes for publisher/page_count yet, because the routing rule requires normalization first when target-format mismatches are present. After normalization is repaired, publisher can be re-evaluated for trust-map adaptation if needed.

**Ceiling assessment (verbatim).**

> Given the supplied probes, the achievable ceiling under the current linkage structure is roughly 76.8%. Normalization should recover most of the remaining improvable headroom, especially on author and possibly some publisher/genres cases if the target flat-string genre format is restored correctly. Page_count and title appear largely structurally limited with current sources, so major gains there are unlikely without different source evidence.

**Per-attribute recommendations.**

- **genres** (improvable, impact 0.07)
  - *Issue.* Validation expects a single flat comma-separated genre string, but the pipeline currently converts genre inputs into list artifacts / nested multi-genre chunks and then unions them, producing values like ['Historical Fiction, Fiction, ...'] instead of atomic items. This is a pre-fusion representation mismatch and the main reason genres fell to 0%.
  - *Fix.* In normalization_node, normalize source genres into atomic genre items before fusion. For source columns that contain delimited strings, split on commas into separate items, trim whitespace, preserve title case, and ensure each cell is represented consistently as a flat list of genre tokens with no nested chunks. Avoid semicolon-oriented formatting when the validation target uses comma-separated flat text. If using PyDI NormalizationSpec is insufficient for list tokenization, apply a preprocessing function before saving normalized CSVs so values like 'Fiction, Romance, Historical' become atomic items ['Fiction','Romance','Historical'] rather than a single list element.
- **author** (improvable, impact 0.025)
  - *Issue.* Source values include annotations and contributor-role text not present in validation, e.g. 'John Barlow (Goodreads Author)' and 'Dina Anastasio (Adaptation), Sharon Ross (Illustrator)', while validation expects the canonical primary author only.
  - *Fix.* In normalization_node, add source-specific author cleanup, especially for goodreads_small: strip trailing '(Goodreads Author)'; remove parenthetical role labels such as '(Adaptation)', '(Illustrator)', '(Editor)', '(Translator)', '(Foreword)', '(Introduction)', '(Afterword)'; when multiple contributors are present and later contributors are role-tagged, keep only the primary canonical author segment. Preserve original casing and just trim surrounding whitespace after cleanup. Implement via PyDI NormalizationSpec text cleanup if supported, otherwise a preprocessing function before normalized CSV export.
- **publisher** (improvable, impact 0.03)
  - *Issue.* Many publisher mismatches are target-format/canonical-name differences such as 'Vintage Books USA' vs 'Vintage' and 'Perennial' vs 'Harper Perennial'. Source comparison shows metabooks_small matches validation best (88% exact), but some current errors are still formatting/canonicalization issues.
  - *Fix.* In normalization_node, preserve publisher casing as title case but additionally standardize obvious suffix noise and imprint variants only where this aligns with validation examples. Focus on removing extraneous corporate/location suffixes that are not used in the validation representation, while avoiding aggressive abbreviation expansion. Re-evaluate after normalization; if publisher remains low, a later pipeline_adaption step should keep prefer_higher_trust and raise metabooks_small trust above goodreads_small/amazon_small.

---

## Books — One-shot normalization (val 0.912, test 0.887)

Run ID: `20260323_035936_books`

### Attempt 1 — decision: `normalization_node`

**Diagnosis (verbatim).**

> The pipeline is near its structural ceiling. Blocking/matching are healthy, and most low scores come from fusion/format issues in already-matched entities. The only clearly fixable next step is normalization of list/string formatting to match the validation set, especially genres. Current fusion also uses suboptimal resolvers for publisher/title, but because pipeline_adaption cannot add preprocessing and genres failures are caused by pre-fusion format handling, normalization must be fixed first. The current inline/list handling is producing malformed nested genre strings with separator mismatches against the validation format (validation expects one plain comma-separated string, while fusion emits list-like/nested-string artifacts). Publisher also shows a format variant issue (HTML entity and imprint suffixes), which is likewise normalization-oriented before any trust-based fusion tuning.

**Reasoning (verbatim).**

> Protected attributes should be preserved: author=90%, language=100%, publish_year=100%, isbn_clean=100%, plus 16 attributes >85%, so no broad fusion changes should be made now. Evidence points away from blocking/matching: all 3 correspondence files are non-empty, blocking recall probes found 0 issues, and pairwise matcher F1 is decent (0.80-0.85). The biggest miss, genres=10%, is explicitly list_format_mismatch and the sample outputs show broken serialization/chunking rather than wrong entity linking. Validation expects genres as a single comma-separated string with title case; fused output contains Python-list-looking strings and split chunks, so this requires source normalization/list-column treatment before fusion, not a resolver-only change. Publisher has some true source disagreement, but also normalization problems such as 'Simon &amp; Schuster (Trade Division)' vs 'Simon & Schuster' and 'Vintage Books USA' vs 'Vintage'. Since normalization must happen first per routing rules, fix formatting/canonicalization now; only after re-evaluation should resolver/trust changes be considered. Overall headroom is small (~1.2-1.3%), so this is a ceiling-near pipeline.

**Ceiling assessment (verbatim).**

> Estimated achievable overall accuracy with current matching structure is about 77.5%. Current 76.25% is already close. Normalization can likely recover roughly 1 point, mostly by correcting genres serialization and a few publisher formatting cases; larger gains are unlikely because publisher/title/page_count remaining errors are mostly structurally limited source disagreements.

**Per-attribute recommendations.**

- **genres** (improvable, impact 0.008)
  - *Issue.* Validation expects a single plain comma-separated title-cased string, but current preprocessing/fusion treats genres as list-like and outputs malformed nested/chunked values (e.g. Python-list representations and semicolon/list artifacts). Probe evidence: exact match is very low but fuzzy is 90-100%, indicating mostly formatting/serialization mismatch, not semantic mismatch.
  - *Fix.* In normalization_node, normalize genres to the validation format before fusion: parse any list-like representations safely into atomic genre items, flatten nested lists/quoted chunks, strip brackets/quotes, split on source delimiters, trim whitespace, preserve duplicates/order when present, and reserialize as one comma+space separated string matching validation title case. Do not rely on downstream union with separator='; '. Use PyDI NormalizationSpec/transform_dataframe for string cleanup and dataset-specific custom parsing where needed so each source stores genres as the same plain string format prior to matching/fusion.
- **publisher** (improvable, impact 0.005)
  - *Issue.* Mixed problem: some values are canonicalization/format mismatches ('Simon &amp; Schuster (Trade Division)' vs 'Simon & Schuster', 'Vintage Books USA' vs 'Vintage'), while others are source disagreement. Current fusion uses longest_string, which is poorly aligned to validation and tends to choose verbose imprint variants.
  - *Fix.* In normalization_node, canonicalize publisher strings to validation style before fusion: HTML-unescape entities, strip business-unit/imprint suffixes in parentheses where validation omits them, collapse whitespace, and standardize common imprint aliases where the validation uses the shorter canonical house name. Preserve title case. After normalization and re-evaluation, consider switching resolver later to prefer_higher_trust with trust favoring metabooks_small if publisher remains underperforming.
- **title** (structurally_limited, impact 0.0)
  - *Issue.* Current resolver is voting, but validation examples often prefer the more complete metabooks title variant rather than majority/shorter variants. However title is already at its measured ceiling (80%) and not the best next-stage target.
  - *Fix.* No change in this normalization pass; preserve current title formatting because title accuracy is already relatively high and mismatches appear to be source disagreement, not a systematic format issue.
- **page_count** (structurally_limited, impact 0.0)
  - *Issue.* Values differ across sources/editions; current maximum already matches the recommended resolver and measured ceiling.
  - *Fix.* Preserve current page_count fusion and normalization.

---

