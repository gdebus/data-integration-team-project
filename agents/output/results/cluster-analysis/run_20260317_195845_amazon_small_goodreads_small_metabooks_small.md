# Cluster Analysis Summary

## 1) Overall Snapshot

- **Overall status:** Mixed.
- **Healthy pair:**
  - **Metabooks ↔ Goodreads** remains healthy and meets target thresholds.
- **Unhealthy pairs:**
  - **Metabooks ↔ Amazon** is close to healthy shape-wise, but still has too many one-to-many matches.
  - **Goodreads ↔ Amazon** is the main problem area, with large ambiguous clusters and a non–long-tail distribution.
- **Target thresholds:**
  - Small cluster ratio: **> 0.90**
  - Large cluster ratio: **< 0.005**
  - One-to-many ratio: **< 0.03**

## 2) What Was Wrong (Before) vs What Happened After (Now)

### `correspondences_metabooks_small_goodreads_small.csv`
- **Status:** Healthy before and after; no material change.
- **Before vs Now**
  - Total clusters: **809 → 809**
  - Max cluster size: **6 → 6**
  - Small cluster ratio: **0.9642 → 0.9642**
  - Large cluster ratio: **0.0037 → 0.0037**
  - One-to-many ratio: **0.0206 → 0.0206**
- **Interpretation:**
  - Already in a good long-tail state.
  - No intervention was needed, and none occurred.

### `correspondences_metabooks_small_amazon_small.csv`
- **Status:** Unhealthy before and after; slight deterioration.
- **Before vs Now**
  - Total clusters: **1933 → 1880**
  - Max cluster size: **5 → 5**
  - Small cluster ratio: **0.9509 → 0.9500**
  - Large cluster ratio: **0.0000 → 0.0000**
  - One-to-many ratio: **0.0383 → 0.0388**
- **Interpretation:**
  - Cluster sizes remain controlled, with no large-cluster problem.
  - The main issue is still **ambiguous multi-match behavior**.
  - The matching adjustment did **not improve precision**; one-to-many got slightly worse.

### `correspondences_goodreads_small_amazon_small.csv`
- **Status:** Unhealthy before and after; noticeably worse.
- **Before vs Now**
  - Total clusters: **1149 → 1327**
  - Max cluster size: **19 → 31**
  - Small cluster ratio: **0.7302 → 0.6843**
  - Large cluster ratio: **0.0531 → 0.0708**
  - One-to-many ratio: **0.2246 → 0.2490**
- **Interpretation:**
  - This pair already had structural overmatching, and it became worse after the update.
  - Large ambiguous clusters expanded further.
  - The distribution moved farther away from a healthy long-tail pattern.

## 3) Deep Analysis Highlights

### Large Clusters

- **Metabooks ↔ Goodreads**
  - Large clusters are rare and capped at size **6**.
  - Only **0.37%** of clusters are large.
  - The few larger clusters are mostly small hub-like ambiguities rather than systemic failure.
  - Example pattern: one Metabooks record linked to **5 Goodreads records**.

- **Metabooks ↔ Amazon**
  - **No large clusters**.
  - This is not a cluster explosion problem.
  - The issue is narrower: too many entities still have **2–3 plausible matches**.

- **Goodreads ↔ Amazon**
  - This pair has a clear **large-cluster problem**.
  - Max cluster size increased to **31**, and **7.08%** of clusters are now large.
  - Large clusters are driven by **catalog-style overlap**: many books by the same author/publisher/year are being grouped together despite weak title agreement.
  - Representative large-cluster themes:
    - Generic/shared titles like **“The Wedding,” “The Secret,” “Ghost…”**
    - Same-author catalogs such as **David Gemmell**, **Anne McCaffrey**, **Richard Paul Evans**, **Louis L'Amour**, **Carolyn Keene**
  - These clusters are sparse-to-moderately dense bipartite structures, which usually indicates **blocking or scoring is too permissive around shared attributes**.

### One-to-Many Clusters

- **Metabooks ↔ Goodreads**
  - One-to-many ratio is low at **0.0206**.
  - Multi-match entities exist, but volume is small:
    - **17** left-side multi-match entities
    - **17** right-side multi-match entities
  - Top hubs:
    - `metabooks_67895` → **5** matches
    - `metabooks_527638` → **5** matches
  - This is acceptable noise, not a major systemic issue.

- **Metabooks ↔ Amazon**
  - One-to-many ratio is **0.0388**, above target.
  - Multi-match counts remain moderate:
    - **75** left-side multi-match entities
    - **63** right-side multi-match entities
  - Top hubs are limited to degree **3**, so ambiguity is present but contained.
  - This suggests the matcher is admitting too many **close-scoring alternatives**, not forming runaway clusters.

- **Goodreads ↔ Amazon**
  - One-to-many ratio is very high at **0.2490**.
  - Multi-match volume is severe:
    - **480** left-side multi-match entities
    - **462** right-side multi-match entities
  - Top hubs:
    - `goodreads_46432` → **8** matches
    - `goodreads_51124` / `goodreads_23870` → **7** matches
    - `amazon_31111` → **9** matches
    - `amazon_141814` / `amazon_25317` → **8** matches
  - This indicates widespread ambiguity across both datasets, not isolated bad records.

### Which Attributes Are Contributing to the Issues

- **Main pattern across problematic clusters:**  
  matches are often being supported by **publish year**, **author**, and sometimes **publisher**, while **title agreement is weak**.

- **Goodreads ↔ Amazon is especially affected by:**
  - **Publish year**
    - Nearly always agrees within tolerance in bad clusters.
    - It is acting as a **non-discriminating shared attribute**, not a reliable separator.
  - **Author**
    - Strong contributor to false links for prolific authors.
    - Same-author catalogs are being pulled together even when titles differ.
  - **Publisher**
    - Sometimes reinforces ambiguity, especially for publishers common across an author’s catalog.
  - **Title**
    - The most important missing discriminator in bad clusters.
    - In many large clusters, title match rates are extremely low even though records are linked.

- **Examples from Goodreads ↔ Amazon large clusters:**
  - Cluster 1:
    - **Publish year:** universally aligned
    - **Title match rate:** **0.1667**
    - **Author match rate:** **0.0556**
  - Cluster 2:
    - **Publish year:** universally aligned
    - **Author match rate:** **0.3448**
    - **Title match rate:** **0.1034**
  - Cluster 3:
    - **Publish year:** universally aligned
    - **Author match rate:** **0.5833**
    - **Title match rate:** **0.0417**
  - Cluster 4:
    - **Publish year:** universally aligned
    - **Author match rate:** **0.6522**
    - **Title match rate:** **0.1304**
  - Cluster 5:
    - **Publish year:** universally aligned
    - **Publisher match rate:** **0.3684**
    - **Title match rate:** **0.1579**
    - **Author match rate:** **0.0**

- **Likely root causes**
  - **Overly broad blocking** on common bibliographic fields.
  - **Matching thresholds too lenient** for records sharing author/year/publisher.
  - **Title similarity not weighted strongly enough** to separate distinct books.
  - **Edition/subtitle variation** may also be muddying title comparisons.
  - **Data quality/normalization issues** in author and publisher fields can make partial agreement look stronger than it should.

### Which Entities Are Matching with Multiple Records and Why

- **Metabooks ↔ Goodreads**
  - Main ambiguous entities:
    - `metabooks_67895` → 5 Goodreads matches
    - `metabooks_527638` → 5 Goodreads matches
  - Likely causes:
    - Generic metadata or insufficiently distinctive fields.
    - Broad candidate generation.

- **Metabooks ↔ Amazon**
  - Main ambiguous entities:
    - `metabooks_158192` → 3 matches
    - `metabooks_96995` → 3 matches
    - `amazon_63248` → 3 matches
  - Likely causes:
    - Several candidates receive similarly high scores.
    - Borderline matches are not being filtered out aggressively enough.

- **Goodreads ↔ Amazon**
  - Main ambiguous Goodreads entities:
    - `goodreads_46432` → 8 matches
    - `goodreads_51124` → 7 matches
    - `goodreads_23870` → 7 matches
    - `goodreads_148` / `goodreads_4130` → 6 matches
  - Main ambiguous Amazon entities:
    - `amazon_31111` → 9 matches
    - `amazon_141814` / `amazon_25317` → 8 matches
    - `amazon_57026` / `amazon_20532` → 7 matches
  - Underlying causes:
    - Many records share the same **author + year + publisher family**.
    - **Title mismatch is not preventing links**.
    - Series/edition catalogs are being conflated into author-level or publisher-level hubs.

## 4) Conclusion

- **Metabooks ↔ Goodreads** is stable and healthy.  
- **Metabooks ↔ Amazon** remains close to acceptable but still needs tighter precision for one-to-many matches.  
- **Goodreads ↔ Amazon** is the key failure point: the latest update worsened cluster inflation, suggesting the current blocking/scoring balance is overmatching on shared author/year/publisher signals and not separating titles strongly enough.