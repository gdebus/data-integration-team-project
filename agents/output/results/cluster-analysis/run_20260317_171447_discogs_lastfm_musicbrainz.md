# Cluster Analysis Summary

## 1) Overall Snapshot

- **All 3 correspondence files remain unhealthy** against the target thresholds.
- **Target thresholds**:
  - Small cluster ratio: **>= 0.90**
  - Large cluster ratio: **<= 0.005**
  - One-to-many ratio: **<= 0.03**
- **Net result after the latest run**:
  - **`discogs_lastfm` improved substantially**
  - **`discogs_musicbrainz` improved modestly**
  - **`musicbrainz_lastfm` regressed sharply**
- Matching configuration was **not updated in this run** because **max update attempts had already been reached**.

---

## 2) What Was Wrong (Before) vs What Happened After (Now)

### `correspondences_musicbrainz_lastfm.csv`
- **Status**: **Worse**
- **Before**
  - Small cluster ratio: **0.9659**
  - Large cluster ratio: **0.0026**
  - One-to-many ratio: **0.0310**
  - Max cluster size: **9**
  - Ambiguous ratio: **0.2917**
- **Now**
  - Small cluster ratio: **0.9098**
  - Large cluster ratio: **0.0233**
  - One-to-many ratio: **0.1335**
  - Max cluster size: **39**
  - Ambiguous ratio: **0.4598**
- **Key takeaway**
  - This file moved from relatively contained ambiguity to **severe over-clustering**.
  - Large ambiguous title-driven groups are now a major issue.

### `correspondences_discogs_lastfm.csv`
- **Status**: **Much better, but still not healthy**
- **Before**
  - Small cluster ratio: **0.8343**
  - Large cluster ratio: **0.0228**
  - One-to-many ratio: **0.1676**
  - Max cluster size: **32**
  - Ambiguous ratio: **0.9130**
- **Now**
  - Small cluster ratio: **0.9529**
  - Large cluster ratio: **0.0030**
  - One-to-many ratio: **0.0416**
  - Max cluster size: **8**
  - Ambiguous ratio: **0.7857**
- **Key takeaway**
  - Cluster sprawl was reduced significantly.
  - Still failing mainly because **ambiguity remains high** and **one-to-many is still above target**.

### `correspondences_discogs_musicbrainz.csv`
- **Status**: **Improved modestly**
- **Before**
  - Small cluster ratio: **0.8771**
  - Large cluster ratio: **0.0047**
  - One-to-many ratio: **0.1257**
  - Max cluster size: **13**
  - Ambiguous ratio: **0.3696**
- **Now**
  - Small cluster ratio: **0.9020**
  - Large cluster ratio: **0.0024**
  - One-to-many ratio: **0.1003**
  - Max cluster size: **11**
  - Ambiguous ratio: **0.3992**
- **Key takeaway**
  - Overall shape improved and now meets the small-cluster target.
  - However, **one-to-many remains high**, especially on the MusicBrainz side.

---

## 3) Deep Analysis Highlights

### What is happening with Large Clusters

#### `musicbrainz_lastfm`
- Large clusters are being driven by **generic album names** such as:
  - **“The Punk Singles Collection”**
  - **“Unplugged”**
  - **“Resurrection”**
  - **“Surrender”**
  - **“Happiness”**
- These clusters have:
  - **Very high name agreement**
  - **Very low artist agreement**
  - **Near-zero track-name and duration agreement**
- Example pattern:
  - In the **“Unplugged”** cluster, **name match rate = 0.925**, but:
    - artist match rate = **0.10**
    - track-name match rate = **0.00**
    - duration match rate = **0.00**
- This strongly suggests the matcher is **over-trusting title similarity** for common release names.

#### `discogs_lastfm`
- Large-cluster behavior is **much more controlled now**.
- Remaining problem clusters are mostly **dense duplicate/edition groups**, not huge sprawl.
- Examples show clusters formed around:
  - **Exact track lists**
  - **Exact track positions**
  - Reissue / variant versions of the same release
- This is better than before, but still produces **multiple close candidates for one LastFM record**.

#### `discogs_musicbrainz`
- Large clusters are smaller, but still show **hub-style matching**.
- Unlike the other two, deep attribute detail is sparse, but patterns indicate:
  - Several Discogs records connect to the same MusicBrainz release
  - Reissues / alternate editions are still collapsing together
- The issue is less about massive title-only clusters and more about **release-variant ambiguity**.

---

### What is happening with One-to-Many Clusters

#### `musicbrainz_lastfm`
- **One-to-many ratio = 0.1335** — the worst of the three current files.
- **361 left-side** and **361 right-side** entities have multiple matches.
- Top examples:
  - `lastFM_4206` → **10 matches**
  - `lastFM_15674` → **8 matches**
  - `lastFM_61186` → **8 matches**
  - `mbrainz_3883`, `mbrainz_14360`, `mbrainz_5210`, `mbrainz_6913` → **6 matches each**
- This indicates widespread candidate overlap rather than isolated bad clusters.

#### `discogs_lastfm`
- **One-to-many ratio = 0.0416**
- Much better than before, but still above target.
- Multi-match entities are concentrated:
  - **19 Discogs** entities with multiple matches
  - **28 LastFM** entities with multiple matches
- Top example:
  - `lastFM_59735` → **7 matches**
- These are fewer and tighter than before, suggesting progress but not enough separation among variants.

#### `discogs_musicbrainz`
- **One-to-many ratio = 0.1003**
- The imbalance is notable:
  - **25 Discogs** entities multi-match
  - **248 MusicBrainz** entities multi-match
- This suggests the **MusicBrainz side contains many closely related release variants** that Discogs records are matching against.

---

### Which attributes are contributing to the issues

#### 1. **Name / title**
- The strongest recurring source of false positives.
- Especially problematic for common titles:
  - “Unplugged”
  - “Resurrection”
  - “Surrender”
  - “Happiness”
  - “Trilogy”
  - “The Singles…”
  - “The Punk Singles Collection”
- Potential reason:
  - **Common album names reused across many artists**
  - Matching thresholds likely allow title similarity to dominate when other fields disagree

#### 2. **Artist**
- Contributes in two different ways:
  - Sometimes too weak to stop false matches when titles are generic
  - Sometimes creates **artist-level hubs** when many releases exist for the same artist
- Examples:
  - `discogs_lastfm` improved because title/artist-only broad matching was reduced
  - `discogs_musicbrainz` still shows hubs concentrated on the MusicBrainz side, likely due to many related releases per artist

#### 3. **Track list / track position**
- Highly discriminative when complete and clean.
- In `discogs_lastfm`, exact track-name and track-position agreement helped form tighter and more believable clusters.
- But in `musicbrainz_lastfm`, track-level signals often show **very low agreement**, yet matches still survive.
- Potential reason:
  - Track data may be **missing, malformed, differently formatted, or underweighted**

#### 4. **Duration**
- Often disagrees in bad clusters, but is not consistently strong enough to reject false matches.
- In `musicbrainz_lastfm`, duration is frequently **0 agreement** inside problem clusters.
- In `discogs_lastfm`, duration appears useful but inconsistent due to:
  - missing values
  - zeros
  - formatting differences
  - release-level vs track-level runtime mismatches

#### 5. **Data quality / normalization**
- Several records show signs of encoding and formatting problems.
- Examples include:
  - garbled text / character encoding
  - punctuation variants
  - abbreviated artist names
  - inconsistent title formatting
- These issues can both:
  - reduce true-match confidence
  - increase accidental overlap on generic terms

---

### Which entities are matching with multiple records and likely causes

#### `musicbrainz_lastfm`
- Notable multi-match entities:
  - `lastFM_4206` → 10 matches
  - `lastFM_15674` → 8
  - `lastFM_61186` → 8
  - `mbrainz_3883` → 6
  - `mbrainz_14360` → 6
- Likely cause:
  - Generic titles create broad candidate pools
  - Other attributes are too weak, noisy, or underweighted to break ties

#### `discogs_lastfm`
- Notable multi-match entities:
  - `lastFM_59735` → 7 matches
  - `discogs_155448` → 4
  - several LastFM IDs with 4 matches
- Likely cause:
  - Multiple Discogs variants / editions map to the same LastFM record
  - LastFM appears less granular or less release-specific than Discogs in some cases

#### `discogs_musicbrainz`
- Notable multi-match entities:
  - `mbrainz_10944` → 6 matches
  - `mbrainz_1867`, `mbrainz_23545`, `mbrainz_2387`, `mbrainz_634` → 5 matches
- Likely cause:
  - MusicBrainz contains multiple related release records that remain hard to separate
  - Blocking or scoring likely still groups edition/reissue variants too broadly

---

### Likely underlying causes overall

- **Generic/shared release titles** are overpowering more specific evidence.
- **Blocking is likely too broad** for common album names and major artists.
- **Thresholds are too permissive** for close-scoring alternatives.
- **Track-level evidence is not consistently decisive**, either because:
  - it is underweighted,
  - incomplete,
  - or inconsistently formatted across sources.
- **Release variants / editions / reissues** are being treated as near duplicates instead of distinct candidates.
- **Max update attempts were reached**, so the system could not continue refining configs despite remaining problems.

---

## 4) Conclusion

- The strongest improvement is in **`discogs_lastfm`**, where cluster size and one-to-many behavior improved a lot, though ambiguity is still too high.
- **`discogs_musicbrainz`** improved modestly but still suffers from release-variant ambiguity, especially on the MusicBrainz side.
- **`musicbrainz_lastfm`** is the main concern: it regressed into strong title-driven over-clustering and should be prioritized for matcher/threshold review once config updates are allowed again.