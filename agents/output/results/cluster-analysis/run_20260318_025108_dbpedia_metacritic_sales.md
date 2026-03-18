# Cluster Analysis Summary

## 1) Overall Snapshot

- **Overall status:** All 3 correspondence files are **unhealthy** and **miss target thresholds**.
- **Target thresholds:**  
  - Small cluster ratio: **0.90**  
  - Large cluster ratio: **0.005**  
  - One-to-many ratio: **0.03**
- **Main pattern across all files:** matching is producing **too many ambiguous, many-to-many clusters**, often driven by a few shared attributes such as **developer, platform, releaseYear, criticScore, ESRB**, rather than unique identifiers.
- **Configuration update status:** No new config changes were applied because **max update attempts were already reached** for all pairs.

---

## 2) What Was Wrong (Before) vs What Happened After (Now)

> No previous comparison snapshot was provided, so “Before vs Now” is interpreted as **expected healthy behavior vs current observed behavior**.

### `correspondences_metacritic_dbpedia.csv`

- **Before (healthy expectation):**
  - Mostly small, clean 1:1 or near 1:1 clusters
  - Low ambiguity and limited hub formation
- **Now:**
  - **4,191 clusters**
  - **Max cluster size: 1,268** → extremely large over-grouped cluster
  - **Small cluster ratio: 46.0%** vs target 90%
  - **Large cluster ratio: 21.5%** vs target 0.5%
  - **One-to-many ratio: 64.2%** vs target 3%
  - **Ambiguous ratio: 53.3%**
  - **Max degree: 27**
  - Scores remain fairly high (**mean 0.868**), which suggests the matcher is confidently linking records that are still ambiguous
- **Interpretation:** Severe ambiguity; likely **over-reliance on broad shared attributes** and/or thresholds that are too permissive.

### `correspondences_metacritic_sales.csv`

- **Before (healthy expectation):**
  - Predominantly pairwise matches with minimal transitive expansion
- **Now:**
  - **5,826 clusters**
  - **Max cluster size: 51**
  - **Small cluster ratio: 85.5%** — closest to healthy among the 3 files, but still below target
  - **Large cluster ratio: 3.35%** vs target 0.5%
  - **One-to-many ratio: 21.3%** vs target 3%
  - **Ambiguous ratio: 14.2%**
  - **Max degree: 7**
  - High scores (**mean 0.903**, p90 = **1.0**)
- **Interpretation:** This is the **least problematic file**, but still shows **transitive over-grouping** and hub formation in specific franchises or games with highly similar metadata.

### `correspondences_dbpedia_sales.csv`

- **Before (healthy expectation):**
  - Mostly isolated 1:1 links, few hubs, small clusters
- **Now:**
  - **2,211 clusters**
  - **Max cluster size: 1,097**
  - **Small cluster ratio: 43.6%** vs target 90%
  - **Large cluster ratio: 24.1%** vs target 0.5%
  - **One-to-many ratio: 70.3%** vs target 3%
  - **Ambiguous ratio: 48.1%**
  - **Max degree: 114** → strongest hub problem in the report
  - Lower score quality than the other files (**mean 0.827**)
- **Interpretation:** Very strong **many-to-many ambiguity**, especially on the sales side, with some records matching extremely broadly.

---

## 3) Deep Analysis Highlights

### Large Clusters: What is happening

- The largest problematic clusters are not random; they are concentrated around **game franchises, sequels, remasters, and platform variants**.
- Examples of oversized cluster themes:
  - **Metacritic ↔ DBpedia:** Madden NFL, NBA 2K, Tom Clancy, LEGO, Need for Speed, Assassin’s Creed
  - **Metacritic ↔ Sales:** Dark Souls, Battlefield, LEGO, Pro Evolution Soccer, Madden NFL, F1
  - **DBpedia ↔ Sales:** Madden/NCAA/NBA Live, Street Fighter / Mega Man, LEGO, Dynasty Warriors / Warriors Orochi, Need for Speed
- These clusters are often **sparse but huge**, meaning many records are weakly connected through shared attributes rather than strong direct identity evidence.
- This indicates **transitive closure is amplifying local ambiguity into large connected components**.

### One-to-Many Clusters: What is happening

- **Metacritic ↔ DBpedia:** very high one-to-many behavior (**64.2%**)
- **DBpedia ↔ Sales:** worst one-to-many behavior (**70.3%**)
- **Metacritic ↔ Sales:** lower but still elevated (**21.3%**)

- Common pattern:
  - A single entity matches many candidates because it shares a common **developer**, **platform**, **release year**, or **score profile**
  - Those candidates then connect onward, creating a wider cluster
- Strongest multi-match examples:
  - **Metacritic ↔ DBpedia**
    - Left side: several Metacritic records with **20 matches**
    - Right side: DBpedia records with up to **27 matches**
  - **Metacritic ↔ Sales**
    - Multi-match counts are smaller; max degree **7**
  - **DBpedia ↔ Sales**
    - Right side is especially problematic: one sales record has **114 matches**
    - Multiple sales records have **70+ matches**

### Which attributes are contributing to the issues

#### Most problematic attributes across files

- **Developer**
  - Most common source of false grouping
  - Very high agreement in bad clusters:
    - Capcom
    - Ubisoft / Ubisoft Montreal / Ubisoft Paris
    - Traveller’s Tales
    - EA Tiburon / EA Canada
    - Omega Force
    - Konami
    - From Software
- **Platform**
  - Helps narrow candidates somewhat, but still too broad for franchises with many releases on the same console
- **ReleaseYear**
  - Frequently aligns across sequels, remasters, or same-year platform ports
- **Name**
  - Often too weak due to:
    - sequels with near-identical titles
    - punctuation differences
    - subtitle variants
    - remasters / re-releases
    - franchise naming patterns
- **Numeric review attributes** in sales matching
  - **criticScore**, **userScore**, and **ESRB** are highly consistent across multiple different records
  - These are especially risky because many different titles can share similar score bands and ratings

#### Likely reasons

- **Data quality / normalization issues**
  - Platform variants: `PS3` vs `PlayStation 3`, `Playstation Portable` vs `PSP`, etc.
  - Developer variants: `Ubisoft` vs `Ubisoft Montreal`, `TT Games` vs `Traveller’s Tales`
  - Title formatting differences and subtitle inconsistencies
- **Blocking choices too broad**
  - Blocking on franchise-like attributes or common developers likely groups too many candidates together
- **Matching thresholds too permissive**
  - Scores remain high even when **name agreement is weak**, meaning non-unique attributes are carrying too much weight
- **Weighting imbalance**
  - Broad attributes such as developer, criticScore, ESRB, or platform appear to outweigh stronger identity signals

### Which entities are matching with multiple records, and why

#### Metacritic ↔ DBpedia

- Broad-match entities are concentrated in:
  - **Madden NFL**
  - **NBA 2K**
  - **Tom Clancy**
  - **LEGO**
  - **Need for Speed**
  - **Assassin’s Creed**
- Underlying cause:
  - Records share the same **developer** and often the same **platform family**
  - **Name match rates are often low**, yet scores remain high
  - This suggests franchise-level similarity is being mistaken for record identity

#### Metacritic ↔ Sales

- Multi-match entities are smaller in scale, but still occur around:
  - **Dark Souls**
  - **Battlefield**
  - **LEGO**
  - **Pro Evolution Soccer**
  - **Madden NFL**
  - **F1**
- Underlying cause:
  - Strong overlap in **criticScore, userScore, ESRB, developer, releaseYear**
  - These attributes are not unique enough across editions/platforms
  - Transitive grouping then merges related but distinct records

#### DBpedia ↔ Sales

- This file has the strongest hub behavior, especially on the **sales side**
- Problem clusters center on:
  - **Madden / NCAA / NBA Live**
  - **Street Fighter / Mega Man**
  - **LEGO**
  - **Dynasty Warriors / Warriors Orochi**
  - **Need for Speed**
- Underlying cause:
  - Sales records appear to act as broad hubs when they share common franchise/developer metadata
  - **Developer is often nearly universal inside bad clusters**, while **name and release year remain weak or only partially matching**
  - This is a strong sign that **developer is over-weighted** and sales records are insufficiently discriminated

---

## 4) Conclusion

- The main issue is **over-matching driven by non-unique shared attributes**, especially **developer**, with support from **platform**, **releaseYear**, and in sales matching **criticScore / ESRB / userScore**.  
- `correspondences_metacritic_sales.csv` is comparatively better, but all three files still show excessive ambiguity and hub formation.  
- The matching setup likely needs **tighter thresholds, reduced weight on broad attributes, and stronger blocking/disambiguation on title + edition/platform-specific signals**.