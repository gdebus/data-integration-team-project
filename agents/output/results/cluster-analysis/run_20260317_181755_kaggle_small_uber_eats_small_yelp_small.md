# Cluster Analysis Summary

## 1) Overall Snapshot

- **2 of 3 correspondence files are healthy and stable**:
  - `correspondences_kaggle_small_uber_eats_small.csv`
  - `correspondences_kaggle_small_yelp_small.csv`
- **1 file is unhealthy**:
  - `correspondences_uber_eats_small_yelp_small.csv`
- Overall issue is concentrated in the **Uber Eats ↔ Yelp** matching, where the cluster shape is no longer long-tail and now shows:
  - larger clusters,
  - more one-to-many relationships,
  - high ambiguity across close-scoring candidates.
- Target thresholds were missed mainly on:
  - **small cluster ratio**
  - **large cluster ratio**
  - **one-to-many ratio**

---

## 2) What Was Wrong (Before) vs What Happened After (Now)

### `correspondences_kaggle_small_uber_eats_small.csv`
- **Before**
  - Healthy long-tail distribution
  - 58 clusters
  - Max cluster size: **2**
  - Small cluster ratio: **1.00**
  - Large cluster ratio: **0.00**
  - One-to-many ratio: **0.00**
- **Now**
  - **No change**
  - Still fully healthy
- **Key takeaway**
  - Matching remains clean and strictly one-to-one.

### `correspondences_kaggle_small_yelp_small.csv`
- **Before**
  - Healthy long-tail distribution
  - 151 clusters
  - Max cluster size: **2**
  - Small cluster ratio: **1.00**
  - Large cluster ratio: **0.00**
  - One-to-many ratio: **0.00**
- **Now**
  - **No change**
  - Still fully healthy
- **Key takeaway**
  - Matching remains stable with no ambiguity or cluster inflation.

### `correspondences_uber_eats_small_yelp_small.csv`
- **Before**
  - Some ambiguity already existed, but pattern was still relatively contained
  - 40 clusters
  - Max cluster size: **3**
  - Small cluster ratio: **0.925**
  - Large cluster ratio: **0.00**
  - One-to-many ratio: **0.0488**
  - Max degree: **2**
  - Ambiguous ratio: **0.50**
- **Now**
  - Situation **worsened significantly**
  - 50 clusters
  - Max cluster size: **11**
  - Small cluster ratio: **0.70**
  - Large cluster ratio: **0.06**
  - One-to-many ratio: **0.30**
  - Max degree: **4**
  - Ambiguous ratio: **0.5714**
- **Change**
  - Total clusters: **+10**
  - Max cluster size: **+8**
  - Small cluster ratio: **-0.225**
  - Large cluster ratio: **+0.06**
  - One-to-many ratio: **+0.2512**
- **Key takeaway**
  - The updated matching behavior created substantially more ambiguous and many-to-many groupings instead of resolving them.

---

## 3) Deep Analysis Highlights

### Large Clusters
- Large clusters appear **only** in `correspondences_uber_eats_small_yelp_small.csv`.
- The largest problematic clusters are:
  - **Cluster of 11 nodes**: 6 Uber Eats + 5 Yelp
  - **Cluster of 9 nodes**: 5 Uber Eats + 4 Yelp
  - **Cluster of 8 nodes**: 3 Uber Eats + 5 Yelp
- These are not random; they are **dense bipartite groups** where many records on one side connect to many records on the other.
- This strongly suggests the matcher is grouping businesses around **shared location-level attributes** rather than isolating unique entity identity.

### One-to-Many Clusters
- Uber Eats ↔ Yelp now has:
  - **one-to-many ratio = 0.30**
  - **max degree = 4**
- Multi-match entities increased sharply:
  - **Left/Uber Eats:** 17 entities with multiple matches
  - **Right/Yelp:** 21 entities with multiple matches
- Most exposed records include:
  - Uber Eats: `uber_eats-08870`, `uber_eats-08319`, `uber_eats-08499`, `uber_eats-08123` — each matching **4** records
  - Yelp: `yelp-01389`, `yelp-01268`, `yelp-01395` — each matching **4** records
- This is a clear sign of **entity ambiguity**, not isolated noise.

### Which Attributes Are Driving the Issues
- The problematic clusters show near-perfect agreement on broad location attributes such as:
  - **street**
  - **city**
  - **state**
  - **postal_code**
  - **country**
- In some clusters, **latitude** also aligns very closely.
- Example patterns:
  - `NW Market St`, Seattle, WA, 98107
  - `Fremont Ave N`, Seattle, WA, 98103
  - `Broadway E`, Seattle, WA, 98102
- These attributes are helping create matches, but they are **not specific enough** to distinguish co-located or nearby businesses.

### Potential Reasons
- **Over-reliance on shared address/location features**
  - Records on the same street or in the same postal code are being pulled together too aggressively.
- **Blocking may be too broad**
  - If candidates are generated mainly from common street/postal patterns, many nearby businesses enter the same comparison pool.
- **Thresholds may be too permissive**
  - Scores in problematic clusters often sit in the **mid-0.72 to mid-0.79 range**, which suggests borderline candidates are still being accepted.
- **Insufficient discrimination from unique attributes**
  - Name and exact address components may not be strong enough to separate businesses sharing the same corridor/building area.
- **Possible data quality / normalization issues**
  - Similar street-only normalization may collapse distinct addresses into the same representation.
  - If suite/unit or finer-grained address detail is missing, different businesses can appear artificially similar.

### Which Entities Are Matching with Multiple Records and Why
- **NW Market St cluster**
  - Uber Eats hubs: `uber_eats-08870`, `uber_eats-08780`, `uber_eats-08972`
  - Yelp hubs: `yelp-01268`, `yelp-01262`, `yelp-01249`
  - Cause:
    - Perfect overlap on street/city/state/postal/country
    - Shared corridor-level address signals are dominating identity resolution
- **Fremont Ave N cluster**
  - Uber Eats hubs: `uber_eats-08399`, `uber_eats-08410`, `uber_eats-07897`
  - Yelp hubs: `yelp-01395`, `yelp-01223`, `yelp-01286`
  - Cause:
    - Strong agreement on geography and even latitude
    - Nearby or co-located businesses likely remain insufficiently separated
- **Broadway E cluster**
  - Uber Eats hubs: `uber_eats-08123`, `uber_eats-08319`
  - Yelp hubs: `yelp-04558`, `yelp-04612`, `yelp-04527`, `yelp-04562`
  - Cause:
    - Extremely ambiguous cluster with **ambiguous ratio = 1.0**
    - Very similar scores and identical street/postal geography indicate overly broad candidate grouping or weak tie-breaking logic

### Score Behavior
- Healthy files have clean one-to-one structures even with strong scores.
- In the unhealthy Uber Eats ↔ Yelp file:
  - overall mean score dropped from **0.8291** to **0.801**
  - median dropped from **0.8545** to **0.7703**
- This indicates the matcher is now admitting **weaker and less separable candidates**, which aligns with the growth in ambiguous clusters.

---

## 4) Conclusion

- The overall system is mostly healthy, but the **Uber Eats ↔ Yelp** correspondence degraded materially after the matching update.  
- The main failure mode is **many-to-many clustering driven by generic shared location attributes**, especially street-level agreement in dense urban areas.  
- Focus next on tightening candidate selection and match acceptance for Uber Eats ↔ Yelp, with stronger emphasis on exact entity-level discriminators over shared geography.