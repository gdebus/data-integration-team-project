# Pipeline Snapshots

notebook_name=Agent III & IV
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=29.41%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
)

import pandas as pd
import os


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

dbpedia = load_csv(
    DATA_DIR + "dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "sales.csv",
    name="sales",
)

datasets = [dbpedia, metacritic, sales]


# --------------------------------
# Schema already matched in provided input files
# Resulting columns are aligned sufficiently for integration
# --------------------------------

print("Schema already aligned")


# --------------------------------
# Perform Blocking
# MUST use the precomputed blocking configuration
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_dbpedia_sales = EmbeddingBlocker(
    dbpedia,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_dbpedia = EmbeddingBlocker(
    metacritic,
    dbpedia,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_sales = EmbeddingBlocker(
    metacritic,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed optimal settings
# preprocess mapping: lower_strip -> lambda x: str(x).lower().strip()
# --------------------------------

preprocess_lower_strip = lambda x: str(x).lower().strip()

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

comparators_metacritic_dbpedia = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
]

comparators_metacritic_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
    ),
    NumericComparator(
        column="userScore",
        max_difference=1.0,
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.45, 0.2, 0.2, 0.15],
    threshold=0.8,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.35, 0.2, 0.15, 0.12, 0.08, 0.1],
    threshold=0.78,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_dbpedia_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"),
    index=False,
)
rb_correspondences_metacritic_dbpedia.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_dbpedia.csv"),
    index=False,
)
rb_correspondences_metacritic_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_dbpedia_sales,
        rb_correspondences_metacritic_dbpedia,
        rb_correspondences_metacritic_sales,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", longest_string)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("criticScore", maximum)
strategy.add_attribute_fuser("userScore", maximum)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("globalSales", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_data = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_data.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=12
node_name=execute_pipeline
accuracy_score=83.19%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
)

import pandas as pd
import os
import re
import math
from collections import Counter, defaultdict


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

dbpedia = load_csv(
    DATA_DIR + "dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "sales.csv",
    name="sales",
)

datasets = [dbpedia, metacritic, sales]


# --------------------------------
# Schema already matched in provided input files
# Resulting columns are aligned sufficiently for integration
# --------------------------------

print("Schema already aligned")


# --------------------------------
# Normalization helpers
# Improve matching robustness and fusion quality
# --------------------------------

def is_nullish(x):
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "nat"}


def clean_text(x):
    if is_nullish(x):
        return None
    return str(x).strip()


def normalize_whitespace(x):
    if is_nullish(x):
        return None
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_name(x):
    if is_nullish(x):
        return None
    s = str(x)
    s = re.sub(r"\([^)]*video game[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\([^)]*computer game[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\([^)]*\bgame\b[^)]*\)", "", s, flags=re.IGNORECASE)
    s = normalize_whitespace(s)
    return s


def normalize_name_for_compare(x):
    s = normalize_name(x)
    if s is None:
        return None
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_year_value(x):
    if is_nullish(x):
        return None
    s = str(x)
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return m.group(0)
    return normalize_whitespace(s)


def normalize_developer(x):
    if is_nullish(x):
        return None
    s = normalize_whitespace(x)
    return s


def normalize_publisher(x):
    if is_nullish(x):
        return None
    s = normalize_whitespace(x)
    s_low = s.lower()

    publisher_map = {
        "scea": "Sony Computer Entertainment",
        "sony computer entertainment america": "Sony Computer Entertainment",
        "sony computer entertainment europe": "Sony Computer Entertainment",
        "sony computer entertainment": "Sony Computer Entertainment",
        "scee": "Sony Computer Entertainment",
        "sce": "Sony Computer Entertainment",
        "ms game studios": "Microsoft Game Studios",
    }
    return publisher_map.get(s_low, s)


def normalize_platform(x):
    if is_nullish(x):
        return None
    s = normalize_whitespace(x)
    s_low = s.lower()

    platform_map = {
        "playstation 3": "PS3",
        "ps3": "PS3",
        "playstation portable": "PSP",
        "psp": "PSP",
        "playstation 4": "PS4",
        "ps4": "PS4",
        "playstation 5": "PS5",
        "ps5": "PS5",
        "playstation 2": "PS2",
        "ps2": "PS2",
        "playstation vita": "PS Vita",
        "ps vita": "PS Vita",
        "xbox 360": "Xbox 360",
        "xbox one": "Xbox One",
        "xbox series x": "Xbox Series X",
        "xbox series x and series s": "Xbox Series X",
        "xbox series s": "Xbox Series X",
        "nintendo ds": "DS",
        "ds": "DS",
        "nintendo 3ds": "3DS",
        "3ds": "3DS",
        "wii u": "Wii U",
        "wii": "Wii",
        "game boy color": "Game Boy Color",
        "gamecube": "GameCube",
        "nintendo gamecube": "GameCube",
        "pc": "PC",
        "windows": "PC",
        "microsoft windows": "PC",
        "android": "Android",
        "android (operating system)": "Android",
        "ios": "iOS",
    }
    return platform_map.get(s_low, s)


def normalize_esrb(x):
    if is_nullish(x):
        return None
    return normalize_whitespace(x).upper()


def normalize_numeric(x):
    if is_nullish(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def preprocess_lower_strip(x):
    if is_nullish(x):
        return ""
    return str(x).lower().strip()


for df in [dbpedia, metacritic, sales]:
    if "name" in df.columns:
        df["name"] = df["name"].apply(normalize_name)
    if "platform" in df.columns:
        df["platform"] = df["platform"].apply(normalize_platform)
    if "developer" in df.columns:
        df["developer"] = df["developer"].apply(normalize_developer)
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(normalize_publisher)
    if "releaseYear" in df.columns:
        df["releaseYear"] = df["releaseYear"].apply(extract_year_value)
    if "ESRB" in df.columns:
        df["ESRB"] = df["ESRB"].apply(normalize_esrb)
    if "criticScore" in df.columns:
        df["criticScore"] = df["criticScore"].apply(normalize_numeric)
    if "userScore" in df.columns:
        df["userScore"] = df["userScore"].apply(normalize_numeric)


# --------------------------------
# Perform Blocking
# MUST use the precomputed blocking configuration
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_dbpedia_sales = EmbeddingBlocker(
    dbpedia,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_dbpedia = EmbeddingBlocker(
    metacritic,
    dbpedia,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_sales = EmbeddingBlocker(
    metacritic,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed optimal settings
# preprocess mapping: lower_strip -> lambda x: str(x).lower().strip()
# --------------------------------

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

comparators_metacritic_dbpedia = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
]

comparators_metacritic_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
    ),
    NumericComparator(
        column="userScore",
        max_difference=1.0,
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.45, 0.2, 0.2, 0.15],
    threshold=0.8,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.35, 0.2, 0.15, 0.12, 0.08, 0.1],
    threshold=0.78,
    id_column="id",
)


# --------------------------------
# Correspondence post-processing to reduce cluster contamination
# Keep more conservative edges and enforce pairwise consistency
# --------------------------------

def detect_score_column(df):
    for col in ["similarity", "score", "confidence", "match_score"]:
        if col in df.columns:
            return col
    return None


score_col_dbpedia_sales = detect_score_column(rb_correspondences_dbpedia_sales)
score_col_metacritic_dbpedia = detect_score_column(rb_correspondences_metacritic_dbpedia)
score_col_metacritic_sales = detect_score_column(rb_correspondences_metacritic_sales)


dbpedia_by_id = dbpedia.set_index("id").to_dict("index")
metacritic_by_id = metacritic.set_index("id").to_dict("index")
sales_by_id = sales.set_index("id").to_dict("index")


def conservative_filter(corr_df, left_lookup, right_lookup, left_name, right_name, score_col=None):
    if corr_df is None or len(corr_df) == 0:
        return corr_df

    left_id_col = None
    right_id_col = None
    for c in corr_df.columns:
        cl = c.lower()
        if left_id_col is None and ("left" in cl and "id" in cl):
            left_id_col = c
        if right_id_col is None and ("right" in cl and "id" in cl):
            right_id_col = c

    if left_id_col is None or right_id_col is None:
        cols = list(corr_df.columns)
        id_like = [c for c in cols if "id" in c.lower()]
        if len(id_like) >= 2:
            left_id_col, right_id_col = id_like[0], id_like[1]
        else:
            return corr_df

    filtered_rows = []

    for _, row in corr_df.iterrows():
        lid = row[left_id_col]
        rid = row[right_id_col]

        lrec = left_lookup.get(lid, {})
        rrec = right_lookup.get(rid, {})

        lname = normalize_name_for_compare(lrec.get("name"))
        rname = normalize_name_for_compare(rrec.get("name"))
        lplat = normalize_platform(lrec.get("platform"))
        rplat = normalize_platform(rrec.get("platform"))
        lyr = extract_year_value(lrec.get("releaseYear"))
        ryr = extract_year_value(rrec.get("releaseYear"))

        keep = True

        if lname and rname and lname != rname:
            if lname not in rname and rname not in lname:
                keep = False

        if keep and lplat and rplat and lplat != rplat:
            keep = False

        if keep and lyr and ryr and lyr != ryr:
            try:
                if abs(int(lyr) - int(ryr)) > 1:
                    keep = False
            except Exception:
                pass

        if keep:
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows, columns=corr_df.columns)

    if score_col and score_col in filtered_df.columns and len(filtered_df) > 0:
        filtered_df = filtered_df.sort_values(score_col, ascending=False)

        best_rows = []
        seen_left = set()
        seen_right = set()
        for _, row in filtered_df.iterrows():
            lid = row[left_id_col]
            rid = row[right_id_col]
            if lid in seen_left or rid in seen_right:
                continue
            seen_left.add(lid)
            seen_right.add(rid)
            best_rows.append(row)

        filtered_df = pd.DataFrame(best_rows, columns=filtered_df.columns)

    return filtered_df.reset_index(drop=True)


rb_correspondences_dbpedia_sales = conservative_filter(
    rb_correspondences_dbpedia_sales,
    dbpedia_by_id,
    sales_by_id,
    "dbpedia",
    "sales",
    score_col_dbpedia_sales,
)

rb_correspondences_metacritic_dbpedia = conservative_filter(
    rb_correspondences_metacritic_dbpedia,
    metacritic_by_id,
    dbpedia_by_id,
    "metacritic",
    "dbpedia",
    score_col_metacritic_dbpedia,
)

rb_correspondences_metacritic_sales = conservative_filter(
    rb_correspondences_metacritic_sales,
    metacritic_by_id,
    sales_by_id,
    "metacritic",
    "sales",
    score_col_metacritic_sales,
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_dbpedia_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"),
    index=False,
)
rb_correspondences_metacritic_dbpedia.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_dbpedia.csv"),
    index=False,
)
rb_correspondences_metacritic_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_dbpedia_sales,
        rb_correspondences_metacritic_dbpedia,
        rb_correspondences_metacritic_sales,
    ],
    ignore_index=True,
)


# --------------------------------
# Source-aware fusion functions
# Prefer metacritic/sales for core attributes; use consensus with normalization
# --------------------------------

SOURCE_PRIORITY = {
    "name": ["metacritic", "sales", "dbpedia"],
    "platform": ["metacritic", "sales", "dbpedia"],
    "releaseYear": ["metacritic", "sales", "dbpedia"],
    "developer": ["metacritic", "sales", "dbpedia"],
    "publisher": ["sales", "metacritic", "dbpedia"],
    "criticScore": ["metacritic", "sales", "dbpedia"],
    "userScore": ["metacritic", "sales", "dbpedia"],
    "ESRB": ["sales", "metacritic", "dbpedia"],
    "series": ["dbpedia", "metacritic", "sales"],
    "globalSales": ["sales", "metacritic", "dbpedia"],
}

def source_from_id(record_id):
    if is_nullish(record_id):
        return None
    s = str(record_id)
    if "_" in s:
        return s.split("_", 1)[0]
    return None


def collect_values(values):
    collected = []
    if values is None:
        return collected
    for v in values:
        if isinstance(v, dict):
            val = v.get("value")
            rid = v.get("id") or v.get("record_id") or v.get("source_id")
            src = v.get("source") or source_from_id(rid)
            if not is_nullish(val):
                collected.append({"value": val, "source": src, "id": rid})
        else:
            if not is_nullish(v):
                collected.append({"value": v, "source": None, "id": None})
    return collected


def prefer_priority_value(values, normalizer=lambda x: x, attribute=None):
    vals = collect_values(values)
    if not vals:
        return None

    buckets = defaultdict(list)
    for item in vals:
        norm = normalizer(item["value"])
        if is_nullish(norm):
            continue
        buckets[str(norm)].append(item)

    if not buckets:
        return None

    priority = SOURCE_PRIORITY.get(attribute, ["metacritic", "sales", "dbpedia"])

    ranked = []
    for norm_val, items in buckets.items():
        sources = [x["source"] for x in items]
        support = len(items)
        best_pri = min([priority.index(s) for s in sources if s in priority], default=len(priority))
        ranked.append((norm_val, items, support, best_pri))

    ranked.sort(key=lambda x: (-x[2], x[3]))

    best_norm, best_items, _, _ = ranked[0]

    best_items_sorted = sorted(
        best_items,
        key=lambda x: priority.index(x["source"]) if x["source"] in priority else len(priority)
    )
    return best_items_sorted[0]["value"]


def fuse_name(values):
    return prefer_priority_value(values, normalizer=normalize_name, attribute="name")


def fuse_platform(values):
    chosen = prefer_priority_value(values, normalizer=normalize_platform, attribute="platform")
    return normalize_platform(chosen) if chosen is not None else None


def fuse_release_year(values):
    chosen = prefer_priority_value(values, normalizer=extract_year_value, attribute="releaseYear")
    return extract_year_value(chosen) if chosen is not None else None


def fuse_developer(values):
    return prefer_priority_value(values, normalizer=normalize_developer, attribute="developer")


def fuse_publisher(values):
    return prefer_priority_value(values, normalizer=normalize_publisher, attribute="publisher")


def fuse_esrb(values):
    chosen = prefer_priority_value(values, normalizer=normalize_esrb, attribute="ESRB")
    return normalize_esrb(chosen) if chosen is not None else None


def fuse_series(values):
    vals = collect_values(values)
    if not vals:
        return None
    dbpedia_vals = [x["value"] for x in vals if x["source"] == "dbpedia" and not is_nullish(x["value"])]
    if dbpedia_vals:
        return max(dbpedia_vals, key=lambda x: len(str(x)))
    non_null = [x["value"] for x in vals if not is_nullish(x["value"])]
    if not non_null:
        return None
    return max(non_null, key=lambda x: len(str(x)))


def fuse_numeric_priority(values, attribute):
    vals = collect_values(values)
    if not vals:
        return None

    priority = SOURCE_PRIORITY.get(attribute, ["metacritic", "sales", "dbpedia"])

    normalized = []
    for item in vals:
        num = normalize_numeric(item["value"])
        if num is not None:
            normalized.append({"value": num, "source": item["source"]})

    if not normalized:
        return None

    counts = Counter([round(x["value"], 3) for x in normalized])
    most_common_count = counts.most_common(1)[0][1]
    common_vals = {k for k, v in counts.items() if v == most_common_count}

    common_items = [x for x in normalized if round(x["value"], 3) in common_vals]
    common_items.sort(
        key=lambda x: priority.index(x["source"]) if x["source"] in priority else len(priority)
    )
    return common_items[0]["value"]


def fuse_critic_score(values):
    return fuse_numeric_priority(values, "criticScore")


def fuse_user_score(values):
    return fuse_numeric_priority(values, "userScore")


def fuse_global_sales(values):
    vals = collect_values(values)
    nums = [normalize_numeric(x["value"]) for x in vals if normalize_numeric(x["value"]) is not None]
    if not nums:
        return None
    return max(nums)


strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", fuse_name)
strategy.add_attribute_fuser("releaseYear", fuse_release_year)
strategy.add_attribute_fuser("developer", fuse_developer)
strategy.add_attribute_fuser("platform", fuse_platform)
strategy.add_attribute_fuser("series", fuse_series)
strategy.add_attribute_fuser("publisher", fuse_publisher)
strategy.add_attribute_fuser("criticScore", fuse_critic_score)
strategy.add_attribute_fuser("userScore", fuse_user_score)
strategy.add_attribute_fuser("ESRB", fuse_esrb)
strategy.add_attribute_fuser("globalSales", fuse_global_sales)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_data = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

if "name" in rb_fused_data.columns:
    rb_fused_data["name"] = rb_fused_data["name"].apply(normalize_name)
if "platform" in rb_fused_data.columns:
    rb_fused_data["platform"] = rb_fused_data["platform"].apply(normalize_platform)
if "releaseYear" in rb_fused_data.columns:
    rb_fused_data["releaseYear"] = rb_fused_data["releaseYear"].apply(extract_year_value)
if "developer" in rb_fused_data.columns:
    rb_fused_data["developer"] = rb_fused_data["developer"].apply(normalize_developer)
if "publisher" in rb_fused_data.columns:
    rb_fused_data["publisher"] = rb_fused_data["publisher"].apply(normalize_publisher)
if "ESRB" in rb_fused_data.columns:
    rb_fused_data["ESRB"] = rb_fused_data["ESRB"].apply(normalize_esrb)

rb_fused_data.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=20
node_name=execute_pipeline
accuracy_score=84.87%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
)

import pandas as pd
import os
import re
from collections import defaultdict


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

dbpedia = load_csv(
    DATA_DIR + "dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "sales.csv",
    name="sales",
)

datasets = [dbpedia, metacritic, sales]


# --------------------------------
# Schema already matched in provided input files
# --------------------------------

print("Schema already aligned")


# --------------------------------
# Normalization helpers
# --------------------------------

def is_nullish(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "nat"}


def normalize_whitespace(x):
    if is_nullish(x):
        return None
    return re.sub(r"\s+", " ", str(x)).strip()


def clean_company_suffixes(s):
    s = re.sub(
        r"\b(incorporated|inc|ltd|llc|co|corp|corporation|gmbh|sa|plc)\b\.?",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s+", " ", s).strip(" ,.-")
    return s


def normalize_name(x):
    if is_nullish(x):
        return None
    s = str(x)
    s = re.sub(r"\([^)]*video game[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\([^)]*computer game[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\([^)]*\bgame\b[^)]*\)", "", s, flags=re.IGNORECASE)
    s = normalize_whitespace(s)
    return s


def normalize_name_for_compare(x):
    s = normalize_name(x)
    if s is None:
        return None
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_year_value(x):
    if is_nullish(x):
        return None
    s = str(x)
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return m.group(0)
    return normalize_whitespace(s)


def canonical_org_text(x):
    if is_nullish(x):
        return None
    s = normalize_whitespace(x)
    s = s.replace("&", " and ")
    s = re.sub(r"[']", "", s)
    s = re.sub(r"[^A-Za-z0-9\s]+", " ", s)
    s = clean_company_suffixes(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalize_developer(x):
    if is_nullish(x):
        return None
    s = normalize_whitespace(x)
    key = canonical_org_text(s)

    developer_map = {
        "tt fusion": "Traveller's Tales",
        "travellers tales": "Traveller's Tales",
        "traveller s tales": "Traveller's Tales",
        "ubisoft paris": "Ubisoft",
        "ubisoft montreal": "Ubisoft",
        "ubisoft pune": "Ubisoft",
        "ubisoft reflections": "Ubisoft",
        "ubisoft shanghai": "Ubisoft",
        "nintendo entertainment analysis and development": "Nintendo",
        "nintendo ead": "Nintendo",
        "rockstar san diego": "Rockstar Games",
        "rockstar north": "Rockstar Games",
        "rockstar leeds": "Rockstar Games",
        "rockstar games": "Rockstar Games",
        "electronic arts canada": "Electronic Arts",
        "ea canada": "Electronic Arts",
        "ea tiburon": "Electronic Arts",
        "sony computer entertainment": "Sony Computer Entertainment",
        "scea": "Sony Computer Entertainment",
        "scee": "Sony Computer Entertainment",
        "sce": "Sony Computer Entertainment",
    }

    return developer_map.get(key, s)


def normalize_publisher(x):
    if is_nullish(x):
        return None
    s = normalize_whitespace(x)
    key = canonical_org_text(s)

    publisher_map = {
        "scea": "Sony Computer Entertainment",
        "scee": "Sony Computer Entertainment",
        "sce": "Sony Computer Entertainment",
        "sony computer entertainment america": "Sony Computer Entertainment",
        "sony computer entertainment europe": "Sony Computer Entertainment",
        "sony computer entertainment": "Sony Computer Entertainment",
        "ms game studios": "Microsoft Game Studios",
        "microsoft game studios": "Microsoft Game Studios",
        "microsoft studios": "Microsoft Game Studios",
        "square enix europe": "Square Enix",
        "square enix ltd": "Square Enix",
        "electronic arts": "Electronic Arts",
        "ea": "Electronic Arts",
        "konami digital entertainment": "Konami",
        "ubisoft entertainment": "Ubisoft",
        "namco bandai games": "Bandai Namco",
        "bandai namco games": "Bandai Namco",
    }

    return publisher_map.get(key, s)


def normalize_platform(x):
    if is_nullish(x):
        return None
    s = normalize_whitespace(x)
    s_low = s.lower()

    platform_map = {
        "playstation 3": "PS3",
        "ps3": "PS3",
        "playstation portable": "PSP",
        "psp": "PSP",
        "playstation 4": "PS4",
        "ps4": "PS4",
        "playstation 5": "PS5",
        "ps5": "PS5",
        "playstation 2": "PS2",
        "ps2": "PS2",
        "playstation vita": "PS Vita",
        "ps vita": "PS Vita",
        "xbox 360": "Xbox 360",
        "xbox one": "Xbox One",
        "xbox series x": "Xbox Series X",
        "xbox series x and series s": "Xbox Series X",
        "xbox series s": "Xbox Series X",
        "nintendo ds": "DS",
        "ds": "DS",
        "nintendo 3ds": "3DS",
        "3ds": "3DS",
        "wii u": "Wii U",
        "wii": "Wii",
        "game boy color": "Game Boy Color",
        "game boy advance": "Game Boy Advance",
        "gba": "Game Boy Advance",
        "gamecube": "GameCube",
        "nintendo gamecube": "GameCube",
        "pc": "PC",
        "windows": "PC",
        "microsoft windows": "PC",
        "android": "Android",
        "android operating system": "Android",
        "ios": "iOS",
    }
    return platform_map.get(s_low, s)


def normalize_esrb(x):
    if is_nullish(x):
        return None
    s = normalize_whitespace(x).upper().replace(" ", "")
    esrb_map = {
        "M17+": "M",
        "MATURE17+": "M",
        "MATURE": "M",
        "AO18+": "AO",
        "ADULTSONLY18+": "AO",
        "EVERYONE10+": "E10+",
        "E10": "E10+",
        "E10PLUS": "E10+",
        "TEEN": "T",
        "EVERYONE": "E",
    }
    return esrb_map.get(s, s)


def normalize_numeric(x):
    if is_nullish(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def preprocess_lower_strip(x):
    if is_nullish(x):
        return ""
    return str(x).lower().strip()


for df in [dbpedia, metacritic, sales]:
    if "name" in df.columns:
        df["name"] = df["name"].apply(normalize_name)
    if "platform" in df.columns:
        df["platform"] = df["platform"].apply(normalize_platform)
    if "developer" in df.columns:
        df["developer"] = df["developer"].apply(normalize_developer)
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(normalize_publisher)
    if "releaseYear" in df.columns:
        df["releaseYear"] = df["releaseYear"].apply(extract_year_value)
    if "ESRB" in df.columns:
        df["ESRB"] = df["ESRB"].apply(normalize_esrb)
    if "criticScore" in df.columns:
        df["criticScore"] = df["criticScore"].apply(normalize_numeric)
    if "userScore" in df.columns:
        df["userScore"] = df["userScore"].apply(normalize_numeric)


# --------------------------------
# Perform Blocking
# MUST use the precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_dbpedia_sales = EmbeddingBlocker(
    dbpedia,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_dbpedia = EmbeddingBlocker(
    metacritic,
    dbpedia,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_sales = EmbeddingBlocker(
    metacritic,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed optimal settings
# --------------------------------

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

comparators_metacritic_dbpedia = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
]

comparators_metacritic_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
    ),
    NumericComparator(
        column="userScore",
        max_difference=1.0,
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.45, 0.2, 0.2, 0.15],
    threshold=0.8,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.35, 0.2, 0.15, 0.12, 0.08, 0.1],
    threshold=0.78,
    id_column="id",
)


# --------------------------------
# Correspondence post-processing
# More precise pairwise filtering to reduce contaminated clusters
# --------------------------------

def detect_score_column(df):
    for col in ["similarity", "score", "confidence", "match_score"]:
        if col in df.columns:
            return col
    return None


score_col_dbpedia_sales = detect_score_column(rb_correspondences_dbpedia_sales)
score_col_metacritic_dbpedia = detect_score_column(rb_correspondences_metacritic_dbpedia)
score_col_metacritic_sales = detect_score_column(rb_correspondences_metacritic_sales)

dbpedia_by_id = dbpedia.set_index("id").to_dict("index")
metacritic_by_id = metacritic.set_index("id").to_dict("index")
sales_by_id = sales.set_index("id").to_dict("index")


def get_id_cols(corr_df):
    left_id_col = None
    right_id_col = None
    for c in corr_df.columns:
        cl = c.lower()
        if left_id_col is None and ("left" in cl and "id" in cl):
            left_id_col = c
        if right_id_col is None and ("right" in cl and "id" in cl):
            right_id_col = c
    if left_id_col is None or right_id_col is None:
        id_like = [c for c in corr_df.columns if "id" in c.lower()]
        if len(id_like) >= 2:
            left_id_col, right_id_col = id_like[0], id_like[1]
    return left_id_col, right_id_col


def conservative_filter(corr_df, left_lookup, right_lookup, score_col=None):
    if corr_df is None or len(corr_df) == 0:
        return corr_df

    left_id_col, right_id_col = get_id_cols(corr_df)
    if left_id_col is None or right_id_col is None:
        return corr_df

    filtered_rows = []

    for _, row in corr_df.iterrows():
        lid = row[left_id_col]
        rid = row[right_id_col]
        lrec = left_lookup.get(lid, {})
        rrec = right_lookup.get(rid, {})

        lname = normalize_name_for_compare(lrec.get("name"))
        rname = normalize_name_for_compare(rrec.get("name"))
        lplat = normalize_platform(lrec.get("platform"))
        rplat = normalize_platform(rrec.get("platform"))
        lyr = extract_year_value(lrec.get("releaseYear"))
        ryr = extract_year_value(rrec.get("releaseYear"))
        ldev = canonical_org_text(lrec.get("developer"))
        rdev = canonical_org_text(rrec.get("developer"))
        lcritic = normalize_numeric(lrec.get("criticScore"))
        rcritic = normalize_numeric(rrec.get("criticScore"))
        luser = normalize_numeric(lrec.get("userScore"))
        ruser = normalize_numeric(rrec.get("userScore"))

        keep = True

        if lname and rname and lname != rname:
            if lname not in rname and rname not in lname:
                keep = False

        if keep and lplat and rplat and lplat != rplat:
            keep = False

        if keep and lyr and ryr:
            try:
                if abs(int(lyr) - int(ryr)) > 1:
                    keep = False
            except Exception:
                pass

        if keep and ldev and rdev:
            if ldev != rdev and ldev not in rdev and rdev not in ldev:
                if not (lname and rname and lname == rname and lplat and rplat and lplat == rplat and lyr and ryr and lyr == ryr):
                    keep = False

        if keep and lcritic is not None and rcritic is not None:
            if abs(lcritic - rcritic) > 15:
                keep = False

        if keep and luser is not None and ruser is not None:
            if abs(luser - ruser) > 2.0:
                keep = False

        if keep:
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows, columns=corr_df.columns)

    if score_col and score_col in filtered_df.columns and len(filtered_df) > 0:
        filtered_df = filtered_df.sort_values(score_col, ascending=False)
        best_rows = []
        seen_left = set()
        seen_right = set()
        for _, row in filtered_df.iterrows():
            lid = row[left_id_col]
            rid = row[right_id_col]
            if lid in seen_left or rid in seen_right:
                continue
            seen_left.add(lid)
            seen_right.add(rid)
            best_rows.append(row)
        filtered_df = pd.DataFrame(best_rows, columns=filtered_df.columns)

    return filtered_df.reset_index(drop=True)


rb_correspondences_dbpedia_sales = conservative_filter(
    rb_correspondences_dbpedia_sales,
    dbpedia_by_id,
    sales_by_id,
    score_col_dbpedia_sales,
)

rb_correspondences_metacritic_dbpedia = conservative_filter(
    rb_correspondences_metacritic_dbpedia,
    metacritic_by_id,
    dbpedia_by_id,
    score_col_metacritic_dbpedia,
)

rb_correspondences_metacritic_sales = conservative_filter(
    rb_correspondences_metacritic_sales,
    metacritic_by_id,
    sales_by_id,
    score_col_metacritic_sales,
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_dbpedia_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"),
    index=False,
)
rb_correspondences_metacritic_dbpedia.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_dbpedia.csv"),
    index=False,
)
rb_correspondences_metacritic_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_dbpedia_sales,
        rb_correspondences_metacritic_dbpedia,
        rb_correspondences_metacritic_sales,
    ],
    ignore_index=True,
)


# --------------------------------
# Source-aware fusion functions
# --------------------------------

SOURCE_PRIORITY = {
    "name": ["metacritic", "sales", "dbpedia"],
    "platform": ["metacritic", "sales", "dbpedia"],
    "releaseYear": ["metacritic", "sales", "dbpedia"],
    "developer": ["metacritic", "sales", "dbpedia"],
    "publisher": ["sales", "metacritic", "dbpedia"],
    "criticScore": ["metacritic", "sales", "dbpedia"],
    "userScore": ["metacritic", "sales", "dbpedia"],
    "ESRB": ["sales", "metacritic", "dbpedia"],
    "series": ["dbpedia", "metacritic", "sales"],
    "globalSales": ["sales", "metacritic", "dbpedia"],
}


def source_from_id(record_id):
    if is_nullish(record_id):
        return None
    s = str(record_id)
    if "_" in s:
        return s.split("_", 1)[0]
    return None


def collect_values(values):
    collected = []
    if values is None:
        return collected
    for v in values:
        if isinstance(v, dict):
            val = v.get("value")
            rid = v.get("id") or v.get("record_id") or v.get("source_id")
            src = v.get("source") or source_from_id(rid)
            if not is_nullish(val):
                collected.append({"value": val, "source": src, "id": rid})
        else:
            if not is_nullish(v):
                collected.append({"value": v, "source": None, "id": None})
    return collected


def priority_rank(source, attribute):
    priority = SOURCE_PRIORITY.get(attribute, ["metacritic", "sales", "dbpedia"])
    return priority.index(source) if source in priority else len(priority)


def choose_consensus_text(values, normalizer, attribute):
    vals = collect_values(values)
    if not vals:
        return None

    buckets = defaultdict(list)
    for item in vals:
        norm = normalizer(item["value"])
        if is_nullish(norm):
            continue
        buckets[str(norm)].append(item)

    if not buckets:
        return None

    ranked = []
    for norm_val, items in buckets.items():
        support = len(items)
        best_pri = min(priority_rank(x["source"], attribute) for x in items)
        ranked.append((norm_val, items, support, best_pri))

    ranked.sort(key=lambda x: (-x[2], x[3], len(str(x[0]))))
    _, items, _, _ = ranked[0]
    items = sorted(items, key=lambda x: priority_rank(x["source"], attribute))
    return items[0]["value"]


def robust_numeric_cluster(values, attribute):
    vals = collect_values(values)
    numeric_items = []
    for item in vals:
        num = normalize_numeric(item["value"])
        if num is not None:
            numeric_items.append({"value": num, "source": item["source"]})

    if not numeric_items:
        return []

    tolerance_map = {
        "criticScore": 5.0,
        "userScore": 1.0,
    }
    tol = tolerance_map.get(attribute, 0.0)

    best_group = []
    best_center = None

    for item in numeric_items:
        center = item["value"]
        group = [x for x in numeric_items if abs(x["value"] - center) <= tol]
        group_sources = len(set(x["source"] for x in group if x["source"] is not None))
        best_sources = len(set(x["source"] for x in best_group if x["source"] is not None))

        if len(group) > len(best_group):
            best_group = group
            best_center = center
        elif len(group) == len(best_group):
            if group_sources > best_sources:
                best_group = group
                best_center = center
            elif group_sources == best_sources and best_group:
                curr_dist = sum(abs(x["value"] - center) for x in group)
                best_dist = sum(abs(x["value"] - best_center) for x in best_group)
                if curr_dist < best_dist:
                    best_group = group
                    best_center = center

    return best_group if best_group else numeric_items


def fuse_numeric_from_group(values, attribute, round_digits=None, prefer_median=True):
    group = robust_numeric_cluster(values, attribute)
    if not group:
        return None

    nums = sorted(x["value"] for x in group)

    if prefer_median:
        n = len(nums)
        if n % 2 == 1:
            result = nums[n // 2]
        else:
            result = (nums[n // 2 - 1] + nums[n // 2]) / 2.0
    else:
        best_item = None
        best_distance = None
        best_priority = None
        for item in group:
            total_distance = sum(abs(item["value"] - other["value"]) for other in group)
            pri = priority_rank(item["source"], attribute)
            if (
                best_item is None
                or total_distance < best_distance
                or (total_distance == best_distance and pri < best_priority)
            ):
                best_item = item
                best_distance = total_distance
                best_priority = pri
        result = best_item["value"]

    if round_digits is not None:
        result = round(result, round_digits)
    return result


def fuse_name(values):
    chosen = choose_consensus_text(values, normalize_name, "name")
    return normalize_name(chosen) if chosen is not None else None


def fuse_platform(values):
    chosen = choose_consensus_text(values, normalize_platform, "platform")
    return normalize_platform(chosen) if chosen is not None else None


def fuse_release_year(values):
    chosen = choose_consensus_text(values, extract_year_value, "releaseYear")
    return extract_year_value(chosen) if chosen is not None else None


def fuse_developer(values):
    chosen = choose_consensus_text(values, normalize_developer, "developer")
    return normalize_developer(chosen) if chosen is not None else None


def fuse_publisher(values):
    chosen = choose_consensus_text(values, normalize_publisher, "publisher")
    return normalize_publisher(chosen) if chosen is not None else None


def fuse_esrb(values):
    chosen = choose_consensus_text(values, normalize_esrb, "ESRB")
    return normalize_esrb(chosen) if chosen is not None else None


def fuse_series(values):
    vals = collect_values(values)
    if not vals:
        return None
    dbpedia_vals = [x["value"] for x in vals if x["source"] == "dbpedia" and not is_nullish(x["value"])]
    if dbpedia_vals:
        return max(dbpedia_vals, key=lambda x: len(str(x)))
    non_null = [x["value"] for x in vals if not is_nullish(x["value"])]
    if not non_null:
        return None
    return max(non_null, key=lambda x: len(str(x)))


def fuse_critic_score(values):
    return fuse_numeric_from_group(values, "criticScore", round_digits=0, prefer_median=True)


def fuse_user_score(values):
    return fuse_numeric_from_group(values, "userScore", round_digits=1, prefer_median=True)


def fuse_global_sales(values):
    vals = collect_values(values)
    nums = [normalize_numeric(x["value"]) for x in vals if normalize_numeric(x["value"]) is not None]
    if not nums:
        return None
    return max(nums)


strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", fuse_name)
strategy.add_attribute_fuser("releaseYear", fuse_release_year)
strategy.add_attribute_fuser("developer", fuse_developer)
strategy.add_attribute_fuser("platform", fuse_platform)
strategy.add_attribute_fuser("series", fuse_series)
strategy.add_attribute_fuser("publisher", fuse_publisher)
strategy.add_attribute_fuser("criticScore", fuse_critic_score)
strategy.add_attribute_fuser("userScore", fuse_user_score)
strategy.add_attribute_fuser("ESRB", fuse_esrb)
strategy.add_attribute_fuser("globalSales", fuse_global_sales)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_data = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

if "name" in rb_fused_data.columns:
    rb_fused_data["name"] = rb_fused_data["name"].apply(normalize_name)
if "platform" in rb_fused_data.columns:
    rb_fused_data["platform"] = rb_fused_data["platform"].apply(normalize_platform)
if "releaseYear" in rb_fused_data.columns:
    rb_fused_data["releaseYear"] = rb_fused_data["releaseYear"].apply(extract_year_value)
if "developer" in rb_fused_data.columns:
    rb_fused_data["developer"] = rb_fused_data["developer"].apply(normalize_developer)
if "publisher" in rb_fused_data.columns:
    rb_fused_data["publisher"] = rb_fused_data["publisher"].apply(normalize_publisher)
if "ESRB" in rb_fused_data.columns:
    rb_fused_data["ESRB"] = rb_fused_data["ESRB"].apply(normalize_esrb)
if "criticScore" in rb_fused_data.columns:
    rb_fused_data["criticScore"] = rb_fused_data["criticScore"].apply(
        lambda x: None if is_nullish(x) else int(round(float(x)))
    )
if "userScore" in rb_fused_data.columns:
    rb_fused_data["userScore"] = rb_fused_data["userScore"].apply(
        lambda x: None if is_nullish(x) else round(float(x), 1)
    )

rb_fused_data.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

