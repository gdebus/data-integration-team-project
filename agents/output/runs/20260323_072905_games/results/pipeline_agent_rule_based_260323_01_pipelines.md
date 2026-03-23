# Pipeline Snapshots

notebook_name=pipeline_agent
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=7
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).

from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy, DataFusionEngine,
    voting, longest_string, most_complete,
    median, average,
    union,
    prefer_higher_trust,
)

import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

def _safe_scalar_isna(x):
    """Safe pd.isna() that handles list/array values without crashing."""
    if isinstance(x, (list, tuple, set)):
        return False  # a list is not null, even if it contains null elements
    try:
        result = pd.isna(x)
        if hasattr(result, '__len__') and not isinstance(result, str):
            return False  # array-like result from pd.isna on non-scalar
        return bool(result)
    except (ValueError, TypeError):
        return False

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [Path.cwd(), Path.cwd() / "agents",
                   Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent,
                   Path(__file__).resolve().parent.parent.parent]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_072905_games/"

# === 1. LOAD DATA ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

dbpedia = load_csv(
    "output/runs/20260323_072905_games/normalization/attempt_1/dbpedia.csv",
    name="dbpedia"
)
metacritic = load_csv(
    "output/runs/20260323_072905_games/normalization/attempt_1/metacritic.csv",
    name="metacritic"
)
sales = load_csv(
    "output/runs/20260323_072905_games/normalization/attempt_1/sales.csv",
    name="sales"
)

# === 2. TARGETED INLINE NORMALIZATION ===
# Keep IDs unchanged. Apply only attribute-specific cleanup supported by probe/profile data.

for df in [dbpedia, metacritic, sales]:
    for col in ["name", "developer", "platform", "series", "publisher", "ESRB"]:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].isna(), df[col].astype(str).str.strip())

    if "releaseYear" in df.columns:
        df["releaseYear"] = pd.to_datetime(df["releaseYear"], errors="coerce")

for df in [metacritic, sales]:
    for col in ["criticScore", "userScore"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

if "globalSales" in sales.columns:
    sales["globalSales"] = pd.to_numeric(sales["globalSales"], errors="coerce")

datasets = [dbpedia, metacritic, sales]

# === 3. LIST NORMALIZATION ===
list_like_columns = detect_list_like_columns(
    [dbpedia, metacritic, sales],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    dbpedia, metacritic, sales = normalize_list_like_columns(
        [dbpedia, metacritic, sales], list_like_columns
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [dbpedia, metacritic, sales]

# === 4. BLOCKING ===
# Use precomputed blocking config exactly. EmbeddingBlocker requires materialize().

print("Performing Blocking")

def _flatten_list_cols_for_blocking(df, text_cols):
    """Flatten list-valued cells to strings so EmbeddingBlocker can embed them."""
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: " ".join(str(v) for v in x) if isinstance(x, (list, tuple, set))
                else ("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
            )
    return out

blocker_dbpedia_metacritic = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(dbpedia, ["name", "developer", "platform"]), _flatten_list_cols_for_blocking(metacritic, ["name", "developer", "platform"]),
    text_cols=["name", "developer", "platform"],
    id_column="id",
    top_k=20,
)
candidates_dbpedia_metacritic = blocker_dbpedia_metacritic.materialize()

blocker_dbpedia_sales = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(dbpedia, ["name", "platform", "releaseYear"]), _flatten_list_cols_for_blocking(sales, ["name", "platform", "releaseYear"]),
    text_cols=["name", "platform", "releaseYear"],
    id_column="id",
    top_k=20,
)
candidates_dbpedia_sales = blocker_dbpedia_sales.materialize()

blocker_metacritic_sales = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(metacritic, ["name", "platform", "releaseYear"]), _flatten_list_cols_for_blocking(sales, ["name", "platform", "releaseYear"]),
    text_cols=["name", "platform", "releaseYear"],
    id_column="id",
    top_k=20,
)
candidates_metacritic_sales = blocker_metacritic_sales.materialize()

# === 5. ENTITY MATCHING ===
# Use matching configuration exactly. For string comparators, map lower_strip safely.

print("Matching Entities")

def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if pd.notna(v))
    return str(x).lower().strip()

threshold_dbpedia_sales = 0.75
threshold_metacritic_dbpedia = 0.8
threshold_metacritic_sales = 0.78

comparators_dbpedia_sales = [
    StringComparator(column="name", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="platform", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="developer", similarity_function="cosine", preprocess=lower_strip, list_strategy="concatenate"),
    DateComparator(column="releaseYear", max_days_difference=365),
]

comparators_dbpedia_metacritic = [
    StringComparator(column="name", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="platform", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="developer", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    DateComparator(column="releaseYear", max_days_difference=366),
]

comparators_metacritic_sales = [
    StringComparator(column="name", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="platform", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="developer", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    NumericComparator(column="criticScore", max_difference=5.0),
    NumericComparator(column="userScore", max_difference=1.0),
    DateComparator(column="releaseYear", max_days_difference=365),
]

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_metacritic = matcher.match(
    df_left=dbpedia,
    df_right=metacritic,
    candidates=candidates_dbpedia_metacritic,
    comparators=comparators_dbpedia_metacritic,
    weights=[0.45, 0.2, 0.2, 0.15],
    threshold=threshold_metacritic_dbpedia,
    id_column="id",
)

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=candidates_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=threshold_dbpedia_sales,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=candidates_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.35, 0.2, 0.15, 0.12, 0.08, 0.1],
    threshold=threshold_metacritic_sales,
    id_column="id",
)

# Fail fast if any expected correspondence set is empty
if rb_correspondences_dbpedia_metacritic.empty:
    raise ValueError("Expected correspondences for dbpedia_metacritic are empty.")
if rb_correspondences_dbpedia_sales.empty:
    raise ValueError("Expected correspondences for dbpedia_sales are empty.")
if rb_correspondences_metacritic_sales.empty:
    raise ValueError("Expected correspondences for metacritic_sales are empty.")

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_dbpedia_metacritic.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_metacritic.csv"), index=False
)
rb_correspondences_dbpedia_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"), index=False
)
rb_correspondences_metacritic_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"), index=False
)

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_dbpedia_metacritic,
        rb_correspondences_dbpedia_sales,
        rb_correspondences_metacritic_sales,
    ],
    ignore_index=True,
)

# === 7. DATA FUSION ===
print("Fusing Data")

trust_map = {"dbpedia": 2, "metacritic": 3, "sales": 3}

strategy = DataFusionStrategy("fusion_strategy")

# String attributes
strategy.add_attribute_fuser("name", voting)
strategy.add_attribute_fuser("developer", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("platform", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("series", most_complete)
strategy.add_attribute_fuser("publisher", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("ESRB", prefer_higher_trust, trust_map=trust_map)

# Numeric attributes
strategy.add_attribute_fuser("criticScore", median)
strategy.add_attribute_fuser("userScore", median)
strategy.add_attribute_fuser("globalSales", median)

# Date attributes
strategy.add_attribute_fuser("releaseYear", prefer_higher_trust, trust_map=trust_map)

# === 8. RUN FUSION ===
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
os.makedirs(FUSION_DIR, exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
)

fused_result = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)


# --- Eval ID: extract ALL source IDs matching validation prefix(es) ---
# When a fused cluster contains multiple records from the same source
# (e.g. multiple metacritic_ IDs), we explode the row so each source
# ID gets its own copy. This ensures evaluation can find any gold ID.
import ast as _ast
_EVAL_PREFIXES = ['metacritic_']
def _explode_eval_ids(df):
    rows = []
    for _, row in df.iterrows():
        try:
            sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        except Exception:
            sources = []
        matched = [str(s) for s in sources if any(str(s).startswith(p) for p in _EVAL_PREFIXES)] if isinstance(sources, (list, tuple)) else []
        if matched:
            for eid in matched:
                r = row.copy()
                r["eval_id"] = eid
                rows.append(r)
        else:
            r = row.copy()
            r["eval_id"] = str(row.get("_id", row.get("id", "")))
            rows.append(r)
    return df.__class__(rows)
fused_result = _explode_eval_ids(fused_result)
fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=12
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).

from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    voting,
    longest_string,
    most_complete,
    median,
    average,
    union,
    prefer_higher_trust,
    maximum,
)

import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

def _safe_scalar_isna(x):
    """Safe pd.isna() that handles list/array values without crashing."""
    if isinstance(x, (list, tuple, set)):
        return False  # a list is not null, even if it contains null elements
    try:
        result = pd.isna(x)
        if hasattr(result, '__len__') and not isinstance(result, str):
            return False  # array-like result from pd.isna on non-scalar
        return bool(result)
    except (ValueError, TypeError):
        return False

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [Path.cwd(), Path.cwd() / "agents",
                   Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent,
                   Path(__file__).resolve().parent.parent.parent]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_072905_games/"

# === 1. LOAD DATA ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

dbpedia = load_csv(
    "output/runs/20260323_072905_games/normalization/attempt_1/dbpedia.csv",
    name="dbpedia"
)
metacritic = load_csv(
    "output/runs/20260323_072905_games/normalization/attempt_1/metacritic.csv",
    name="metacritic"
)
sales = load_csv(
    "output/runs/20260323_072905_games/normalization/attempt_1/sales.csv",
    name="sales"
)

# === 2. TARGETED INLINE NORMALIZATION ===
# Keep IDs unchanged. Apply only attribute-specific cleanup supported by probe/profile data.

for df in [dbpedia, metacritic, sales]:
    for col in ["name", "developer", "platform", "series", "publisher", "ESRB"]:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].isna(), df[col].astype(str).str.strip())

    if "releaseYear" in df.columns:
        df["releaseYear"] = pd.to_datetime(df["releaseYear"], errors="coerce")

for df in [metacritic, sales]:
    for col in ["criticScore", "userScore"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

if "globalSales" in sales.columns:
    sales["globalSales"] = pd.to_numeric(sales["globalSales"], errors="coerce")

datasets = [dbpedia, metacritic, sales]

# === 3. LIST NORMALIZATION ===
list_like_columns = detect_list_like_columns(
    [dbpedia, metacritic, sales],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    dbpedia, metacritic, sales = normalize_list_like_columns(
        [dbpedia, metacritic, sales], list_like_columns
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [dbpedia, metacritic, sales]

# === 4. BLOCKING ===
# Use precomputed blocking config exactly. EmbeddingBlocker requires materialize().

print("Performing Blocking")

def _flatten_list_cols_for_blocking(df, text_cols):
    """Flatten list-valued cells to strings so EmbeddingBlocker can embed them."""
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: " ".join(str(v) for v in x) if isinstance(x, (list, tuple, set))
                else ("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
            )
    return out

blocker_dbpedia_metacritic = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(dbpedia, ["name", "developer", "platform"]), _flatten_list_cols_for_blocking(metacritic, ["name", "developer", "platform"]),
    text_cols=["name", "developer", "platform"],
    id_column="id",
    top_k=20,
)
candidates_dbpedia_metacritic = blocker_dbpedia_metacritic.materialize()

blocker_dbpedia_sales = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(dbpedia, ["name", "platform", "releaseYear"]), _flatten_list_cols_for_blocking(sales, ["name", "platform", "releaseYear"]),
    text_cols=["name", "platform", "releaseYear"],
    id_column="id",
    top_k=20,
)
candidates_dbpedia_sales = blocker_dbpedia_sales.materialize()

blocker_metacritic_sales = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(metacritic, ["name", "platform", "releaseYear"]), _flatten_list_cols_for_blocking(sales, ["name", "platform", "releaseYear"]),
    text_cols=["name", "platform", "releaseYear"],
    id_column="id",
    top_k=20,
)
candidates_metacritic_sales = blocker_metacritic_sales.materialize()

# === 5. ENTITY MATCHING ===
# Use matching configuration exactly. For string comparators, map lower_strip safely.

print("Matching Entities")

def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if pd.notna(v))
    return str(x).lower().strip()

threshold_dbpedia_sales = 0.75
threshold_metacritic_dbpedia = 0.8
threshold_metacritic_sales = 0.78

comparators_dbpedia_sales = [
    StringComparator(column="name", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="platform", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="developer", similarity_function="cosine", preprocess=lower_strip, list_strategy="concatenate"),
    DateComparator(column="releaseYear", max_days_difference=365),
]

comparators_dbpedia_metacritic = [
    StringComparator(column="name", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="platform", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="developer", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    DateComparator(column="releaseYear", max_days_difference=366),
]

comparators_metacritic_sales = [
    StringComparator(column="name", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="platform", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="developer", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    NumericComparator(column="criticScore", max_difference=5.0),
    NumericComparator(column="userScore", max_difference=1.0),
    DateComparator(column="releaseYear", max_days_difference=365),
]

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_metacritic = matcher.match(
    df_left=dbpedia,
    df_right=metacritic,
    candidates=candidates_dbpedia_metacritic,
    comparators=comparators_dbpedia_metacritic,
    weights=[0.45, 0.2, 0.2, 0.15],
    threshold=threshold_metacritic_dbpedia,
    id_column="id",
)

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=candidates_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=threshold_dbpedia_sales,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=candidates_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.35, 0.2, 0.15, 0.12, 0.08, 0.1],
    threshold=threshold_metacritic_sales,
    id_column="id",
)

# Fail fast if any expected correspondence set is empty
if rb_correspondences_dbpedia_metacritic.empty:
    raise ValueError("Expected correspondences for dbpedia_metacritic are empty.")
if rb_correspondences_dbpedia_sales.empty:
    raise ValueError("Expected correspondences for dbpedia_sales are empty.")
if rb_correspondences_metacritic_sales.empty:
    raise ValueError("Expected correspondences for metacritic_sales are empty.")

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_dbpedia_metacritic.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_metacritic.csv"), index=False
)
rb_correspondences_dbpedia_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"), index=False
)
rb_correspondences_metacritic_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"), index=False
)

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_dbpedia_metacritic,
        rb_correspondences_dbpedia_sales,
        rb_correspondences_metacritic_sales,
    ],
    ignore_index=True,
)
from PyDI.fusion import shortest_string
from PyDI.entitymatching import MaximumBipartiteMatching



# === 7. POST-CLUSTERING ===
print("Applying Post-Clustering")

refined_dbpedia_metacritic = MaximumBipartiteMatching().cluster(rb_correspondences_dbpedia_metacritic)
refined_dbpedia_sales = MaximumBipartiteMatching().cluster(rb_correspondences_dbpedia_sales)
refined_metacritic_sales = MaximumBipartiteMatching().cluster(rb_correspondences_metacritic_sales)

all_rb_correspondences = pd.concat(
    [
        refined_dbpedia_metacritic,
        refined_dbpedia_sales,
        refined_metacritic_sales,
    ],
    ignore_index=True,
)

# === 8. DATA FUSION ===
print("Fusing Data")

trust_map = {"dbpedia": 1, "metacritic": 3, "sales": 2}

strategy = DataFusionStrategy("fusion_strategy")

# String attributes
strategy.add_attribute_fuser("name", shortest_string)
strategy.add_attribute_fuser("developer", voting)
strategy.add_attribute_fuser("platform", voting)
strategy.add_attribute_fuser("series", most_complete)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("ESRB", prefer_higher_trust, trust_map={"dbpedia": 1, "metacritic": 3, "sales": 2})

# Numeric attributes
strategy.add_attribute_fuser("criticScore", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("userScore", median)
strategy.add_attribute_fuser("globalSales", median)

# Date attributes
strategy.add_attribute_fuser("releaseYear", voting)




# === 8. RUN FUSION ===
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
os.makedirs(FUSION_DIR, exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
)

fused_result = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)


# --- Eval ID: extract ALL source IDs matching validation prefix(es) ---
# When a fused cluster contains multiple records from the same source
# (e.g. multiple metacritic_ IDs), we explode the row so each source
# ID gets its own copy. This ensures evaluation can find any gold ID.
import ast as _ast
_EVAL_PREFIXES = ['metacritic_']
def _explode_eval_ids(df):
    rows = []
    for _, row in df.iterrows():
        try:
            sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        except Exception:
            sources = []
        matched = [str(s) for s in sources if any(str(s).startswith(p) for p in _EVAL_PREFIXES)] if isinstance(sources, (list, tuple)) else []
        if matched:
            for eid in matched:
                r = row.copy()
                r["eval_id"] = eid
                rows.append(r)
        else:
            r = row.copy()
            r["eval_id"] = str(row.get("_id", row.get("id", "")))
            rows.append(r)
    return df.__class__(rows)
fused_result = _explode_eval_ids(fused_result)
fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

