from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker, StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    voting,
    most_complete,
    median,
    earliest,
    prefer_higher_trust,
    maximum,
)
import pandas as pd
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
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260322_175903_games/"

# === 1. LOAD DATA ===
dbpedia = load_csv(
    "output/runs/20260322_175903_games/normalization/attempt_1/dbpedia.csv",
    name="dbpedia",
)
metacritic = load_csv(
    "output/runs/20260322_175903_games/normalization/attempt_1/metacritic.csv",
    name="metacritic",
)
sales = load_csv(
    "output/runs/20260322_175903_games/normalization/attempt_1/sales.csv",
    name="sales",
)

# === 1b. TARGETED INLINE NORMALIZATION ===
def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if pd.notna(v))
    return str(x).lower().strip()

for df in [dbpedia, metacritic, sales]:
    for col in ["name", "developer", "platform"]:
        if col in df.columns:
            df[col] = df[col].astype("object")

for df in [metacritic, sales]:
    for col in ["criticScore", "userScore"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# === 2. LIST NORMALIZATION ===
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

# === 3. BLOCKING ===
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

# === 4. ENTITY MATCHING ===
print("Matching Entities")

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

if rb_correspondences_dbpedia_metacritic.empty:
    raise ValueError("Empty correspondences for dbpedia_metacritic")
if rb_correspondences_dbpedia_sales.empty:
    raise ValueError("Empty correspondences for dbpedia_sales")
if rb_correspondences_metacritic_sales.empty:
    raise ValueError("Empty correspondences for metacritic_sales")

# === 5. SAVE CORRESPONDENCES ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_dbpedia_metacritic.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_metacritic.csv"),
    index=False,
)
rb_correspondences_dbpedia_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"),
    index=False,
)
rb_correspondences_metacritic_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"),
    index=False,
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



# === 6. POST-CLUSTERING ===
print("Applying Post-Clustering")

post_clusterer = MaximumBipartiteMatching()
refined_correspondences_dbpedia_metacritic = post_clusterer.cluster(rb_correspondences_dbpedia_metacritic)
refined_correspondences_dbpedia_sales = post_clusterer.cluster(rb_correspondences_dbpedia_sales)
refined_correspondences_metacritic_sales = post_clusterer.cluster(rb_correspondences_metacritic_sales)

all_rb_correspondences = pd.concat(
    [
        refined_correspondences_dbpedia_metacritic,
        refined_correspondences_dbpedia_sales,
        refined_correspondences_metacritic_sales,
    ],
    ignore_index=True,
)

# === 7. DATA FUSION ===
print("Fusing Data")

trust_map = {"dbpedia": 1, "metacritic": 3, "sales": 2}
release_year_trust_map = {"dbpedia": 1, "metacritic": 2, "sales": 3}
esrb_trust_map = {"dbpedia": 1, "metacritic": 4, "sales": 3}
publisher_trust_map = {"dbpedia": 1, "metacritic": 2, "sales": 5}
critic_score_trust_map = {"dbpedia": 1, "metacritic": 2, "sales": 4}
user_score_trust_map = {"dbpedia": 1, "metacritic": 3, "sales": 2}

strategy = DataFusionStrategy("fusion_strategy")

trust_map_releaseYear = {"sales": 3, "metacritic": 2, "dbpedia": 1}
trust_map_ESRB = {"metacritic": 2, "sales": 1}
trust_map_publisher = {"sales": 1}
trust_map_criticScore = {"sales": 2, "metacritic": 1}
strategy.add_attribute_fuser("name", shortest_string)
strategy.add_attribute_fuser("releaseYear", prefer_higher_trust, trust_map=trust_map_releaseYear)
strategy.add_attribute_fuser("developer", voting)
strategy.add_attribute_fuser("platform", shortest_string)
strategy.add_attribute_fuser("series", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("criticScore", prefer_higher_trust, trust_map=trust_map_criticScore)
strategy.add_attribute_fuser("userScore", prefer_higher_trust, trust_map=user_score_trust_map)
strategy.add_attribute_fuser("ESRB", prefer_higher_trust, trust_map=trust_map_ESRB)
strategy.add_attribute_fuser("publisher", prefer_higher_trust, trust_map=trust_map_publisher)
strategy.add_attribute_fuser("globalSales", prefer_higher_trust, trust_map={"dbpedia": 1, "metacritic": 1, "sales": 5})




# === 7. RUN FUSION ===
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


# --- Eval ID: extract source ID matching validation prefix(es) for reliable evaluation ---
import ast as _ast
_EVAL_PREFIXES = ['metacritic_']
def _extract_eval_id(row):
    try:
        sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        if isinstance(sources, (list, tuple)):
            for sid in sources:
                s = str(sid)
                if any(s.startswith(p) for p in _EVAL_PREFIXES):
                    return s
    except Exception:
        pass
    return str(row.get("_id", row.get("id", "")))
fused_result["eval_id"] = fused_result.apply(_extract_eval_id, axis=1)
fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)