from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker, StringComparator, NumericComparator, RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy, DataFusionEngine,
    longest_string, most_complete, median, maximum, union,
    prefer_higher_trust,
)
import pandas as pd
import numpy as np
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
OUTPUT_DIR = "output/runs/20260322_155058_books/"

# === 1. LOAD DATA ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

amazon_small = load_csv(
    "output/runs/20260322_155058_books/normalization/attempt_1/amazon_small.csv",
    name="amazon_small",
)
goodreads_small = load_csv(
    "output/runs/20260322_155058_books/normalization/attempt_1/goodreads_small.csv",
    name="goodreads_small",
)
metabooks_small = load_csv(
    "output/runs/20260322_155058_books/normalization/attempt_1/metabooks_small.csv",
    name="metabooks_small",
)

def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if v is not None and not pd.isna(v))
    try:
        _is_na = pd.isna(x)
    except (ValueError, TypeError):
        _is_na = False
    if _is_na:
        return ""
    return str(x).lower().strip()

# targeted inline normalization supported by probe data
for df in [amazon_small, goodreads_small, metabooks_small]:
    for col in ["title", "author", "publisher", "genres", "language", "bookformat", "edition", "isbn_clean"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

# === 2. SCHEMA ALIGNMENT ===
# Align to amazon_small as reference where needed
goodreads_small = goodreads_small.rename(columns={})
metabooks_small = metabooks_small.rename(columns={})

# === 3. LIST NORMALIZATION ===
list_like_columns = detect_list_like_columns(
    [amazon_small, goodreads_small, metabooks_small],
    exclude_columns={"id", "_id", "isbn_clean"},
)
if list_like_columns:
    amazon_small, goodreads_small, metabooks_small = normalize_list_like_columns(
        [amazon_small, goodreads_small, metabooks_small], list_like_columns
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [amazon_small, goodreads_small, metabooks_small]

# === 4. BLOCKING ===
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

blocker_amazon_goodreads = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(amazon_small, ["title", "author", "publish_year"]), _flatten_list_cols_for_blocking(goodreads_small, ["title", "author", "publish_year"]),
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=20,
)
candidates_amazon_goodreads = blocker_amazon_goodreads.materialize()

blocker_amazon_metabooks = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(amazon_small, ["title", "author", "publisher"]), _flatten_list_cols_for_blocking(metabooks_small, ["title", "author", "publisher"]),
    text_cols=["title", "author", "publisher"],
    id_column="id",
    top_k=15,
)
candidates_amazon_metabooks = blocker_amazon_metabooks.materialize()

blocker_goodreads_metabooks = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(goodreads_small, ["title", "author", "publisher"]), _flatten_list_cols_for_blocking(metabooks_small, ["title", "author", "publisher"]),
    text_cols=["title", "author", "publisher"],
    id_column="id",
    top_k=20,
)
candidates_goodreads_metabooks = blocker_goodreads_metabooks.materialize()

# === 5. ENTITY MATCHING ===
print("Matching Entities")

threshold_goodreads_small_amazon_small = 0.66
threshold_metabooks_small_amazon_small = 0.72
threshold_metabooks_small_goodreads_small = 0.68

comparators_amazon_goodreads = [
    StringComparator(column="title", similarity_function="jaro_winkler", preprocess=lower_strip),
    StringComparator(column="author", similarity_function="jaro_winkler", preprocess=lower_strip),
    NumericComparator(column="publish_year", method="absolute_difference", max_difference=1.0),
    StringComparator(column="publisher", similarity_function="cosine", preprocess=lower_strip),
]

comparators_amazon_metabooks = [
    StringComparator(column="title", similarity_function="cosine", preprocess=lower_strip),
    StringComparator(column="author", similarity_function="jaro_winkler", preprocess=lower_strip),
    StringComparator(column="publisher", similarity_function="jaro_winkler", preprocess=lower_strip),
    NumericComparator(column="publish_year", method="absolute_difference", max_difference=1.0),
]

comparators_goodreads_metabooks = [
    StringComparator(column="title", similarity_function="jaro_winkler", preprocess=lower_strip),
    StringComparator(column="author", similarity_function="jaro_winkler", preprocess=lower_strip),
    StringComparator(column="publisher", similarity_function="jaro_winkler", preprocess=lower_strip),
    StringComparator(column="genres", similarity_function="jaccard", preprocess=lower_strip, list_strategy="set_jaccard"),
    NumericComparator(column="publish_year", method="absolute_difference", max_difference=1.0),
]

matcher = RuleBasedMatcher()

rb_correspondences_amazon_goodreads = matcher.match(
    df_left=amazon_small,
    df_right=goodreads_small,
    candidates=candidates_amazon_goodreads,
    comparators=comparators_amazon_goodreads,
    weights=[0.42, 0.33, 0.15, 0.1],
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_amazon_metabooks = matcher.match(
    df_left=amazon_small,
    df_right=metabooks_small,
    candidates=candidates_amazon_metabooks,
    comparators=comparators_amazon_metabooks,
    weights=[0.5, 0.25, 0.15, 0.1],
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_goodreads_metabooks = matcher.match(
    df_left=goodreads_small,
    df_right=metabooks_small,
    candidates=candidates_goodreads_metabooks,
    comparators=comparators_goodreads_metabooks,
    weights=[0.42, 0.23, 0.12, 0.15, 0.08],
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

if rb_correspondences_amazon_goodreads.empty:
    raise ValueError("Empty correspondences for pair amazon_small_goodreads_small")
if rb_correspondences_amazon_metabooks.empty:
    raise ValueError("Empty correspondences for pair amazon_small_metabooks_small")
if rb_correspondences_goodreads_metabooks.empty:
    raise ValueError("Empty correspondences for pair goodreads_small_metabooks_small")

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_amazon_goodreads.to_csv(
    os.path.join(CORR_DIR, "correspondences_amazon_small_goodreads_small.csv"),
    index=False,
)
rb_correspondences_amazon_metabooks.to_csv(
    os.path.join(CORR_DIR, "correspondences_amazon_small_metabooks_small.csv"),
    index=False,
)
rb_correspondences_goodreads_metabooks.to_csv(
    os.path.join(CORR_DIR, "correspondences_goodreads_small_metabooks_small.csv"),
    index=False,
)

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_amazon_goodreads,
        rb_correspondences_amazon_metabooks,
        rb_correspondences_goodreads_metabooks,
    ],
    ignore_index=True,
)

# === 7. DATA FUSION ===
print("Fusing Data")

trust_map = {"amazon_small": 3, "goodreads_small": 2, "metabooks_small": 2}

strategy = DataFusionStrategy("fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("isbn_clean", prefer_higher_trust, trust_map=trust_map)

strategy.add_attribute_fuser("publish_year", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("rating", median)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", median)

strategy.add_attribute_fuser("language", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("bookformat", most_complete)
strategy.add_attribute_fuser("edition", most_complete)

strategy.add_attribute_fuser("genres", union, separator="; ")

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
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)


# --- Eval ID: extract source ID matching validation prefix for reliable evaluation ---
import ast as _ast
_EVAL_PREFIX = "amazon_"
def _extract_eval_id(row):
    try:
        sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        if isinstance(sources, (list, tuple)):
            for sid in sources:
                if str(sid).startswith(_EVAL_PREFIX):
                    return str(sid)
    except Exception:
        pass
    return str(row.get("_id", row.get("id", "")))
fused_result["eval_id"] = fused_result.apply(_extract_eval_id, axis=1)
fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)