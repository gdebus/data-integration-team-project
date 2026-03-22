from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker, StringComparator, DateComparator, RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy, DataFusionEngine,
    voting, most_complete, median, earliest,
    prefer_higher_trust,
)
import pandas as pd
import os


# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260322_102925_companies/"


# === 1. LOAD DATA ===
forbes = load_csv(
    "output/runs/20260322_102925_companies/normalization/attempt_1/forbes.csv",
    name="forbes",
)
dbpedia = load_csv(
    "output/runs/20260322_102925_companies/normalization/attempt_1/dbpedia.csv",
    name="dbpedia",
)
fullcontact = load_csv(
    "output/runs/20260322_102925_companies/normalization/attempt_1/fullcontact.csv",
    name="fullcontact",
)

datasets = [forbes, dbpedia, fullcontact]


# === 2. TARGETED INLINE NORMALIZATION ===
def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if pd.notna(v))
    return str(x).lower().strip()


for df in datasets:
    for col in ["name", "country", "industry", "city", "keypeople_name", "Sector", "Continent"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

for df in [dbpedia, fullcontact]:
    if "founded" in df.columns:
        df["founded"] = pd.to_datetime(df["founded"], errors="coerce")


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

blocker_forbes_dbpedia = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(forbes, ["name", "country", "industry"]), _flatten_list_cols_for_blocking(dbpedia, ["name", "country", "industry"]),
    text_cols=["name", "country", "industry"],
    id_column="id",
    top_k=15,
)
candidates_forbes_dbpedia = blocker_forbes_dbpedia.materialize()

blocker_forbes_fullcontact = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(forbes, ["name", "country"]), _flatten_list_cols_for_blocking(fullcontact, ["name", "country"]),
    text_cols=["name", "country"],
    id_column="id",
    top_k=20,
)
candidates_forbes_fullcontact = blocker_forbes_fullcontact.materialize()

blocker_fullcontact_dbpedia = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(fullcontact, ["name", "country", "city"]), _flatten_list_cols_for_blocking(dbpedia, ["name", "country", "city"]),
    text_cols=["name", "country", "city"],
    id_column="id",
    top_k=20,
)
candidates_fullcontact_dbpedia = blocker_fullcontact_dbpedia.materialize()


# === 4. ENTITY MATCHING ===
print("Matching Entities")

threshold_forbes_dbpedia = 0.68
threshold_forbes_fullcontact = 0.82
threshold_fullcontact_dbpedia = 0.72

comparators_forbes_dbpedia = [
    StringComparator(column="name", similarity_function="jaro_winkler", preprocess=lower_strip),
    StringComparator(column="country", similarity_function="jaccard", preprocess=lower_strip),
    StringComparator(column="industry", similarity_function="cosine", preprocess=lower_strip),
]

comparators_forbes_fullcontact = [
    StringComparator(column="name", similarity_function="jaccard", preprocess=lower_strip),
    StringComparator(column="country", similarity_function="jaccard", preprocess=lower_strip),
]

comparators_fullcontact_dbpedia = [
    StringComparator(column="name", similarity_function="jaro_winkler", preprocess=lower_strip),
    StringComparator(column="country", similarity_function="jaccard", preprocess=lower_strip),
    StringComparator(column="city", similarity_function="jaccard", preprocess=lower_strip),
    DateComparator(column="founded", max_days_difference=730),
]

matcher = RuleBasedMatcher()

corr_forbes_dbpedia = matcher.match(
    df_left=forbes,
    df_right=dbpedia,
    candidates=candidates_forbes_dbpedia,
    comparators=comparators_forbes_dbpedia,
    weights=[0.55, 0.2, 0.25],
    threshold=threshold_forbes_dbpedia,
    id_column="id",
)

corr_forbes_fullcontact = matcher.match(
    df_left=forbes,
    df_right=fullcontact,
    candidates=candidates_forbes_fullcontact,
    comparators=comparators_forbes_fullcontact,
    weights=[0.8, 0.2],
    threshold=threshold_forbes_fullcontact,
    id_column="id",
)

corr_fullcontact_dbpedia = matcher.match(
    df_left=fullcontact,
    df_right=dbpedia,
    candidates=candidates_fullcontact_dbpedia,
    comparators=comparators_fullcontact_dbpedia,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=threshold_fullcontact_dbpedia,
    id_column="id",
)

if corr_forbes_dbpedia.empty:
    raise ValueError("Empty correspondences for expected pair: forbes_dbpedia")
if corr_forbes_fullcontact.empty:
    raise ValueError("Empty correspondences for expected pair: forbes_fullcontact")
if corr_fullcontact_dbpedia.empty:
    raise ValueError("Empty correspondences for expected pair: fullcontact_dbpedia")


# === 5. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

corr_forbes_dbpedia.to_csv(
    os.path.join(CORR_DIR, "correspondences_forbes_dbpedia.csv"),
    index=False,
)
corr_forbes_fullcontact.to_csv(
    os.path.join(CORR_DIR, "correspondences_forbes_fullcontact.csv"),
    index=False,
)
corr_fullcontact_dbpedia.to_csv(
    os.path.join(CORR_DIR, "correspondences_fullcontact_dbpedia.csv"),
    index=False,
)

all_correspondences = pd.concat(
    [corr_forbes_dbpedia, corr_forbes_fullcontact, corr_fullcontact_dbpedia],
    ignore_index=True,
)
from PyDI.entitymatching import MaximumBipartiteMatching

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



# === 6. POST-CLUSTERING ===
refined_forbes_dbpedia = MaximumBipartiteMatching().cluster(corr_forbes_dbpedia)
refined_forbes_fullcontact = corr_forbes_fullcontact
refined_fullcontact_dbpedia = MaximumBipartiteMatching().cluster(corr_fullcontact_dbpedia)

all_correspondences = pd.concat(
    [refined_forbes_dbpedia, refined_forbes_fullcontact, refined_fullcontact_dbpedia],
    ignore_index=True,
)

# === 7. DATA FUSION ===
print("Fusing Data")

trust_map = {"forbes": 3, "dbpedia": 2, "fullcontact": 1}
city_trust_map = {"fullcontact": 3, "dbpedia": 2, "forbes": 1}

strategy = DataFusionStrategy("fusion_strategy")

trust_map_city = {"fullcontact": 3, "dbpedia": 2, "forbes": 1}
trust_map_founded = {"dbpedia": 3, "forbes": 2, "fullcontact": 1}
strategy.add_attribute_fuser("name", voting)
strategy.add_attribute_fuser("country", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("city", prefer_higher_trust, trust_map=trust_map_city)
strategy.add_attribute_fuser("industry", most_complete)
strategy.add_attribute_fuser("keypeople_name", most_complete)
strategy.add_attribute_fuser("Sector", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("Continent", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("Market Value", median)
strategy.add_attribute_fuser("Sales", most_complete)
strategy.add_attribute_fuser("Profits", median)
strategy.add_attribute_fuser("assets", most_complete)
strategy.add_attribute_fuser("Rank", median)
strategy.add_attribute_fuser("founded", prefer_higher_trust, trust_map=trust_map_founded)





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
    datasets=[forbes, dbpedia, fullcontact],
    correspondences=all_correspondences,
    id_column="id",
    include_singletons=True,
)

fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)