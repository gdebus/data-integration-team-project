# Pipeline Snapshots

notebook_name=AdaptationPipeline_Games
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=3
node_name=execute_pipeline
accuracy_score=80.67%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, maximum

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/games/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (XML)
dbpedia = load_xml(
    DATA_DIR + "dbpedia.xml",
    name="dbpedia",
)

metacritic = load_xml(
    DATA_DIR + "metacritic.xml",
    name="metacritic",
)

sales = load_xml(
    DATA_DIR + "sales.xml",
    name="sales",
)

# --------------------------------
# Light normalization helpers (kept minimal & robust)
# --------------------------------
def _to_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)

def normalize_name(x: object) -> str:
    s = _to_str(x).lower().strip()
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def year_from_date(x: object):
    s = _to_str(x).strip()
    if not s:
        return np.nan
    # Handles "YYYY-01-01" or "YYYY"
    m = re.match(r"^(\d{4})", s)
    if not m:
        return np.nan
    return float(m.group(1))

def normalize_platform(x: object) -> str:
    s = _to_str(x).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_devpub(x: object) -> str:
    s = _to_str(x).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

for df in (dbpedia, metacritic, sales):
    # common normalized columns used for blocking/matching
    if "name" in df.columns:
        df["name_norm"] = df["name"].map(normalize_name)
    else:
        df["name_norm"] = ""

    if "platform" in df.columns:
        df["platform_norm"] = df["platform"].map(normalize_platform)
    else:
        df["platform_norm"] = ""

    if "developer" in df.columns:
        df["developer_norm"] = df["developer"].map(normalize_devpub)
    else:
        df["developer_norm"] = ""

    if "publisher" in df.columns:
        df["publisher_norm"] = df["publisher"].map(normalize_devpub)

    if "releaseYear" in df.columns:
        df["release_year_num"] = df["releaseYear"].map(year_from_date)
    else:
        df["release_year_num"] = np.nan

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of dbpedia.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True
)

# Match schema of dbpedia with metacritic and rename metacritic to dbpedia schema
schema_correspondences = schema_matcher.match(dbpedia, metacritic)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
metacritic = metacritic.rename(columns=rename_map)

# Match schema of dbpedia with sales and rename sales to dbpedia schema
schema_correspondences = schema_matcher.match(dbpedia, sales)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
sales = sales.rename(columns=rename_map)

# --------------------------------
# Blocking (MANDATORY POLICY)
# Use informative identity signals: canonical name + disambiguators.
# Here: name_norm + release_year_num + platform_norm (all high alignment across datasets).
# --------------------------------

print("Performing Blocking")

blocker_d2m = StandardBlocker(
    dbpedia, metacritic,
    on=["name_norm", "platform_norm", "release_year_num"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

blocker_d2s = StandardBlocker(
    dbpedia, sales,
    on=["name_norm", "platform_norm", "release_year_num"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

blocker_m2s = StandardBlocker(
    metacritic, sales,
    on=["name_norm", "platform_norm", "release_year_num"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

# --------------------------------
# Matching configuration (rule-based)
# --------------------------------

comparators_d2m = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="platform_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="release_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="developer_norm",
        similarity_function="jaccard",
    ),
]

comparators_d2s = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="platform_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="release_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="developer_norm",
        similarity_function="jaccard",
    ),
]

comparators_m2s = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="platform_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="release_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="developer_norm",
        similarity_function="jaccard",
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_d2m = rb_matcher.match(
    df_left=dbpedia,
    df_right=metacritic,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.55, 0.20, 0.15, 0.10],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_d2s = rb_matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_d2s,
    comparators=comparators_d2s,
    weights=[0.55, 0.20, 0.15, 0.10],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_m2s = rb_matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_m2s,
    comparators=comparators_m2s,
    weights=[0.55, 0.20, 0.15, 0.10],
    threshold=0.75,
    id_column="id",
)

print("Fusing Data")

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [rb_correspondences_d2m, rb_correspondences_d2s, rb_correspondences_m2s],
    ignore_index=True
)

# --------------------------------
# Data Fusion
# Output schema follows dbpedia (dataset1) due to schema matching step.
# --------------------------------

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# dbpedia base attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", longest_string)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)

# if present after schema matching / shared fields
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("criticScore", maximum)
strategy.add_attribute_fuser("userScore", maximum)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("globalSales", maximum)

# keep normalized helper columns deterministic
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("platform_norm", longest_string)
strategy.add_attribute_fuser("developer_norm", longest_string)
strategy.add_attribute_fuser("release_year_num", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=8
node_name=execute_pipeline
accuracy_score=67.23%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    maximum,
    prefer_higher_trust,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/games/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

dbpedia = load_xml(DATA_DIR + "dbpedia.xml", name="dbpedia")
metacritic = load_xml(DATA_DIR + "metacritic.xml", name="metacritic")
sales = load_xml(DATA_DIR + "sales.xml", name="sales")

# --------------------------------
# Normalization helpers
# --------------------------------
_CORP_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "gmbh", "sarl", "sa", "ag", "bv", "plc", "pte", "kg", "oy", "ab", "spa", "s p a"
}
_DEV_PUB_SYNONYMS = {
    "nintendo ead": "nintendo",
    "nintendo ead tokyo": "nintendo",
    "sony computer entertainment": "sony",
    "sony interactive entertainment": "sony",
    "microsoft game studios": "microsoft",
    "microsoft studios": "microsoft",
    "electronic arts": "ea",
    "ea games": "ea",
    "rockstar games": "rockstar",
    "rockstar north": "rockstar",
    "activision publishing": "activision",
    "square enix": "squareenix",
}
_PLATFORM_SYNONYMS = {
    "playstation 4": "ps4",
    "ps 4": "ps4",
    "ps4": "ps4",
    "playstation 5": "ps5",
    "ps 5": "ps5",
    "ps5": "ps5",
    "playstation 3": "ps3",
    "ps 3": "ps3",
    "ps3": "ps3",
    "playstation 2": "ps2",
    "ps 2": "ps2",
    "ps2": "ps2",
    "playstation": "ps1",
    "ps 1": "ps1",
    "ps1": "ps1",
    "xbox one": "xboxone",
    "xbox series x": "xboxseries",
    "xbox series s": "xboxseries",
    "xbox 360": "xbox360",
    "nintendo switch": "switch",
    "wii u": "wiiu",
    "wii": "wii",
    "nintendo 3ds": "3ds",
    "nintendo ds": "ds",
    "game boy color": "gbc",
    "game boy advance": "gba",
    "gamecube": "gcn",
    "pc": "pc",
    "windows": "pc",
    "mac": "mac",
    "ios": "ios",
    "android": "android",
}

def _is_null(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))

def _to_str(x) -> str:
    if _is_null(x):
        return ""
    return str(x)

def normalize_name(x: object) -> str:
    s = _to_str(x).lower().strip()
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def year_from_date(x: object):
    s = _to_str(x).strip()
    if not s:
        return np.nan
    m = re.match(r"^(\d{4})", s)
    if not m:
        return np.nan
    return float(m.group(1))

def normalize_platform(x: object) -> str:
    s = _to_str(x).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # canonicalize common platform variants
    return _PLATFORM_SYNONYMS.get(s, s)

def normalize_org(x: object) -> str:
    s = _to_str(x).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    # canonicalize well-known variants first
    s = _DEV_PUB_SYNONYMS.get(s, s)
    parts = [p for p in s.split() if p not in _CORP_SUFFIXES]
    s2 = " ".join(parts).strip()
    return _DEV_PUB_SYNONYMS.get(s2, s2)

def normalize_esrb(x: object) -> str:
    s = _to_str(x).upper().strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = s.replace("RATING PENDING", "RP")
    s = s.replace("MATURE", "M")
    s = s.replace("EVERYONE", "E")
    s = s.replace("TEEN", "T")
    s = s.replace("ADULTS ONLY", "AO")
    s = re.sub(r"[^A-Z0-9]+", "", s)
    # common abbreviations
    if s in {"E10PLUS", "E10"}:
        return "E10+"
    if s in {"EC", "E", "T", "M", "AO", "RP"}:
        return s
    if s == "E10+":
        return s
    return s

def to_float(x: object):
    s = _to_str(x).strip()
    if not s:
        return np.nan
    # keep digits and dot
    s = re.sub(r"[^0-9.]+", "", s)
    if s == "" or s == ".":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_user_score(x: object):
    """
    User score often comes as 0-10 float. Some sources provide 0-100; normalize to 0-10.
    """
    v = to_float(x)
    if np.isnan(v):
        return np.nan
    if v > 10.0 and v <= 100.0:
        v = v / 10.0
    return float(v)

def normalize_critic_score(x: object):
    """
    Critic score typically 0-100.
    """
    v = to_float(x)
    if np.isnan(v):
        return np.nan
    # If somehow given as 0-10, scale up
    if v <= 10.0:
        v = v * 10.0
    return float(v)

def normalize_global_sales(x: object):
    v = to_float(x)
    if np.isnan(v):
        return np.nan
    return float(v)

for df in (dbpedia, metacritic, sales):
    df["name_norm"] = df["name"].map(normalize_name) if "name" in df.columns else ""
    df["platform_norm"] = df["platform"].map(normalize_platform) if "platform" in df.columns else ""
    df["release_year_num"] = df["releaseYear"].map(year_from_date) if "releaseYear" in df.columns else np.nan

    df["developer_norm"] = df["developer"].map(normalize_org) if "developer" in df.columns else ""
    if "publisher" in df.columns:
        df["publisher_norm"] = df["publisher"].map(normalize_org)
    else:
        df["publisher_norm"] = ""

    if "ESRB" in df.columns:
        df["ESRB_norm"] = df["ESRB"].map(normalize_esrb)
    else:
        df["ESRB_norm"] = ""

    if "criticScore" in df.columns:
        df["critic_score_num"] = df["criticScore"].map(normalize_critic_score)
    else:
        df["critic_score_num"] = np.nan

    if "userScore" in df.columns:
        df["user_score_num"] = df["userScore"].map(normalize_user_score)
    else:
        df["user_score_num"] = np.nan

    if "globalSales" in df.columns:
        df["global_sales_num"] = df["globalSales"].map(normalize_global_sales)
    else:
        df["global_sales_num"] = np.nan

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of dbpedia.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

schema_correspondences = schema_matcher.match(dbpedia, metacritic)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metacritic = metacritic.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(dbpedia, sales)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
sales = sales.rename(columns=rename_map)

# Ensure helper columns exist after renaming (schema matching may affect column names in sources)
for df in (metacritic, sales):
    if "name_norm" not in df.columns and "name" in df.columns:
        df["name_norm"] = df["name"].map(normalize_name)
    if "platform_norm" not in df.columns and "platform" in df.columns:
        df["platform_norm"] = df["platform"].map(normalize_platform)
    if "release_year_num" not in df.columns and "releaseYear" in df.columns:
        df["release_year_num"] = df["releaseYear"].map(year_from_date)
    if "developer_norm" not in df.columns and "developer" in df.columns:
        df["developer_norm"] = df["developer"].map(normalize_org)
    if "publisher_norm" not in df.columns:
        df["publisher_norm"] = df["publisher"].map(normalize_org) if "publisher" in df.columns else ""
    if "ESRB_norm" not in df.columns:
        df["ESRB_norm"] = df["ESRB"].map(normalize_esrb) if "ESRB" in df.columns else ""
    if "critic_score_num" not in df.columns:
        df["critic_score_num"] = df["criticScore"].map(normalize_critic_score) if "criticScore" in df.columns else np.nan
    if "user_score_num" not in df.columns:
        df["user_score_num"] = df["userScore"].map(normalize_user_score) if "userScore" in df.columns else np.nan
    if "global_sales_num" not in df.columns:
        df["global_sales_num"] = df["globalSales"].map(normalize_global_sales) if "globalSales" in df.columns else np.nan

# --------------------------------
# Blocking (MANDATORY POLICY)
# Use informative identity signals: canonical name + disambiguator.
# Avoid too-strict multi-column exact blocking that hurts recall.
# --------------------------------

print("Performing Blocking")

blocker_d2m = StandardBlocker(
    dbpedia,
    metacritic,
    on=["name_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2s = StandardBlocker(
    dbpedia,
    sales,
    on=["name_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_m2s = StandardBlocker(
    metacritic,
    sales,
    on=["name_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching configuration (rule-based)
# Strengthen developer matching by normalization; keep year/platform as disambiguators.
# --------------------------------

comparators_d2m = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="platform_norm", similarity_function="jaccard"),
    NumericComparator(column="release_year_num", max_difference=1),
    StringComparator(column="developer_norm", similarity_function="jaccard"),
]

comparators_d2s = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="platform_norm", similarity_function="jaccard"),
    NumericComparator(column="release_year_num", max_difference=1),
    StringComparator(column="developer_norm", similarity_function="jaccard"),
]

comparators_m2s = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="platform_norm", similarity_function="jaccard"),
    NumericComparator(column="release_year_num", max_difference=1),
    StringComparator(column="developer_norm", similarity_function="jaccard"),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_d2m = rb_matcher.match(
    df_left=dbpedia,
    df_right=metacritic,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.65, 0.15, 0.10, 0.10],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_d2s = rb_matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_d2s,
    comparators=comparators_d2s,
    weights=[0.65, 0.15, 0.10, 0.10],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_m2s = rb_matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_m2s,
    comparators=comparators_m2s,
    weights=[0.65, 0.15, 0.10, 0.10],
    threshold=0.78,
    id_column="id",
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d2m, rb_correspondences_d2s, rb_correspondences_m2s],
    ignore_index=True,
)

# --------------------------------
# Data Fusion
# Fix low accuracies:
# - developer: prefer higher-trust source instead of longest_string
# - userScore: avoid taking maximum (often wrong); prefer higher-trust and normalized numeric
# --------------------------------

# Trust: metacritic > sales > dbpedia for scores/ESRB; sales > metacritic > dbpedia for publisher/sales.
SOURCE_TRUST = {"metacritic": 0.95, "sales": 0.9, "dbpedia": 0.7}

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Base schema (dbpedia)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", longest_string)

# Prefer trusted source for org fields to fix mismatches
strategy.add_attribute_fuser("developer", prefer_higher_trust, trust_map=SOURCE_TRUST)
strategy.add_attribute_fuser("publisher", prefer_higher_trust, trust_map=SOURCE_TRUST)

strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)

# Ratings/scores: prefer trusted source; keep numeric helpers deterministic
strategy.add_attribute_fuser("ESRB", prefer_higher_trust, trust_map=SOURCE_TRUST)

strategy.add_attribute_fuser("criticScore", prefer_higher_trust, trust_map=SOURCE_TRUST)
strategy.add_attribute_fuser("userScore", prefer_higher_trust, trust_map=SOURCE_TRUST)

strategy.add_attribute_fuser("globalSales", prefer_higher_trust, trust_map=SOURCE_TRUST)

# Helper/normalized columns
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("platform_norm", longest_string)
strategy.add_attribute_fuser("developer_norm", prefer_higher_trust, trust_map=SOURCE_TRUST)
strategy.add_attribute_fuser("publisher_norm", prefer_higher_trust, trust_map=SOURCE_TRUST)
strategy.add_attribute_fuser("ESRB_norm", prefer_higher_trust, trust_map=SOURCE_TRUST)

strategy.add_attribute_fuser("release_year_num", maximum)
strategy.add_attribute_fuser("critic_score_num", maximum)
strategy.add_attribute_fuser("user_score_num", maximum)
strategy.add_attribute_fuser("global_sales_num", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=13
node_name=execute_pipeline
accuracy_score=84.87%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    maximum,
    prefer_higher_trust,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/games/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

dbpedia = load_xml(DATA_DIR + "dbpedia.xml", name="dbpedia")
metacritic = load_xml(DATA_DIR + "metacritic.xml", name="metacritic")
sales = load_xml(DATA_DIR + "sales.xml", name="sales")

# --------------------------------
# Normalization helpers
# --------------------------------

_CORP_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "gmbh", "sarl", "sa", "ag", "bv", "plc", "pte", "kg", "oy", "ab", "spa", "s", "p", "a"
}

_DEV_PUB_SYNONYMS = {
    "nintendo ead": "nintendo",
    "nintendo ead tokyo": "nintendo",
    "sony computer entertainment": "sony",
    "sony interactive entertainment": "sony",
    "microsoft game studios": "microsoft",
    "microsoft studios": "microsoft",
    "electronic arts": "ea",
    "ea games": "ea",
    "rockstar games": "rockstar",
    "rockstar north": "rockstar",
    "activision publishing": "activision",
    "square enix": "square enix",
    "squareenix": "square enix",
    "take two interactive": "take two",
    "take two interactive software": "take two",
    "warner bros interactive": "warner bros",
    "warner bros interactive entertainment": "warner bros",
}

_PLATFORM_SYNONYMS = {
    "playstation 4": "ps4",
    "ps 4": "ps4",
    "ps4": "ps4",
    "playstation 5": "ps5",
    "ps 5": "ps5",
    "ps5": "ps5",
    "playstation 3": "ps3",
    "ps 3": "ps3",
    "ps3": "ps3",
    "playstation 2": "ps2",
    "ps 2": "ps2",
    "ps2": "ps2",
    "playstation": "ps1",
    "ps 1": "ps1",
    "ps1": "ps1",
    "xbox one": "xbox one",
    "xboxone": "xbox one",
    "xbox series x": "xbox series",
    "xbox series s": "xbox series",
    "xbox series": "xbox series",
    "xbox 360": "xbox 360",
    "nintendo switch": "switch",
    "switch": "switch",
    "wii u": "wii u",
    "wiiu": "wii u",
    "wii": "wii",
    "nintendo 3ds": "3ds",
    "3ds": "3ds",
    "nintendo ds": "ds",
    "ds": "ds",
    "game boy color": "gbc",
    "gameboy color": "gbc",
    "gbc": "gbc",
    "game boy advance": "gba",
    "gameboy advance": "gba",
    "gba": "gba",
    "gamecube": "gcn",
    "nintendo gamecube": "gcn",
    "gcn": "gcn",
    "pc": "pc",
    "windows": "pc",
    "mac": "mac",
    "ios": "ios",
    "android": "android",
}

_EDITION_STOPWORDS = {
    "edition", "definitive", "ultimate", "goty", "game", "of", "the", "year",
    "remastered", "hd", "complete", "collection", "bundle", "deluxe", "standard",
    "enhanced", "anniversary"
}

def _is_null(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))

def _to_str(x) -> str:
    if _is_null(x):
        return ""
    return str(x)

def normalize_name(x: object) -> str:
    s = _to_str(x).lower().strip()
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_name_loose(x: object) -> str:
    """
    Looser canonicalization to improve matching across editions:
    removes common edition tokens while keeping discriminative terms.
    """
    s = normalize_name(x)
    if not s:
        return ""
    toks = [t for t in s.split() if t not in _EDITION_STOPWORDS]
    # keep at least something
    if not toks:
        toks = s.split()
    return " ".join(toks).strip()

def year_from_date(x: object):
    s = _to_str(x).strip()
    if not s:
        return np.nan
    m = re.match(r"^(\d{4})", s)
    if not m:
        return np.nan
    return float(m.group(1))

def normalize_platform(x: object) -> str:
    s = _to_str(x).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _PLATFORM_SYNONYMS.get(s, s)

def normalize_org(x: object) -> str:
    s = _to_str(x).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    s = _DEV_PUB_SYNONYMS.get(s, s)
    parts = [p for p in s.split() if p not in _CORP_SUFFIXES]
    s2 = " ".join(parts).strip()
    return _DEV_PUB_SYNONYMS.get(s2, s2)

def normalize_esrb(x: object) -> str:
    s = _to_str(x).upper().strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = s.replace("RATING PENDING", "RP")
    s = s.replace("MATURE", "M")
    s = s.replace("EVERYONE", "E")
    s = s.replace("TEEN", "T")
    s = s.replace("ADULTS ONLY", "AO")
    s = re.sub(r"[^A-Z0-9\+]+", "", s)

    if s in {"E10PLUS", "E10"}:
        return "E10+"
    if s in {"EC", "E", "T", "M", "AO", "RP", "E10+"}:
        return s
    return s

def to_float(x: object):
    s = _to_str(x).strip()
    if not s:
        return np.nan
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.]+", "", s)
    if s == "" or s == ".":
        return np.nan
    # if multiple dots, keep first and remove the rest
    if s.count(".") > 1:
        first = s.find(".")
        s = s[: first + 1] + s[first + 1 :].replace(".", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_user_score(x: object):
    """
    Normalize user score to 0-10 float.
    Handles 0-100 encodings and "tbd"/missing.
    """
    s_raw = _to_str(x).strip().lower()
    if s_raw in {"", "tbd", "na", "n/a", "none", "null"}:
        return np.nan
    v = to_float(x)
    if np.isnan(v):
        return np.nan
    # heuristics: metacritic user score is typically 0-10; other sources might have 0-100
    if v > 10.0 and v <= 100.0:
        v = v / 10.0
    # bound to plausible range
    if v < 0.0 or v > 10.0:
        return np.nan
    return float(v)

def normalize_critic_score(x: object):
    """
    Normalize critic score to 0-100 float.
    """
    s_raw = _to_str(x).strip().lower()
    if s_raw in {"", "tbd", "na", "n/a", "none", "null"}:
        return np.nan
    v = to_float(x)
    if np.isnan(v):
        return np.nan
    if v <= 10.0:
        v = v * 10.0
    if v < 0.0 or v > 100.0:
        return np.nan
    return float(v)

def normalize_global_sales(x: object):
    v = to_float(x)
    if np.isnan(v) or v < 0:
        return np.nan
    return float(v)

def stringify_score_critic(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        return str(int(round(float(v))))
    except Exception:
        return ""

def stringify_score_user(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        return f"{float(v):.1f}".rstrip("0").rstrip(".")
    except Exception:
        return ""

# Add helper columns before schema matching (safe) and re-check after renaming (below)
for df in (dbpedia, metacritic, sales):
    if "name" in df.columns:
        df["name_norm"] = df["name"].map(normalize_name)
        df["name_loose"] = df["name"].map(normalize_name_loose)
    else:
        df["name_norm"] = ""
        df["name_loose"] = ""

    df["platform_norm"] = df["platform"].map(normalize_platform) if "platform" in df.columns else ""
    df["release_year_num"] = df["releaseYear"].map(year_from_date) if "releaseYear" in df.columns else np.nan

    df["developer_norm"] = df["developer"].map(normalize_org) if "developer" in df.columns else ""
    df["publisher_norm"] = df["publisher"].map(normalize_org) if "publisher" in df.columns else ""

    df["ESRB_norm"] = df["ESRB"].map(normalize_esrb) if "ESRB" in df.columns else ""

    df["critic_score_num"] = df["criticScore"].map(normalize_critic_score) if "criticScore" in df.columns else np.nan
    df["user_score_num"] = df["userScore"].map(normalize_user_score) if "userScore" in df.columns else np.nan
    df["global_sales_num"] = df["globalSales"].map(normalize_global_sales) if "globalSales" in df.columns else np.nan

    # canonical string versions for fusion output consistency
    df["criticScore_str"] = df["critic_score_num"].map(stringify_score_critic)
    df["userScore_str"] = df["user_score_num"].map(stringify_score_user)

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# Resulting columns for all datasets will have the schema of dbpedia.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

schema_correspondences = schema_matcher.match(dbpedia, metacritic)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metacritic = metacritic.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(dbpedia, sales)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
sales = sales.rename(columns=rename_map)

# Ensure helper columns exist after renaming (schema matching may affect column names)
for df in (metacritic, sales):
    if "name_norm" not in df.columns and "name" in df.columns:
        df["name_norm"] = df["name"].map(normalize_name)
    if "name_loose" not in df.columns and "name" in df.columns:
        df["name_loose"] = df["name"].map(normalize_name_loose)
    if "platform_norm" not in df.columns and "platform" in df.columns:
        df["platform_norm"] = df["platform"].map(normalize_platform)
    if "release_year_num" not in df.columns and "releaseYear" in df.columns:
        df["release_year_num"] = df["releaseYear"].map(year_from_date)
    if "developer_norm" not in df.columns and "developer" in df.columns:
        df["developer_norm"] = df["developer"].map(normalize_org)
    if "publisher_norm" not in df.columns:
        df["publisher_norm"] = df["publisher"].map(normalize_org) if "publisher" in df.columns else ""
    if "ESRB_norm" not in df.columns:
        df["ESRB_norm"] = df["ESRB"].map(normalize_esrb) if "ESRB" in df.columns else ""
    if "critic_score_num" not in df.columns:
        df["critic_score_num"] = df["criticScore"].map(normalize_critic_score) if "criticScore" in df.columns else np.nan
    if "user_score_num" not in df.columns:
        df["user_score_num"] = df["userScore"].map(normalize_user_score) if "userScore" in df.columns else np.nan
    if "global_sales_num" not in df.columns:
        df["global_sales_num"] = df["globalSales"].map(normalize_global_sales) if "globalSales" in df.columns else np.nan

    if "criticScore_str" not in df.columns:
        df["criticScore_str"] = df["critic_score_num"].map(stringify_score_critic)
    if "userScore_str" not in df.columns:
        df["userScore_str"] = df["user_score_num"].map(stringify_score_user)

# --------------------------------
# Blocking (MANDATORY POLICY)
# Use informative identity signals: canonical title/name (+ optional disambiguator).
# Here: name_loose improves recall across editions/variants, still a strong identity signal.
# --------------------------------

print("Performing Blocking")

blocker_d2m = StandardBlocker(
    dbpedia,
    metacritic,
    on=["name_loose"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2s = StandardBlocker(
    dbpedia,
    sales,
    on=["name_loose"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_m2s = StandardBlocker(
    metacritic,
    sales,
    on=["name_loose"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching configuration (rule-based)
# Boost platform correctness by increasing platform weight and using normalized platform.
# Keep year as disambiguator (±1).
# --------------------------------

comparators_d2m = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="platform_norm", similarity_function="jaccard"),
    NumericComparator(column="release_year_num", max_difference=1),
    StringComparator(column="developer_norm", similarity_function="jaccard"),
]

comparators_d2s = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="platform_norm", similarity_function="jaccard"),
    NumericComparator(column="release_year_num", max_difference=1),
    StringComparator(column="developer_norm", similarity_function="jaccard"),
]

comparators_m2s = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="platform_norm", similarity_function="jaccard"),
    NumericComparator(column="release_year_num", max_difference=1),
    StringComparator(column="developer_norm", similarity_function="jaccard"),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

# weights: make platform a stronger disambiguator to improve platform accuracy,
# while keeping name dominant for identity
W = [0.60, 0.25, 0.10, 0.05]
TH = 0.80

rb_correspondences_d2m = rb_matcher.match(
    df_left=dbpedia,
    df_right=metacritic,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=W,
    threshold=TH,
    id_column="id",
)

rb_correspondences_d2s = rb_matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_d2s,
    comparators=comparators_d2s,
    weights=W,
    threshold=TH,
    id_column="id",
)

rb_correspondences_m2s = rb_matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_m2s,
    comparators=comparators_m2s,
    weights=W,
    threshold=TH,
    id_column="id",
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d2m, rb_correspondences_d2s, rb_correspondences_m2s],
    ignore_index=True,
)

# --------------------------------
# Data Fusion
# Fix low accuracies:
# - platform: prefer higher-trust (metacritic/sales) over dbpedia
# - userScore: prefer metacritic; avoid max; use normalized string version for output
# - criticScore: prefer metacritic/sales; avoid max
# --------------------------------

# Trust: metacritic best for platform/scores/ESRB/userScore; sales best for publisher/globalSales.
SOURCE_TRUST = {"metacritic": 0.97, "sales": 0.92, "dbpedia": 0.75}

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Base schema (dbpedia)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", prefer_higher_trust, trust_map=SOURCE_TRUST)

# Organizations
strategy.add_attribute_fuser("developer", prefer_higher_trust, trust_map=SOURCE_TRUST)
strategy.add_attribute_fuser("publisher", prefer_higher_trust, trust_map={"sales": 0.98, "metacritic": 0.85, "dbpedia": 0.70})

# Platform: prefer trusted sources (improves platform accuracy)
strategy.add_attribute_fuser("platform", prefer_higher_trust, trust_map=SOURCE_TRUST)

# Series mainly from dbpedia
strategy.add_attribute_fuser("series", longest_string)

# Ratings/scores: prefer trusted sources; also keep normalized numeric/string helpers
strategy.add_attribute_fuser("ESRB", prefer_higher_trust, trust_map=SOURCE_TRUST)

# Output score fields: use canonical string columns if present
strategy.add_attribute_fuser("criticScore", prefer_higher_trust, trust_map={"metacritic": 0.99, "sales": 0.93, "dbpedia": 0.60})
strategy.add_attribute_fuser("userScore", prefer_higher_trust, trust_map={"metacritic": 0.99, "sales": 0.90, "dbpedia": 0.60})

strategy.add_attribute_fuser("globalSales", prefer_higher_trust, trust_map={"sales": 0.99, "metacritic": 0.50, "dbpedia": 0.30})

# Helper/normalized columns
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("name_loose", longest_string)
strategy.add_attribute_fuser("platform_norm", prefer_higher_trust, trust_map=SOURCE_TRUST)
strategy.add_attribute_fuser("developer_norm", prefer_higher_trust, trust_map=SOURCE_TRUST)
strategy.add_attribute_fuser("publisher_norm", prefer_higher_trust, trust_map={"sales": 0.98, "metacritic": 0.85, "dbpedia": 0.70})
strategy.add_attribute_fuser("ESRB_norm", prefer_higher_trust, trust_map=SOURCE_TRUST)

# Numeric helpers: keep maximum for year/sales only; scores should not take maximum
strategy.add_attribute_fuser("release_year_num", maximum)
strategy.add_attribute_fuser("global_sales_num", maximum)

strategy.add_attribute_fuser("critic_score_num", prefer_higher_trust, trust_map={"metacritic": 0.99, "sales": 0.93, "dbpedia": 0.60})
strategy.add_attribute_fuser("user_score_num", prefer_higher_trust, trust_map={"metacritic": 0.99, "sales": 0.90, "dbpedia": 0.60})

# Canonical score strings (deterministic formatting for evaluation)
strategy.add_attribute_fuser("criticScore_str", prefer_higher_trust, trust_map={"metacritic": 0.99, "sales": 0.93, "dbpedia": 0.60})
strategy.add_attribute_fuser("userScore_str", prefer_higher_trust, trust_map={"metacritic": 0.99, "sales": 0.90, "dbpedia": 0.60})

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# Post-fusion: if criticScore/userScore are empty but *_str exists, fill for output consistency
if "criticScore" in rb_fused_standard_blocker.columns and "criticScore_str" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["criticScore"] = rb_fused_standard_blocker["criticScore"].fillna("")
    rb_fused_standard_blocker.loc[
        (rb_fused_standard_blocker["criticScore"].astype(str).str.strip() == "")
        & (rb_fused_standard_blocker["criticScore_str"].astype(str).str.strip() != ""),
        "criticScore",
    ] = rb_fused_standard_blocker["criticScore_str"]

if "userScore" in rb_fused_standard_blocker.columns and "userScore_str" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["userScore"] = rb_fused_standard_blocker["userScore"].fillna("")
    rb_fused_standard_blocker.loc[
        (rb_fused_standard_blocker["userScore"].astype(str).str.strip() == "")
        & (rb_fused_standard_blocker["userScore_str"].astype(str).str.strip() != ""),
        "userScore",
    ] = rb_fused_standard_blocker["userScore_str"]

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

