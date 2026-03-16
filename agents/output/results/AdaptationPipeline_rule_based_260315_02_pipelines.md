# Pipeline Snapshots

notebook_name=AdaptationPipeline
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=3
node_name=execute_pipeline
accuracy_score=61.34%
------------------------------------------------------------

```python
from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/games/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_xml(
    DATA_DIR + "dbpedia.xml",
    name="dbpedia",
)

good_dataset_name_2 = load_xml(
    DATA_DIR + "metacritic.xml",
    name="metacritic",
)

good_dataset_name_3 = load_xml(
    DATA_DIR + "sales.xml",
    name="sales",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# --------------------------------
# Lightweight normalization for blocking/matching
# --------------------------------

def normalize_name(value):
    if pd.isna(value):
        return value
    return (
        str(value)
        .lower()
        .strip()
        .replace(":", " ")
        .replace("-", " ")
        .replace("_", " ")
    )

def extract_year(value):
    if pd.isna(value):
        return value
    s = str(value).strip()
    return s[:4] if len(s) >= 4 else s

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "name" in df.columns:
        df["name_norm"] = df["name"].apply(normalize_name)
    if "releaseYear" in df.columns:
        df["releaseYear_norm"] = df["releaseYear"].apply(extract_year)
    if "criticScore" in df.columns:
        df["criticScore_num"] = pd.to_numeric(df["criticScore"], errors="coerce")
    if "userScore" in df.columns:
        df["userScore_num"] = pd.to_numeric(df["userScore"], errors="coerce")
    if "globalSales" in df.columns:
        df["globalSales_num"] = pd.to_numeric(df["globalSales"], errors="coerce")

# --------------------------------
# Blocking
# Use informative identity signal: canonical game name
# --------------------------------

print("Performing Blocking")

blocker_d2m = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    on=["name_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2s = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["name_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_m2s = StandardBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    on=["name_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching
# Use shared entity signals: name, developer, platform, release year
# --------------------------------

comparators_d2m = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="platform",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="releaseYear_norm",
        max_difference=1,
    ),
]

comparators_d2s = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="platform",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="releaseYear_norm",
        max_difference=1,
    ),
]

comparators_m2s = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="platform",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="releaseYear_norm",
        max_difference=1,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_d2m = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.55, 0.2, 0.15, 0.1],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_d2s = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_d2s,
    comparators=comparators_d2s,
    weights=[0.55, 0.2, 0.15, 0.1],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_m2s = matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_m2s,
    comparators=comparators_m2s,
    weights=[0.55, 0.2, 0.15, 0.1],
    threshold=0.75,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_d2m.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_dbpedia_metacritic.csv",
    ),
    index=False,
)

rb_correspondences_d2s.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_dbpedia_sales.csv",
    ),
    index=False,
)

rb_correspondences_m2s.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metacritic_sales.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d2m, rb_correspondences_d2s, rb_correspondences_m2s],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", longest_string)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("criticScore", maximum)
strategy.add_attribute_fuser("userScore", maximum)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("globalSales", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

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
accuracy_score=84.87%
------------------------------------------------------------

```python
from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import re
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/games/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_xml(
    DATA_DIR + "dbpedia.xml",
    name="dbpedia",
)

good_dataset_name_2 = load_xml(
    DATA_DIR + "metacritic.xml",
    name="metacritic",
)

good_dataset_name_3 = load_xml(
    DATA_DIR + "sales.xml",
    name="sales",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# --------------------------------
# Normalization helpers
# --------------------------------

def normalize_text(value):
    if pd.isna(value):
        return None
    s = str(value).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def normalize_name(value):
    s = normalize_text(value)
    if s is None:
        return None
    s = s.replace("game of the year edition", "")
    s = s.replace("goty edition", "")
    s = s.replace("special edition", "")
    s = s.replace("collector s edition", "")
    s = s.replace("collectors edition", "")
    s = s.replace("definitive edition", "")
    s = s.replace("complete edition", "")
    s = s.replace("remastered", "")
    s = s.replace("hd", "hd")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def normalize_platform(value):
    s = normalize_text(value)
    if s is None:
        return None
    platform_map = {
        "playstation 4": "ps4",
        "ps4": "ps4",
        "playstation 3": "ps3",
        "ps3": "ps3",
        "playstation 2": "ps2",
        "ps2": "ps2",
        "playstation": "ps",
        "playstation vita": "ps vita",
        "ps vita": "ps vita",
        "playstation portable": "psp",
        "psp": "psp",
        "xbox one": "xbox one",
        "xbox 360": "xbox 360",
        "xbox series x": "xbox series x",
        "xbox series s": "xbox series s",
        "nintendo switch": "switch",
        "switch": "switch",
        "wii u": "wii u",
        "wii": "wii",
        "gamecube": "gamecube",
        "nintendo gamecube": "gamecube",
        "game boy advance": "gba",
        "gba": "gba",
        "game boy color": "gbc",
        "gbc": "gbc",
        "game boy": "gb",
        "gb": "gb",
        "nintendo ds": "ds",
        "ds": "ds",
        "nintendo 3ds": "3ds",
        "3ds": "3ds",
        "pc": "pc",
        "windows": "pc",
        "mac": "mac",
        "ios": "ios",
        "android": "android",
        "dreamcast": "dreamcast",
    }
    return platform_map.get(s, s)

def normalize_company(value):
    s = normalize_text(value)
    if s is None:
        return None
    s = re.sub(r"\bincorporated\b", "inc", s)
    s = re.sub(r"\bcorporation\b", "corp", s)
    s = re.sub(r"\bcompany\b", "co", s)
    s = re.sub(r"\blimited\b", "ltd", s)
    s = re.sub(r"\binteractive\b", "", s)
    s = re.sub(r"\bentertainment\b", "", s)
    s = re.sub(r"\bstudios\b", "studio", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def extract_year(value):
    if pd.isna(value):
        return None
    m = re.search(r"(\d{4})", str(value))
    if not m:
        return None
    return pd.to_numeric(m.group(1), errors="coerce")

def to_float(value):
    return pd.to_numeric(value, errors="coerce")

def prefer_non_null(values):
    vals = [v for v in values if pd.notna(v) and str(v).strip() != ""]
    if not vals:
        return None
    return vals[0]

def most_common_string(values):
    vals = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() != ""]
    if not vals:
        return None
    counts = pd.Series(vals).value_counts()
    max_count = counts.iloc[0]
    candidates = counts[counts == max_count].index.tolist()
    return max(candidates, key=len)

def average_numeric_as_string(values):
    nums = [pd.to_numeric(v, errors="coerce") for v in values]
    nums = [v for v in nums if pd.notna(v)]
    if not nums:
        return None
    avg = sum(nums) / len(nums)
    if float(avg).is_integer():
        return str(int(avg))
    return str(round(avg, 1))

def max_numeric_as_string(values):
    nums = [pd.to_numeric(v, errors="coerce") for v in values]
    nums = [v for v in nums if pd.notna(v)]
    if not nums:
        return None
    mx = max(nums)
    if float(mx).is_integer():
        return str(int(mx))
    return str(round(mx, 1))

def min_numeric_as_string(values):
    nums = [pd.to_numeric(v, errors="coerce") for v in values]
    nums = [v for v in nums if pd.notna(v)]
    if not nums:
        return None
    mn = min(nums)
    if float(mn).is_integer():
        return str(int(mn))
    return str(round(mn, 1))

def best_user_score(values):
    nums = [pd.to_numeric(v, errors="coerce") for v in values]
    nums = [v for v in nums if pd.notna(v)]
    if not nums:
        return None
    in_expected_range = [v for v in nums if 0 <= v <= 10]
    candidates = in_expected_range if in_expected_range else nums
    avg = sum(candidates) / len(candidates)
    return str(round(avg, 1))

def best_critic_score(values):
    nums = [pd.to_numeric(v, errors="coerce") for v in values]
    nums = [v for v in nums if pd.notna(v)]
    if not nums:
        return None
    in_expected_range = [v for v in nums if 0 <= v <= 100]
    candidates = in_expected_range if in_expected_range else nums
    avg = sum(candidates) / len(candidates)
    return str(int(round(avg)))

def best_platform(values):
    vals = [normalize_platform(v) for v in values if pd.notna(v) and str(v).strip() != ""]
    vals = [v for v in vals if v]
    if not vals:
        return None
    counts = pd.Series(vals).value_counts()
    max_count = counts.iloc[0]
    candidates = counts[counts == max_count].index.tolist()
    return max(candidates, key=len)

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "name" in df.columns:
        df["name_norm"] = df["name"].apply(normalize_name)
    if "releaseYear" in df.columns:
        df["releaseYear_num"] = df["releaseYear"].apply(extract_year)
    if "developer" in df.columns:
        df["developer_norm"] = df["developer"].apply(normalize_company)
    if "publisher" in df.columns:
        df["publisher_norm"] = df["publisher"].apply(normalize_company)
    if "platform" in df.columns:
        df["platform_norm"] = df["platform"].apply(normalize_platform)
    if "criticScore" in df.columns:
        df["criticScore_num"] = df["criticScore"].apply(to_float)
    if "userScore" in df.columns:
        df["userScore_num"] = df["userScore"].apply(to_float)
    if "globalSales" in df.columns:
        df["globalSales_num"] = df["globalSales"].apply(to_float)

# --------------------------------
# Blocking
# Use strong identity signals: normalized title + release year
# --------------------------------

print("Performing Blocking")

blocker_d2m = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    on=["name_norm", "releaseYear_num"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2s = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["name_norm", "releaseYear_num"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_m2s = StandardBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    on=["name_norm", "releaseYear_num"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching
# Rely mostly on title, year, platform, developer
# --------------------------------

comparators_d2m = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="releaseYear_num",
        max_difference=0,
    ),
    StringComparator(
        column="platform_norm",
        similarity_function="jaccard",
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
    NumericComparator(
        column="releaseYear_num",
        max_difference=0,
    ),
    StringComparator(
        column="platform_norm",
        similarity_function="jaccard",
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
    NumericComparator(
        column="releaseYear_num",
        max_difference=0,
    ),
    StringComparator(
        column="platform_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="developer_norm",
        similarity_function="jaccard",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_d2m = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.6, 0.15, 0.15, 0.1],
    threshold=0.82,
    id_column="id",
)

rb_correspondences_d2s = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_d2s,
    comparators=comparators_d2s,
    weights=[0.6, 0.15, 0.15, 0.1],
    threshold=0.82,
    id_column="id",
)

rb_correspondences_m2s = matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_m2s,
    comparators=comparators_m2s,
    weights=[0.6, 0.15, 0.15, 0.1],
    threshold=0.82,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_d2m.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_dbpedia_metacritic.csv",
    ),
    index=False,
)

rb_correspondences_d2s.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_dbpedia_sales.csv",
    ),
    index=False,
)

rb_correspondences_m2s.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metacritic_sales.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d2m, rb_correspondences_d2s, rb_correspondences_m2s],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", prefer_non_null)
strategy.add_attribute_fuser("developer", most_common_string)
strategy.add_attribute_fuser("platform", best_platform)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("criticScore", best_critic_score)
strategy.add_attribute_fuser("userScore", best_user_score)
strategy.add_attribute_fuser("ESRB", most_common_string)
strategy.add_attribute_fuser("publisher", most_common_string)
strategy.add_attribute_fuser("globalSales", max_numeric_as_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
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
accuracy_score=59.66%
------------------------------------------------------------

```python
from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import re
from collections import Counter
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/games/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_xml(
    DATA_DIR + "dbpedia.xml",
    name="dbpedia",
)

good_dataset_name_2 = load_xml(
    DATA_DIR + "metacritic.xml",
    name="metacritic",
)

good_dataset_name_3 = load_xml(
    DATA_DIR + "sales.xml",
    name="sales",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# --------------------------------
# Normalization helpers
# --------------------------------

def normalize_text(value):
    if pd.isna(value):
        return None
    s = str(value).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[:/\\|,_\-]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def normalize_name(value):
    s = normalize_text(value)
    if s is None:
        return None

    removable_phrases = [
        "game of the year edition",
        "goty edition",
        "special edition",
        "collector s edition",
        "collectors edition",
        "definitive edition",
        "complete edition",
        "limited edition",
        "ultimate edition",
        "gold edition",
        "platinum edition",
        "remastered",
        "edition",
    ]
    for phrase in removable_phrases:
        s = re.sub(rf"\b{re.escape(phrase)}\b", " ", s)

    roman_map = {
        " ii ": " 2 ",
        " iii ": " 3 ",
        " iv ": " 4 ",
        " v ": " 5 ",
        " vi ": " 6 ",
        " vii ": " 7 ",
        " viii ": " 8 ",
        " ix ": " 9 ",
        " x ": " 10 ",
    }
    s = f" {s} "
    for k, v in roman_map.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def blocking_name(value):
    s = normalize_name(value)
    if s is None:
        return None
    s = re.sub(r"\bthe\b", " ", s)
    s = re.sub(r"\band\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def normalize_platform(value):
    s = normalize_text(value)
    if s is None:
        return None
    platform_map = {
        "playstation 5": "ps5",
        "ps5": "ps5",
        "playstation 4": "ps4",
        "ps4": "ps4",
        "playstation 3": "ps3",
        "ps3": "ps3",
        "playstation 2": "ps2",
        "ps2": "ps2",
        "playstation": "ps",
        "playstation vita": "ps vita",
        "ps vita": "ps vita",
        "vita": "ps vita",
        "playstation portable": "psp",
        "psp": "psp",
        "xbox one": "xbox one",
        "xbox 360": "xbox 360",
        "xbox": "xbox",
        "xbox series x": "xbox series x",
        "xbox series s": "xbox series s",
        "nintendo switch": "switch",
        "switch": "switch",
        "wii u": "wii u",
        "wii": "wii",
        "gamecube": "gamecube",
        "nintendo gamecube": "gamecube",
        "game boy advance": "gba",
        "gba": "gba",
        "game boy color": "gbc",
        "gbc": "gbc",
        "game boy": "gb",
        "gb": "gb",
        "nintendo ds": "ds",
        "ds": "ds",
        "nintendo 3ds": "3ds",
        "3ds": "3ds",
        "pc": "pc",
        "windows": "pc",
        "microsoft windows": "pc",
        "mac": "mac",
        "ios": "ios",
        "android": "android",
        "dreamcast": "dreamcast",
    }
    return platform_map.get(s, s)

def normalize_company(value):
    s = normalize_text(value)
    if s is None:
        return None
    s = re.sub(r"\bincorporated\b", "inc", s)
    s = re.sub(r"\bcorporation\b", "corp", s)
    s = re.sub(r"\bcompany\b", "co", s)
    s = re.sub(r"\blimited\b", "ltd", s)
    s = re.sub(r"\binteractive\b", "", s)
    s = re.sub(r"\bentertainment\b", "", s)
    s = re.sub(r"\bstudios\b", "studio", s)
    s = re.sub(r"\bstudioes\b", "studio", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def normalize_esrb(value):
    s = normalize_text(value)
    if s is None:
        return None
    mapping = {
        "e": "E",
        "everyone": "E",
        "e10": "E10+",
        "e10+": "E10+",
        "everyone 10+": "E10+",
        "t": "T",
        "teen": "T",
        "m": "M",
        "mature": "M",
        "mature 17+": "M",
        "ao": "AO",
        "adults only": "AO",
        "rp": "RP",
        "rating pending": "RP",
        "ec": "EC",
        "early childhood": "EC",
        "k a": "KA",
        "ka": "KA",
    }
    return mapping.get(s, str(value).strip())

def extract_year(value):
    if pd.isna(value):
        return None
    m = re.search(r"(\d{4})", str(value))
    if not m:
        return None
    return pd.to_numeric(m.group(1), errors="coerce")

def year_string(value):
    y = extract_year(value)
    if pd.isna(y):
        return None
    return str(int(y))

def to_float(value):
    return pd.to_numeric(value, errors="coerce")

def prefer_non_null(values):
    vals = [v for v in values if pd.notna(v) and str(v).strip() != ""]
    if not vals:
        return None
    return vals[0]

def canonical_from_norm(values, normalizer=None):
    raw_vals = [v for v in values if pd.notna(v) and str(v).strip() != ""]
    if not raw_vals:
        return None
    if normalizer is None:
        counts = Counter([str(v).strip() for v in raw_vals])
        best = max(counts.items(), key=lambda x: (x[1], len(x[0])))
        return best[0]

    grouped = {}
    for v in raw_vals:
        raw = str(v).strip()
        norm = normalizer(raw)
        key = norm if norm else raw.lower()
        grouped.setdefault(key, []).append(raw)

    best_key, best_group = max(grouped.items(), key=lambda x: (len(x[1]), max(len(v) for v in x[1])))
    counts = Counter(best_group)
    best_raw = max(counts.items(), key=lambda x: (x[1], len(x[0])))
    return best_raw[0]

def best_platform(values):
    raw_vals = [v for v in values if pd.notna(v) and str(v).strip() != ""]
    if not raw_vals:
        return None

    grouped = {}
    for v in raw_vals:
        raw = str(v).strip()
        norm = normalize_platform(raw)
        key = norm if norm else raw.lower()
        grouped.setdefault(key, []).append(raw)

    best_key, best_group = max(grouped.items(), key=lambda x: (len(x[1]), max(len(v) for v in x[1])))
    counts = Counter(best_group)
    best_raw = max(counts.items(), key=lambda x: (x[1], len(x[0])))
    return best_raw[0]

def best_user_score(values):
    nums = []
    raw_strings = []
    for v in values:
        if pd.isna(v) or str(v).strip() == "":
            continue
        raw = str(v).strip()
        num = pd.to_numeric(raw, errors="coerce")
        if pd.notna(num) and 0 <= num <= 10:
            nums.append(float(num))
            raw_strings.append(raw)

    if not nums:
        return None

    rounded_1 = [round(v, 1) for v in nums]
    counts = Counter(rounded_1)
    most_common_value, freq = max(counts.items(), key=lambda x: (x[1], -abs(x[0] - sum(nums) / len(nums))))
    if freq >= 2:
        return f"{most_common_value:.1f}"

    avg = sum(nums) / len(nums)
    return f"{round(avg, 1):.1f}"

def best_critic_score(values):
    nums = []
    for v in values:
        if pd.isna(v) or str(v).strip() == "":
            continue
        num = pd.to_numeric(v, errors="coerce")
        if pd.notna(num) and 0 <= num <= 100:
            nums.append(float(num))

    if not nums:
        return None

    rounded_int = [int(round(v)) for v in nums]
    counts = Counter(rounded_int)
    most_common_value, freq = max(counts.items(), key=lambda x: (x[1], -abs(x[0] - sum(nums) / len(nums))))
    if freq >= 2:
        return str(most_common_value)

    return str(int(round(sum(nums) / len(nums))))

def max_numeric_as_string(values):
    nums = [pd.to_numeric(v, errors="coerce") for v in values]
    nums = [v for v in nums if pd.notna(v)]
    if not nums:
        return None
    mx = max(nums)
    if float(mx).is_integer():
        return str(int(mx))
    return str(round(mx, 1))

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "name" in df.columns:
        df["name_norm"] = df["name"].apply(normalize_name)
        df["name_block"] = df["name"].apply(blocking_name)
    if "releaseYear" in df.columns:
        df["releaseYear_num"] = df["releaseYear"].apply(extract_year)
        df["releaseYear_clean"] = df["releaseYear"].apply(year_string)
    if "developer" in df.columns:
        df["developer_norm"] = df["developer"].apply(normalize_company)
    if "publisher" in df.columns:
        df["publisher_norm"] = df["publisher"].apply(normalize_company)
    if "platform" in df.columns:
        df["platform_norm"] = df["platform"].apply(normalize_platform)
    if "criticScore" in df.columns:
        df["criticScore_num"] = df["criticScore"].apply(to_float)
    if "userScore" in df.columns:
        df["userScore_num"] = df["userScore"].apply(to_float)
    if "ESRB" in df.columns:
        df["ESRB_norm"] = df["ESRB"].apply(normalize_esrb)
    if "globalSales" in df.columns:
        df["globalSales_num"] = df["globalSales"].apply(to_float)

# --------------------------------
# Blocking
# Use strongest identity signal available: title
# --------------------------------

print("Performing Blocking")

blocker_d2m = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    on=["name_block"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2s = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["name_block"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_m2s = StandardBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    on=["name_block"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching
# Stronger emphasis on title and platform, year slightly relaxed
# --------------------------------

comparators_d2m = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="releaseYear_num",
        max_difference=1,
    ),
    StringComparator(
        column="platform_norm",
        similarity_function="jaccard",
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
    NumericComparator(
        column="releaseYear_num",
        max_difference=1,
    ),
    StringComparator(
        column="platform_norm",
        similarity_function="jaccard",
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
    NumericComparator(
        column="releaseYear_num",
        max_difference=1,
    ),
    StringComparator(
        column="platform_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="developer_norm",
        similarity_function="jaccard",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_d2m = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.68, 0.12, 0.15, 0.05],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_d2s = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_d2s,
    comparators=comparators_d2s,
    weights=[0.68, 0.12, 0.15, 0.05],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_m2s = matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_m2s,
    comparators=comparators_m2s,
    weights=[0.68, 0.12, 0.15, 0.05],
    threshold=0.78,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_d2m.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_dbpedia_metacritic.csv",
    ),
    index=False,
)

rb_correspondences_d2s.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_dbpedia_sales.csv",
    ),
    index=False,
)

rb_correspondences_m2s.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metacritic_sales.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d2m, rb_correspondences_d2s, rb_correspondences_m2s],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", prefer_non_null)
strategy.add_attribute_fuser("developer", lambda values: canonical_from_norm(values, normalize_company))
strategy.add_attribute_fuser("platform", best_platform)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("criticScore", best_critic_score)
strategy.add_attribute_fuser("userScore", best_user_score)
strategy.add_attribute_fuser("ESRB", lambda values: canonical_from_norm([normalize_esrb(v) for v in values if pd.notna(v)]))
strategy.add_attribute_fuser("publisher", lambda values: canonical_from_norm(values, normalize_company))
strategy.add_attribute_fuser("globalSales", max_numeric_as_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

