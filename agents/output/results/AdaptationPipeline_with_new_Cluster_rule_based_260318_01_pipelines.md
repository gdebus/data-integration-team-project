# Pipeline Snapshots

notebook_name=AdaptationPipeline_with_new_Cluster
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=17.65%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, maximum
import pandas as pd
import os
import shutil

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

dbpedia = load_csv(
    DATA_DIR + "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "output/schema-matching/metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "output/schema-matching/sales.csv",
    name="sales",
)

datasets = [dbpedia, metacritic, sales]

# --------------------------------
# Perform Blocking
# MUST use precomputed blocking configuration
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
# Matching configuration
# MUST use precomputed matching configuration
# --------------------------------

lower_strip = lambda x: str(x).lower().strip()

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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

threshold_dbpedia_sales = 0.75
threshold_metacritic_dbpedia = 0.8
threshold_metacritic_sales = 0.78

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=threshold_dbpedia_sales,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.45, 0.2, 0.2, 0.15],
    threshold=threshold_metacritic_dbpedia,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.35, 0.2, 0.15, 0.12, 0.08, 0.1],
    threshold=threshold_metacritic_sales,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
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

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
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
node_index=9
node_name=execute_pipeline
accuracy_score=17.65%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, maximum
import pandas as pd
import os
import shutil

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

dbpedia = load_csv(
    DATA_DIR + "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "output/schema-matching/metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "output/schema-matching/sales.csv",
    name="sales",
)

datasets = [dbpedia, metacritic, sales]

# --------------------------------
# Perform Blocking
# MUST use precomputed blocking configuration
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
# Matching configuration
# MUST use precomputed matching configuration
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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

threshold_dbpedia_sales = 0.75
threshold_metacritic_dbpedia = 0.8
threshold_metacritic_sales = 0.78

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.36, 0.19, 0.11, 0.34],
    threshold=threshold_dbpedia_sales,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.42, 0.23, 0.08, 0.27],
    threshold=threshold_metacritic_dbpedia,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.28, 0.16, 0.08, 0.2, 0.08, 0.2],
    threshold=threshold_metacritic_sales,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
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
node_index=12
node_name=execute_pipeline
accuracy_score=17.65%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, maximum
import pandas as pd
import os
import shutil

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

dbpedia = load_csv(
    DATA_DIR + "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "output/schema-matching/metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "output/schema-matching/sales.csv",
    name="sales",
)

datasets = [dbpedia, metacritic, sales]

# --------------------------------
# Perform Blocking
# MUST use precomputed blocking configuration
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
# Matching configuration
# MUST use precomputed matching configuration
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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

threshold_dbpedia_sales = 0.75
threshold_metacritic_dbpedia = 0.8
threshold_metacritic_sales = 0.78

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.5, 0.24, 0.16, 0.1],
    threshold=threshold_dbpedia_sales,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.5, 0.27, 0.13, 0.1],
    threshold=threshold_metacritic_dbpedia,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.2, 0.18, 0.12, 0.22, 0.18, 0.1],
    threshold=threshold_metacritic_sales,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
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

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

============================================================
PIPELINE SNAPSHOT 04 START
============================================================
node_index=19
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, maximum
import pandas as pd
import os
import shutil

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

dbpedia = load_csv(
    DATA_DIR + "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "output/schema-matching/metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "output/schema-matching/sales.csv",
    name="sales",
)

datasets = [dbpedia, metacritic, sales]

# --------------------------------
# Perform Blocking
# MUST use precomputed blocking configuration
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
# Matching configuration
# MUST use precomputed matching configuration
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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

threshold_dbpedia_sales = 0.75
threshold_metacritic_dbpedia = 0.8
threshold_metacritic_sales = 0.78

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.5, 0.24, 0.16, 0.1],
    threshold=threshold_dbpedia_sales,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.5, 0.27, 0.13, 0.1],
    threshold=threshold_metacritic_dbpedia,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.2, 0.18, 0.12, 0.22, 0.18, 0.1],
    threshold=threshold_metacritic_sales,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
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

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 04 END
============================================================

============================================================
PIPELINE SNAPSHOT 05 START
============================================================
node_index=26
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, maximum
import pandas as pd
import os
import shutil

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

dbpedia = load_csv(
    DATA_DIR + "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "output/schema-matching/metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "output/schema-matching/sales.csv",
    name="sales",
)

datasets = [dbpedia, metacritic, sales]

# --------------------------------
# Perform Blocking
# MUST use precomputed blocking configuration
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
# Matching configuration
# MUST use precomputed matching configuration
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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

threshold_dbpedia_sales = 0.75
threshold_metacritic_dbpedia = 0.8
threshold_metacritic_sales = 0.78

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.5, 0.24, 0.16, 0.1],
    threshold=threshold_dbpedia_sales,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.5, 0.27, 0.13, 0.1],
    threshold=threshold_metacritic_dbpedia,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.2, 0.18, 0.12, 0.22, 0.18, 0.1],
    threshold=threshold_metacritic_sales,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
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

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 05 END
============================================================

