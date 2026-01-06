from pathlib import Path
import pandas as pd
from PyDI.io import load_parquet
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import MaximumBipartiteMatching

from pathlib import Path
import pandas as pd
from PyDI.io import load_parquet
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_colwidth', 100)

ROOT = Path.cwd()

DATA_DIR = ROOT / "parquet"
OUTPUT_DIR = ROOT / "output"
MLDS_DIR = ROOT / "ml-datasets"
BLOCK_EVAL_DIR = OUTPUT_DIR / "blocking_evaluation"
CORR_DIR = OUTPUT_DIR / "correspondences"

PIPELINE_DIR = OUTPUT_DIR / "data_fusion"
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

amazon_sample = load_parquet(
    DATA_DIR / "amazon_sample.parquet",
    name="amazon_sample"
)

goodreads_sample = load_parquet(
    DATA_DIR / "goodreads_sample.parquet",
    name="goodreads_sample"
)

metabooks_sample = load_parquet(
  DATA_DIR / "metabooks_sample.parquet",
  name="metabooks_sample"
)

import re

def clean_text(t):
  t = str(t).lower()
  t = re.sub(r'<.*?>', '', t)
  t = re.sub(r'[^a-z0-9\s]', '', t)
  t = re.sub(r'\s+', ' ', t).strip()
  return t

amazon_sample['clean_title'] = amazon_sample['title'].apply(clean_text)
goodreads_sample['clean_title'] = goodreads_sample['title'].apply(clean_text)
metabooks_sample['clean_title'] = metabooks_sample['title'].apply(clean_text)
# Clean Author
amazon_sample["clean_author"] = amazon_sample["author"].apply(clean_text)
goodreads_sample["clean_author"] = goodreads_sample["author"].apply(clean_text)
metabooks_sample["clean_author"] = metabooks_sample["author"].apply(clean_text)
#Clean Publisher
amazon_sample["clean_publisher"] = amazon_sample["publisher"].apply(clean_text)
metabooks_sample["clean_publisher"] = metabooks_sample["publisher"].apply(clean_text)
goodreads_sample["clean_publisher"] = goodreads_sample["publisher"].apply(clean_text)

from PyDI.io import load_csv

train_m2a = load_csv(
    MLDS_DIR / "train_MA.csv",
    name="train_metabooks_amazon",
    header=0,
    names=['id1', 'id2', 'label'],
    add_index=False
)

test_m2a = load_csv(
    MLDS_DIR / "test_MA.csv",
    name="test_metabooks_amazon",
    header=0,
    names=['id1', 'id2', 'label'],
    add_index=False
)

train_m2g = load_csv(
    MLDS_DIR / "train_MG.csv",
    name="train_metabooks_goodreads",
    header=0,
    names=['id1', 'id2', 'label'],
    add_index=False
)

test_m2g = load_csv(
    MLDS_DIR / "test_MG.csv",
    name="test_metabooks_goodreads",
    header=0,
    names=['id1', 'id2', 'label'],
    add_index=False
)

from PyDI.entitymatching import EmbeddingBlocker

embedding_blocker_m2a = EmbeddingBlocker(
    metabooks_sample, amazon_sample,
    text_cols=['clean_title', 'clean_author','publish_year'],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=200,
    output_dir=BLOCK_EVAL_DIR,
    id_column='id'
)


embedding_blocker_m2g = EmbeddingBlocker(
    metabooks_sample, goodreads_sample,
    text_cols=['clean_title', 'clean_author','publish_year'],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=200,
    output_dir=BLOCK_EVAL_DIR,
    id_column='id'
)

embedding_candidates_m2a = embedding_blocker_m2a.materialize()
embedding_candidates_m2g = embedding_blocker_m2g.materialize()

from PyDI.entitymatching import RuleBasedMatcher
from PyDI.entitymatching import StringComparator, NumericComparator

# Base comparators used by both combos
comparators_base = [
    StringComparator(column="clean_title", similarity_function="cosine"),
    NumericComparator(column="publish_year", method="absolute_difference", max_difference=2),
    StringComparator(column="clean_publisher", similarity_function="cosine"),
]

comparators_m2a = comparators_base + [
    StringComparator(column="clean_author", similarity_function="cosine"),
]

comparators_m2g = comparators_base + [
    StringComparator(column="clean_author", similarity_function="jaro_winkler"),
    NumericComparator(column="page_count", max_difference=5),
]

matcher = RuleBasedMatcher()

# matching metabooks > amazon
correspondences_m2a,debug_m2a = matcher.match(
    df_left=metabooks_sample,
    df_right=amazon_sample, 
    candidates=embedding_blocker_m2a,
    comparators=comparators_m2a,
    debug=True,
    weights=[0.5, 0.1, 0.15,0.25], 
    threshold=0.4,
    id_column='id'
)

# matching metabooks > goodreads
correspondences_m2g,debug_m2g = matcher.match(
    df_left=metabooks_sample,
    df_right=goodreads_sample, 
    candidates=embedding_blocker_m2g,
    comparators=comparators_m2g,
    debug=True,
    weights=[0.5, 0.1,0.1,0.2,0.05], 
    threshold=0.4,
    id_column='id'
)

from PyDI.entitymatching import MaximumBipartiteMatching

# We are using Maxmimum Bipartite Matching to refine results to 1:1 matches
clusterer = MaximumBipartiteMatching()
mbm_correspondences_m2a = clusterer.cluster(correspondences_m2a)
mbm_correspondences_m2g = clusterer.cluster(correspondences_m2g)
all_correspondences = pd.concat([mbm_correspondences_m2a, mbm_correspondences_m2g], ignore_index=True)

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
import pandas as pd

metabooks_sample.attrs["trust_score"] = 3
goodreads_sample.attrs["trust_score"] = 2
amazon_sample.attrs["trust_score"] = 1
strategy = DataFusionStrategy('books_fusion_strategy')

strategy.add_attribute_fuser('title', longest_string)
strategy.add_attribute_fuser('author', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('publish_year', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('publisher', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('language', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('price', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('page_count', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('genres',union)

engine = DataFusionEngine(strategy, debug=True, debug_format='json',
                          debug_file= PIPELINE_DIR / "debug_fusion_ml_sn_blocker.jsonl")

fused_rb_emblocker = engine.run(
    datasets=[amazon_sample, metabooks_sample, goodreads_sample],
    correspondences=all_correspondences,
    id_column="id",
    include_singletons=False
)

from PyDI.fusion import tokenized_match, boolean_match,numeric_tolerance_match,set_equality_match

import numpy as np
import re, ast, numpy as np, pandas as pd


def categories_set_equal(a, b) -> bool:
    """Return True if a and b contain the same unique categories (order/type agnostic)."""
    def to_set(x):
        def items(v):
            # missing
            if v is None or (isinstance(v, float) and np.isnan(v)): return []
            # numpy array → recurse over elements
            if isinstance(v, np.ndarray): 
                out=[]; [out.extend(items(e)) for e in v.flatten()]; return out
            # python containers → recurse over elements
            if isinstance(v, (list, tuple, set)):
                out=[]; [out.extend(items(e)) for e in v]; return out
            # scalar/string: try parse stringified list; else split by delimiters
            s = str(v).strip()
            if s == "" or s.lower() in {"nan","none"}: return []
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)): return items(parsed)
            except Exception:
                pass
            return [p.strip() for p in re.split(r"[|,;/]", s) if p.strip()]
        return {it.lower() for it in items(x)}
    return to_set(a) == to_set(b)

fused_dataset = load_parquet(PIPELINE_DIR / "fused_rb_emblocker.parquet")
fused_dataset["publish_year"] = fused_dataset["publish_year"].astype("Int16")
fused_dataset["page_count"] = fused_dataset["page_count"].astype("Int32")
golden_fused_dataset= load_parquet(MLDS_DIR / "golden_fused_books.parquet")

strategy.add_evaluation_function("title", tokenized_match)
strategy.add_evaluation_function("author", tokenized_match)
strategy.add_evaluation_function("publisher", tokenized_match)
strategy.add_evaluation_function("publish_year", numeric_tolerance_match)
strategy.add_evaluation_function("price", numeric_tolerance_match)
strategy.add_evaluation_function("page_count", numeric_tolerance_match)
strategy.add_evaluation_function("language", tokenized_match)
strategy.add_evaluation_function("genres", categories_set_equal)

from PyDI.fusion import DataFusionEvaluator
fused_dataset.drop_duplicates(subset='isbn_clean', keep='first',inplace=True)
# Create evaluator with our fusion strategy
evaluator = DataFusionEvaluator(strategy, debug=True, debug_file=OUTPUT_DIR / "data_fusion" / "debug_fusion_eval.jsonl", debug_format="json")

# Evaluate the fused results against the gold standard
print("Evaluating fusion results against gold standard...")
evaluation_results = evaluator.evaluate(
    fused_df=fused_dataset,
    fused_id_column='isbn_clean',
    gold_df=golden_fused_dataset,
    gold_id_column='isbn_clean',
)

# Display evaluation metrics
print("\nFusion Evaluation Results:")
print("=" * 40)
for metric, value in evaluation_results.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.3f}")
    else:
        print(f"  {metric}: {value}")
        
print(f"\nOverall Accuracy: {evaluation_results.get('overall_accuracy', 0):.1%}")
