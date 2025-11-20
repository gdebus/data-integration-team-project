import os
import pandas as pd
from pathlib import Path
from PyDI.io import load_parquet
from PyDI.entitymatching import TokenBlocker, RuleBasedMatcher, StringComparator
from PyDI.fusion import (
    DataFusionEngine,
    DataFusionStrategy,
    longest_string,
    prefer_higher_trust,
    union,
)


def main(file_paths):
    """
    This script demonstrates a full PyDI pipeline for integrating book datasets.
    """
    # Load and preprocess datasets
    datasets = []
    for file_path in file_paths:
        df = load_parquet(file_path)
        df.columns = [col.lower().replace("_", "") for col in df.columns]
        # Add a dataset name attribute for provenance
        df.attrs["name"] = Path(file_path).stem
        datasets.append(df)

    # Define a unified schema. This is a simplified example.
    # A more robust solution would involve a more detailed schema mapping step.
    unified_schema = {
        "id": "string",
        "title": "string",
        "author": "string",
        "rating": "float64",
        "numratings": "int64",
        "language": "string",
        "genres": "string",
        "bookformat": "string",
        "edition": "string",
        "pagecount": "int64",
        "publisher": "string",
        "publishyear": "int64",
        "price": "float64",
        "isbnclean": "string",
    }

    # Normalize datasets to the unified schema
    normalized_datasets = []
    for df in datasets:
        # Rename columns that match the unified schema
        df = df.rename(
            columns={col: col for col in df.columns if col in unified_schema}
        )
        # Add missing columns with None
        for col in unified_schema:
            if col not in df.columns:
                df[col] = None
        # Cast columns to the correct type
        for col, dtype in unified_schema.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    df[col] = None  # Or handle conversion errors as needed
        normalized_datasets.append(df[list(unified_schema.keys())])

    # Combine all datasets into a single DataFrame for blocking
    all_books = pd.concat(normalized_datasets, ignore_index=True)

    # Perform blocking
    blocker = TokenBlocker(column="title")
    candidate_pairs = blocker.block(all_books, all_books)

    # Perform matching
    comparators = [
        StringComparator(column="title", similarity_function="jaccard"),
        StringComparator(column="author", similarity_function="jaro_winkler"),
        StringComparator(column="isbnclean", similarity_function="exact"),
    ]
    matcher = RuleBasedMatcher(
        comparators=comparators, weights=[0.5, 0.3, 0.2], threshold=0.7
    )
    matches = matcher.match(
        df_left=all_books, df_right=all_books, candidates=candidate_pairs
    )

    # Perform data fusion
    strategy = DataFusionStrategy(name="book_fusion_strategy")
    strategy.add_attribute_fuser("title", longest_string)
    strategy.add_attribute_fuser("author", prefer_higher_trust)
    strategy.add_attribute_fuser("rating", prefer_higher_trust)
    strategy.add_attribute_fuser("numratings", prefer_higher_trust)
    strategy.add_attribute_fuser("publisher", prefer_higher_trust)
    strategy.add_attribute_fuser("publishyear", prefer_higher_trust)

    engine = DataFusionEngine(strategy, debug=True, debug_format="json")
    fused_data = engine.run(
        datasets=normalized_datasets, correspondences=matches, id_column="id"
    )

    print("Fused Data:")
    print(fused_data.head())
    print(f"Number of fused records: {len(fused_data)}")


if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = Path(__file__).parent.absolute()

    # Construct the absolute paths to the dataset files
    file_paths = [
        current_dir / ".." / "datasets" / "amazon.parquet",
        current_dir / ".." / "datasets" / "goodreads.parquet",
        current_dir / ".." / "datasets" / "metabooks.parquet",
    ]

    # Convert the Path objects to strings before passing them to the main function
    main([str(p) for p in file_paths])
