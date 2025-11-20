from pathlib import Path
import pandas as pd
import json
import os
from typing import List, Tuple, Dict, Any
from PyDI.profiling import DataProfiler
from PyDI.io import load_csv, load_parquet, load_xml


class LoaderProfiler:
    def __init__(self, file_paths: List[str]):
        """
        file_paths: list of file paths (csv/xml/parquet)
        """
        if len(file_paths) < 2:
            raise ValueError("Please provide at least 2 dataset paths.")

        self.paths = [Path(p) for p in file_paths]
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.profiles: Dict[str, str] = {}
        self.summary: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        """Read file into DataFrame based on suffix."""
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return load_csv(path)
        elif suffix == ".parquet":
            return load_parquet(path)
        elif suffix == ".xml":
            return load_xml(path, nested_handling="aggregate")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _profile_df(self, df: pd.DataFrame) -> str:
        """
        Use PyDI DataProfiler to profile the df.
        Returns the profiler output as string.
        """
        profiler = DataProfiler()
        profile = profiler.summary(df, print_summary=False)
        try:
            profile_str = json.dumps(profile, default=str)
        except Exception:
            profile_str = json.dumps({"profile_str": str(profile)})

        return profile_str

    def run(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Load and profile all files.
        Returns:
          - dataframes dict: {name: DataFrame}
          - profiles dict: {name: profile_df}
        """
        for p in self.paths:
            name = p.stem
            try:
                df = self._read_file(p)
            except Exception as e:
                print(f"[ERROR] Reading {p}: {e}")
                continue

            # Profile
            try:
                profile_df = self._profile_df(df)
            except Exception as e:
                print(f"[WARN] Profiling failed for {name}: {e}")
                profile_df = json.dumps(
                    [
                        {
                            "column_name": c,
                            "data_type": str(df[c].dtype),
                            "num_nulls": int(df[c].isna().sum()),
                            "num_unique": int(df[c].nunique(dropna=True)),
                        }
                        for c in df.columns
                    ],
                    default=str,
                )

            self.dataframes[name] = df
            self.profiles[name] = profile_df
            # small summary
            self.summary[name] = {
                "num_rows": int(df.shape[0]),
                "num_columns": int(df.shape[1]),
                "file_path": str(p),
            }

            print(
                f"Loaded & profiled '{name}' ({self.summary[name]['num_rows']} rows, {self.summary[name]['num_columns']} cols)"
            )

        return self.dataframes, self.profiles

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        return self.summary


if __name__ == "__main__":
    ROOT = Path.cwd()
    DATA_DIR = ROOT / "books-integration" / "agents-pipeline" / "datasets"

    paths = [
        str(DATA_DIR / "amazon.parquet"),
        str(DATA_DIR / "goodreads.parquet"),
        str(DATA_DIR / "metabooks.parquet"),
    ]

    lp = LoaderProfiler(paths)
    dataframes, profiles = lp.run()

    print("Loaded datasets:", list(dataframes.keys()))

    print(profiles["amazon"])

    print(lp.get_summary())
