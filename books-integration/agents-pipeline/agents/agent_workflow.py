import os
import re
import textwrap
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from data_loader import LoaderProfiler
from reference_db import query_pydi_reference
from dotenv import load_dotenv

load_dotenv()


def build_agent():
    """Create Groq LLM"""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    return llm


def extract_code_from_response(llm_output: str) -> str:
    """Extract first triple-backtick code block from LLM output."""
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", llm_output)
    if m:
        return m.group(1).strip()
    return llm_output.strip()


class IntegrationAgent:
    def __init__(self, work_dir: str = "agent_work"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.llm = build_agent()

        self.system_prompt = (
            "You are an expert PyDI developer. You must produce valid Python code that uses only the PyDI "
            "library (no other data integration libraries). The code should be executable in a standard Python "
            "environment where PyDI is installed. When asked to produce code, return the code inside a triple "
            "backtick markdown block and do not include any additional commentary. "
            "Keep code minimal and focused on the requested task (schema matching / blocking / fusion etc.)."
        )

    def load_and_profile(
        self, file_paths: List[str]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Use LoaderProfiler to load & profile input datasets.
        Returns (dataframes dict, profiles dict).
        """
        profiler = LoaderProfiler(file_paths)
        dataframes, profiles = profiler.run()
        return dataframes, profiles

    def query_reference(self, query: str) -> str:
        """
        Uses query_pydi_reference if available.
        Otherwise returns empty string.
        """
        if query_pydi_reference is None:
            return ""
        try:
            return query_pydi_reference(query)
        except Exception as e:
            return f"[Reference query failed: {e}]"

    def build_prompt_for_code_generation(
        self,
        profiles: Dict[str, str],
        task_description: str,
        include_reference: bool = True,
    ) -> str:
        """
        Compose a comprehensive user prompt containing:
         - task description (what code to generate)
         - profile summary for each dataset
         - reference excerpt
         - a one-shot example of a correct PyDI script
        """

        profile_snippets = []
        for name, prof in profiles.items():
            profile_snippets.append(f"Dataset: {name}\n{prof}")

        profile_section = "\n\n---\n\n".join(profile_snippets)

        # reference context
        reference_section = ""
        if include_reference:
            try:
                ref_text = self.query_reference(
                    "Show examples of PyDI data loading, profiling, Entity Matching usage, blocking, matching and data fusion usage."
                )

                if ref_text:
                    reference_section = (
                        "\n\n-----\n\nReference excerpts (from PyDI reference):\n"
                        + ref_text
                    )
            except Exception:
                reference_section = ""

        # One-shot example
        one_shot_example_path = (
            Path.cwd()
            / "books-integration"
            / "agents-pipeline"
            / "agents"
            / "input"
            / "one_shot_example.py"
        )
        with open(one_shot_example_path, "r", encoding="utf-8") as f:
            one_shot_example = f.read()

        # user prompt
        user_prompt = textwrap.dedent(
            f"""
        Task:
        {task_description}

        Use the sample dataset profiles below, the reference excerpts, and the one-shot example to generate a single, standalone Python script that uses **only** the PyDI library to accomplish the task for all provided datasets.

        The script must:
          - Follow the structure of the one-shot example.
          - Load data from files using proper loader `PyDI.io.load_parquet` or `PyDI.io.load_csv` or `PyDI.io.load_xml`.
          - Define a unified schema based on the provided dataset profiles.
          - Normalize each dataset to the unified schema.
          - Perform blocking on the DataFrames with the columns.
          - Perform entity matching.
          - Perform data fusion using `PyDI.fusion.DataFusionEngine` and a `DataFusionStrategy`.
          - Define a `main()` function that accepts a list of file paths as an argument.
          - Include a `if __name__ == "__main__":` block to call the `main` function with the correct Parquet file paths.

        Dataset profiles:
        {profile_section}

        {reference_section}

        Here is an example of a correct PyDI pipeline script. Use this as a guide to generate the new script.
        ```python
        {one_shot_example}
        ```

        Provide the Python code only, inside a triple-backtick code block, and nothing else.
        """
        )
        return user_prompt

    def generate_pydi_code(
        self,
        profiles: Dict[str, str],
        task_description: str,
        output_py_path: str,
        include_reference: bool = True,
        temperature: float = 0.0,
    ) -> Tuple[str, str]:
        """
        generates PyDI-only code for the task, save it to output_py_path.
        Returns (saved_path, generated_code_str)
        """
        user_prompt = self.build_prompt_for_code_generation(
            profiles=profiles,
            task_description=task_description,
            include_reference=include_reference,
        )

        messages = [("system", self.system_prompt), ("human", user_prompt)]
        try:
            llm_reply = self.llm.invoke(messages).content
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {e}")

        code = extract_code_from_response(llm_reply)

        if "PyDI" not in code and "pydi" not in code.lower():
            print(
                "[WARNING] Generated code does not seem to reference PyDI. Saving anyway."
            )

        # Save to file
        out_path = Path(output_py_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated PyDI integration script\n")
            f.write("# Generated by Groq LLaMA - IntegrationAgent\n\n")
            f.write(code)
            f.write("\n")

        print(f"[OK] Saved generated code to: {out_path}")
        return str(out_path), code


def main():
    """
    - building reference DB
    - loading & profiling 2+ datasets
    - generating PyDI integration code and saving it to disk
    """

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
    if not GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY in environment before running.")

    # paths to datasets
    ROOT = Path.cwd()
    DATA_DIR = ROOT / "books-integration" / "agents-pipeline" / "datasets"
    OUTPUT_DIR = ROOT / "books-integration" / "agents-pipeline" / "agents" / "output"

    datasets = [
        str(DATA_DIR / "amazon.parquet"),
        str(DATA_DIR / "goodreads.parquet"),
        str(DATA_DIR / "metabooks.parquet"),
    ]

    # create agent
    agent = IntegrationAgent()

    # load & profile datasets
    print(">> Loading & profiling datasets...")
    dataframes, profiles = agent.load_and_profile(datasets)

    # task description
    task_description = (
        "Create a PyDI-based script that reads the given datasets, performs schema matching to a unified "
        "target schema, then performs identity resolution (blocking + matching) and data fusion "
        "Use sensible defaults for blocking "
        "and matching; rely on PyDI's TokenBlocker and matchers. Include provenance in the output."
    )

    # generate code and save
    save_path = str(OUTPUT_DIR / "generated_pydi_pipeline.py")
    saved_path, code = agent.generate_pydi_code(
        profiles=profiles, task_description=task_description, output_py_path=save_path
    )

    print("\n=== Generated code saved ===")
    print(saved_path)
    print(code[:1000])


if __name__ == "__main__":
    main()
