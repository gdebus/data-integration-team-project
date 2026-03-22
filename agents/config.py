"""Centralized configuration for the Agent Pipeline.

All magic numbers, thresholds, and timeouts live here.
Every value can be overridden via environment variable.
"""

import os


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


# ── Directories ──────────────────────────────────────────────────────────────

OUTPUT_DIR = os.getenv("AGENT_OUTPUT_DIR", "output/")
INPUT_DIR = os.getenv("AGENT_INPUT_DIR", "input/")

# ── Output paths ─────────────────────────────────────────────────────────────

FUSED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
PIPELINE_CODE_PATH = os.path.join(OUTPUT_DIR, "code", "pipeline.py")
EVALUATION_CODE_PATH = os.path.join(OUTPUT_DIR, "code", "evaluation.py")
EVALUATION_JSON_PATH = os.path.join(OUTPUT_DIR, "pipeline_evaluation", "pipeline_evaluation.json")
DEBUG_EVAL_JSONL_PATH = os.path.join(OUTPUT_DIR, "pipeline_evaluation", "debug_fusion_eval.jsonl")
DIAGNOSTICS_CODE_PATH = os.path.join(OUTPUT_DIR, "code", "integration_diagnostics.py")
DIAGNOSTICS_JSON_PATH = os.path.join(OUTPUT_DIR, "pipeline_evaluation", "integration_diagnostics.json")
DIAGNOSTICS_MD_PATH = os.path.join(OUTPUT_DIR, "pipeline_evaluation", "integration_diagnostics.md")
HUMAN_REVIEW_CODE_PATH = os.path.join(OUTPUT_DIR, "code", "human_review_export.py")
HUMAN_REVIEW_DIR = os.path.join(OUTPUT_DIR, "human_review")
CORRESPONDENCES_DIR = os.path.join(OUTPUT_DIR, "correspondences")
CODE_DIR = os.path.join(OUTPUT_DIR, "code")

# ── LLM / API ───────────────────────────────────────────────────────────────

OPENAI_REQUEST_TIMEOUT = _env_int("OPENAI_REQUEST_TIMEOUT", 600)
OPENAI_MAX_RETRIES = _env_int("OPENAI_MAX_RETRIES", 3)
LLM_REQUEST_TIMEOUT = _env_int("LLM_REQUEST_TIMEOUT", OPENAI_REQUEST_TIMEOUT)
LLM_INVOKE_MAX_ATTEMPTS = _env_int("LLM_INVOKE_MAX_ATTEMPTS", 2)
PIPELINE_TOOLCALL_LOOP_LIMIT = _env_int("PIPELINE_TOOLCALL_LOOP_LIMIT", 6)

INCLUDE_DOCS = os.getenv("AGENT_INCLUDE_DOCS", "").strip().lower() in {"1", "true", "yes"}
USE_LLM = os.getenv("AGENT_USE_LLM", "gpt").strip().lower()

# ── Quality gates ────────────────────────────────────────────────────────────

QUALITY_GATE_THRESHOLD = _env_float("QUALITY_GATE_THRESHOLD", 0.85)
MATCHING_F1_GATE = _env_float("MATCHING_F1_GATE", 0.65)
MATCHING_F1_THRESHOLD = _env_float("MATCHING_F1_THRESHOLD", 0.75)
BLOCKING_PC_THRESHOLD = _env_float("BLOCKING_PC_THRESHOLD", 0.9)
ID_MAPPED_COVERAGE_THRESHOLD = _env_float("ID_MAPPED_COVERAGE_THRESHOLD", 0.95)
ID_DIRECT_COVERAGE_THRESHOLD = _env_float("ID_DIRECT_COVERAGE_THRESHOLD", 0.7)
MISMATCH_RATIO_THRESHOLD = _env_float("MISMATCH_RATIO_THRESHOLD", 0.4)
MISSING_VALUE_THRESHOLD = _env_float("MISSING_VALUE_THRESHOLD", 0.2)
MISSING_FUSED_VALUE_THRESHOLD = _env_float("MISSING_FUSED_VALUE_THRESHOLD", 0.25)
ID_MAPPED_COVERAGE_LOW = _env_float("ID_MAPPED_COVERAGE_LOW", 0.8)
ID_DIRECT_COVERAGE_LOW = _env_float("ID_DIRECT_COVERAGE_LOW", 0.5)

# ── Execution timeouts & retries ─────────────────────────────────────────────

PIPELINE_EXEC_TIMEOUT = _env_int("PIPELINE_EXEC_TIMEOUT", 3600)
EVAL_EXEC_TIMEOUT = _env_int("EVAL_EXEC_TIMEOUT", 3600)
DIAGNOSTICS_EXEC_TIMEOUT = _env_int("DIAGNOSTICS_EXEC_TIMEOUT", 1200)
HUMAN_REVIEW_EXEC_TIMEOUT = _env_int("HUMAN_REVIEW_EXEC_TIMEOUT", 1800)
PIPELINE_EXEC_MAX_ATTEMPTS = _env_int("PIPELINE_EXEC_MAX_ATTEMPTS", 3)

# ── Graph execution ──────────────────────────────────────────────────────────
GRAPH_RECURSION_LIMIT = _env_int("GRAPH_RECURSION_LIMIT", 200)

# ── Matching / Blocking ─────────────────────────────────────────────────────

MATCHING_MAX_ATTEMPTS = _env_int("MATCHING_MAX_ATTEMPTS", 8)
MATCHING_MAX_ERROR_RETRIES = _env_int("MATCHING_MAX_ERROR_RETRIES", 2)
BLOCKING_MAX_CANDIDATES = _env_int("BLOCKING_MAX_CANDIDATES", 350000)
BLOCKING_MAX_ATTEMPTS = _env_int("BLOCKING_MAX_ATTEMPTS", 5)
BLOCKING_MAX_ERROR_RETRIES = _env_int("BLOCKING_MAX_ERROR_RETRIES", 2)
SKIP_BLOCKING_TESTER = os.getenv("SKIP_BLOCKING_TESTER", "").strip().lower() in {"1", "true", "yes"}
SKIP_MATCHING_TESTER = os.getenv("SKIP_MATCHING_TESTER", "").strip().lower() in {"1", "true", "yes"}

# ── Truncation ───────────────────────────────────────────────────────────────

STDERR_MAX_LENGTH = _env_int("STDERR_MAX_LENGTH", 50000)
DIAGNOSTICS_ERROR_MAX_LENGTH = _env_int("DIAGNOSTICS_ERROR_MAX_LENGTH", 5000)
ERROR_RAW_SNIPPET_LENGTH = _env_int("ERROR_RAW_SNIPPET_LENGTH", 500)

# ── Regression thresholds ────────────────────────────────────────────────────

REGRESSION_MINOR = _env_float("REGRESSION_MINOR", 0.02)
REGRESSION_MODERATE = _env_float("REGRESSION_MODERATE", 0.08)
REGRESSION_SEVERE = _env_float("REGRESSION_SEVERE", 0.12)
REGRESSION_CATASTROPHIC = _env_float("REGRESSION_CATASTROPHIC", 0.20)

# ── Investigation ────────────────────────────────────────────────────────────

MAX_INVESTIGATION_ATTEMPTS = _env_int("MAX_INVESTIGATION_ATTEMPTS", 4)
INVESTIGATION_MAX_TURNS = _env_int("INVESTIGATION_MAX_TURNS", 4)
INVESTIGATION_CODE_TIMEOUT = _env_int("INVESTIGATION_CODE_TIMEOUT", 120)
INVESTIGATION_TRANSCRIPT_DIR = os.path.join(OUTPUT_DIR, "investigation")

# ── Evidence-based routing ──────────────────────────────────────────────────
ROUTING_BLOCKING_CANDIDATE_MIN = _env_int("ROUTING_BLOCKING_CANDIDATE_MIN", 10)
ROUTING_MATCHING_F1_MIN = _env_float("ROUTING_MATCHING_F1_MIN", 0.60)
ROUTING_STAGE_DIAGNOSIS_WEIGHT = _env_float("ROUTING_STAGE_DIAGNOSIS_WEIGHT", 0.35)
ROUTING_BASE_THRESHOLD = _env_float("ROUTING_BASE_THRESHOLD", 0.72)

# ── Probe runner ────────────────────────────────────────────────────────────
MAX_PROBES = _env_int("MAX_PROBES", 10)
PROBE_MAX_RUNTIME_SECONDS = _env_float("PROBE_MAX_RUNTIME_SECONDS", 6.0)
PROBE_MAX_EVENTS = _env_int("PROBE_MAX_EVENTS", 500)
CUSTOM_PROBE_TIMEOUT_SECONDS = _env_float("CUSTOM_PROBE_TIMEOUT_SECONDS", 2.0)
MAX_CUSTOM_PROBES = _env_int("MAX_CUSTOM_PROBES", 2)

# ── Evaluation execution ──────────────────────────────────────────────────
EVAL_EXEC_MAX_ATTEMPTS = _env_int("EVAL_EXEC_MAX_ATTEMPTS", 3)

# ── Normalization rejection ──────────────────────────────────────────────
NORMALIZATION_MAX_CONSECUTIVE_REJECTIONS = _env_int("NORMALIZATION_MAX_CONSECUTIVE_REJECTIONS", 2)

# ── Mismatch sampler ────────────────────────────────────────────────────────
MISMATCH_SAMPLE_ATTRIBUTES = _env_int("MISMATCH_SAMPLE_ATTRIBUTES", 5)
MISMATCH_SAMPLE_ROWS = _env_int("MISMATCH_SAMPLE_ROWS", 5)

# ── Source attribution probe ───────────────────────────────────────────────
SOURCE_ATTRIBUTION_MAX_RECORDS = _env_int("SOURCE_ATTRIBUTION_MAX_RECORDS", 50)

# ── Attribute improvability analysis ─────────────────────────────────────
STAGNATION_WINDOW = _env_int("STAGNATION_WINDOW", 2)
STAGNATION_DELTA_THRESHOLD = _env_float("STAGNATION_DELTA_THRESHOLD", 0.02)

# ── LLM rate card (USD per 1M tokens: input, output) ────────────────────────

OPENAI_RATE_CARD = {
    "gpt-5.4": (2.50, 15.0),
    "gpt-5.2": (1.25, 10.0),
    "gpt-5": (1.25, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (5.0, 15.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-nano": (0.10, 0.40),
}


def truncate_stderr(text: str) -> str:
    """Truncates stderr/error output to configured max length."""
    if not text:
        return ""
    return text[:STDERR_MAX_LENGTH]


def configure_run_output(run_dir: str) -> None:
    """Redirect all output paths to a per-run directory.

    Must be called before any pipeline node writes output so that every
    module that accesses ``config.OUTPUT_DIR`` (via ``import config``)
    picks up the run-specific path.
    """
    global OUTPUT_DIR, FUSED_OUTPUT_PATH, PIPELINE_CODE_PATH, EVALUATION_CODE_PATH
    global EVALUATION_JSON_PATH, DEBUG_EVAL_JSONL_PATH, DIAGNOSTICS_CODE_PATH
    global DIAGNOSTICS_JSON_PATH, DIAGNOSTICS_MD_PATH, HUMAN_REVIEW_CODE_PATH
    global HUMAN_REVIEW_DIR, CORRESPONDENCES_DIR, CODE_DIR
    global INVESTIGATION_TRANSCRIPT_DIR

    OUTPUT_DIR = run_dir
    FUSED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
    PIPELINE_CODE_PATH = os.path.join(OUTPUT_DIR, "code", "pipeline.py")
    EVALUATION_CODE_PATH = os.path.join(OUTPUT_DIR, "code", "evaluation.py")
    EVALUATION_JSON_PATH = os.path.join(OUTPUT_DIR, "pipeline_evaluation", "pipeline_evaluation.json")
    DEBUG_EVAL_JSONL_PATH = os.path.join(OUTPUT_DIR, "pipeline_evaluation", "debug_fusion_eval.jsonl")
    DIAGNOSTICS_CODE_PATH = os.path.join(OUTPUT_DIR, "code", "integration_diagnostics.py")
    DIAGNOSTICS_JSON_PATH = os.path.join(OUTPUT_DIR, "pipeline_evaluation", "integration_diagnostics.json")
    DIAGNOSTICS_MD_PATH = os.path.join(OUTPUT_DIR, "pipeline_evaluation", "integration_diagnostics.md")
    HUMAN_REVIEW_CODE_PATH = os.path.join(OUTPUT_DIR, "code", "human_review_export.py")
    HUMAN_REVIEW_DIR = os.path.join(OUTPUT_DIR, "human_review")
    CORRESPONDENCES_DIR = os.path.join(OUTPUT_DIR, "correspondences")
    CODE_DIR = os.path.join(OUTPUT_DIR, "code")
    INVESTIGATION_TRANSCRIPT_DIR = os.path.join(OUTPUT_DIR, "investigation")
