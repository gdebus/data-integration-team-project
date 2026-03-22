"""Matching config validation and quality-gate helpers."""
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import MATCHING_F1_THRESHOLD


def config_has_list_based_comparators(cfg: Dict[str, Any]) -> bool:
    try:
        strategies = cfg.get("matching_strategies", {}) if isinstance(cfg, dict) else {}
        for _, pair_cfg in strategies.items():
            comps = pair_cfg.get("comparators", []) if isinstance(pair_cfg, dict) else []
            for comp in comps:
                if bool((comp or {}).get("list_strategy")):
                    return True
        return False
    except Exception:
        return False


def config_matches_datasets(cfg: Dict[str, Any], dataset_paths: List[str]) -> bool:
    try:
        if not isinstance(cfg, dict):
            return False
        cfg_names = cfg.get("dataset_names", [])
        if not isinstance(cfg_names, list) or not cfg_names:
            return True
        expected = sorted([Path(p).stem for p in dataset_paths])
        got = sorted([str(x) for x in cfg_names])
        return expected == got
    except Exception:
        return False


def matching_config_needs_refresh(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(cfg, dict):
        return True, "matching config is not a dict"
    strategies = cfg.get("matching_strategies", {})
    if not isinstance(strategies, dict) or not strategies:
        return True, "matching strategies are missing"
    weak_pairs = []
    for pair_name, pair_cfg in strategies.items():
        if not isinstance(pair_cfg, dict):
            weak_pairs.append(f"{pair_name}:invalid")
            continue
        try:
            f1 = float(pair_cfg.get("f1", 0.0) or 0.0)
        except Exception:
            f1 = 0.0
        failure_tags = {str(tag) for tag in pair_cfg.get("failure_tags", []) if str(tag).strip()}
        if "low_matching_quality" in failure_tags or f1 < MATCHING_F1_THRESHOLD:
            weak_pairs.append(f"{pair_name}:F1={f1:.3f}")
    if weak_pairs:
        return True, "low-quality matching strategies present (" + ", ".join(weak_pairs[:5]) + ")"
    return False, ""
