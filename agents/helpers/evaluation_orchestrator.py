from typing import Any, Dict

from _resolve_import import ensure_project_root
ensure_project_root()

from config import EVAL_EXEC_MAX_ATTEMPTS
from workflow_logging import log_agent_action


def run_evaluation_node(agent, state: Dict[str, Any]) -> Dict[str, Any]:
    log_agent_action(
        agent,
        step="evaluation_node",
        action="start",
        why="Run evaluation loop",
        improvement="Generates and executes evaluation in one node",
    )
    print("[EVAL] Starting evaluation node")

    local_state = dict(state)
    local_state["evaluation_execution_attempts"] = 0
    local_state["evaluation_execution_result"] = ""

    updates: Dict[str, Any] = {}
    max_attempts = EVAL_EXEC_MAX_ATTEMPTS

    for attempt_idx in range(1, max_attempts + 1):
        print(f"[EVAL] Code generation + execution (attempt {attempt_idx}/{max_attempts})")
        adaption_updates = agent.evaluation_adaption(local_state)
        local_state.update(adaption_updates)
        updates.update(adaption_updates)

        execution_updates = agent.execute_evaluation(local_state)
        local_state.update(execution_updates)
        updates.update(execution_updates)

        result = str(local_state.get("evaluation_execution_result", "")).lower()

        if result.startswith("success"):
            print("[EVAL] Evaluation succeeded")
            break

        error_info = local_state.get("evaluation_error_classification", {})
        if isinstance(error_info, dict) and not error_info.get("retryable", True):
            print(f"[EVAL] Non-retryable error: {error_info.get('category')}")
            break

        if int(local_state.get("evaluation_execution_attempts", 0)) >= max_attempts:
            break

    return updates
