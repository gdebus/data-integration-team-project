from typing import Any, Dict


def run_evaluation_node(agent, state: Dict[str, Any]) -> Dict[str, Any]:
    agent._log_action(
        "evaluation_node",
        "start",
        "Run evaluation loop",
        "Generates and executes evaluation in one node",
    )
    print("[*] Running consolidated evaluation node")

    local_state = dict(state)
    local_state["evaluation_execution_attempts"] = 0
    local_state["evaluation_execution_result"] = ""

    updates: Dict[str, Any] = {}
    max_attempts = 3

    for attempt_idx in range(1, max_attempts + 1):
        print(f"[EVALUATION EXECUTION] Attempt {attempt_idx}/{max_attempts}")
        adaption_updates = agent.evaluation_adaption(local_state)
        local_state.update(adaption_updates)
        updates.update(adaption_updates)

        execution_updates = agent.execute_evaluation(local_state)
        local_state.update(execution_updates)
        updates.update(execution_updates)

        result = str(local_state.get("evaluation_execution_result", "")).lower()

        print(f"[EVALUATION EXECUTION] Status: {local_state.get('evaluation_execution_result', '')}")
        if result.startswith("success"):
            break
        if int(local_state.get("evaluation_execution_attempts", 0)) >= max_attempts:
            break

    return updates
