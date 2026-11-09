"""LangGraph workflow with Supervisor-selected conditional agent routing."""
from langgraph.graph import END, START, StateGraph

from graph.state import AgentState
from agents.supervisor import supervisor_node
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.verifier import verifier_node
from agents.reporter import reporter_node

AGENT_NODES = {
    "researcher": researcher_node,
    "analyst": analyst_node,
    "verifier": verifier_node,
    "reporter": reporter_node,
}


def track_completion(agent_name, agent_function):
    def wrapped_node(state):
        result = agent_function(state) or {}
        completed = list(state.get("completed_agents") or [])
        if agent_name not in completed:
            completed.append(agent_name)
        warnings = list(state.get("workflow_warnings") or [])
        if result.get("research_error"):
            warnings.append(str(result["research_error"]))
        if result.get("verification_error"):
            warnings.append("Verification stage failed: " + str(result["verification_error"]))
        for field in ("analysis", "verification"):
            value = result.get(field)
            if isinstance(value, str) and any(marker in value.lower() for marker in ("could not be generated", "could not be completed", "failed because", "error code:")):
                warnings.append(f"{agent_name} stage did not complete successfully.")
        return {**result, "completed_agents": completed, "workflow_warnings": warnings}
    return wrapped_node


def route_next_agent(state):
    tasks = (state.get("plan") or {}).get("tasks", [])
    completed = set(state.get("completed_agents") or [])
    for task in tasks:
        name = task.get("agent")
        if name in AGENT_NODES and name not in completed:
            print(f"\n[Router] Next agent: {name}")
            return name
    print("\n[Router] Planned workflow stages completed.")
    return "end"


workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
for name, function in AGENT_NODES.items():
    workflow.add_node(name, track_completion(name, function))

route_map = {name: name for name in AGENT_NODES}
route_map["end"] = END
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", route_next_agent, route_map)
for name in AGENT_NODES:
    workflow.add_conditional_edges(name, route_next_agent, route_map)

app = workflow.compile()
