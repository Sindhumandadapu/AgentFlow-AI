"""CLI entry point for AgentFlow AI."""
from datetime import datetime
from pathlib import Path

from graph.workflow import app

BACKEND_DIR = Path(__file__).resolve().parent
REPORT_DIR = BACKEND_DIR / "reports"


def run_agentflow(user_request: str) -> dict:
    initial_state = {
        "user_request": user_request,
        "plan": None,
        "research": None,
        "research_records": [],
        "research_sources": [],
        "research_error": None,
        "analysis": None,
        "verification": None,
        "verification_records": [],
        "verification_error": None,
        "final_report": None,
        "completion_status": None,
        "completion_reasons": [],
        "completed_agents": [],
        "workflow_status": None,
        "workflow_warnings": [],
    }
    result = app.invoke(initial_state, config={"recursion_limit": 20})

    plan = result.get("plan") or {}
    print("\n========== EXECUTION SUMMARY ==========")
    print("Task type:", plan.get("task_type", "general"))
    print("Agents used:", " -> ".join(result.get("completed_agents", [])) or "None")
    print("Workflow status:", (result.get("completion_status") or result.get("workflow_status") or "partial").upper())
    for reason in result.get("completion_reasons", []):
        print("Completion note:", reason)

    print("\n========== ANSWER ==========")
    report = result.get("final_report") or "I couldn't produce a final answer for this request."
    print(report)
    for warning in result.get("workflow_warnings", []):
        print(f"\n[Limitation] {warning}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"agentflow_report_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[AgentFlow] Report saved to: {report_path}")
    return result


def main():
    print("\n================================")
    print("       AGENTFLOW AI")
    print("  Multi-Agent Assistant")
    print("================================")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            request = input("Enter your task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if request.lower() in {"exit", "quit", "q"}:
            print("\nThank you for using AgentFlow AI!")
            break
        if not request:
            print("Please enter a non-empty task.")
            continue

        try:
            print("\n[AgentFlow] Starting execution...")
            run_agentflow(request)
            print("\n[AgentFlow] Execution finished.")
        except Exception as exc:
            print(f"\n[AgentFlow] Execution failed: {type(exc).__name__}: {exc}")
            print("Check backend/.env, MCP connectivity, and the error message above.")


if __name__ == "__main__":
    main()
