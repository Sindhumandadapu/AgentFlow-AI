
"""FastAPI interface for the AgentFlow AI React chat frontend."""

from datetime import datetime
from pathlib import Path
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph.workflow import app as workflow_app


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
REPORT_DIR = BACKEND_DIR / "reports"

app = FastAPI(
    title="AgentFlow AI API",
    version="1.0.0"
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=10000
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "AgentFlow AI"
    }


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.post("/api/chat")
def chat(payload: ChatRequest):

    # 1. Validate user message
    message = payload.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question or task."
        )

    # 2. Initialize LangGraph state
    initial_state = {
        "user_request": message,
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

    # 3. Execute the five-agent workflow
    try:

        print("\n========== AGENTFLOW REQUEST ==========")
        print(f"User Request: {message}")

        result = workflow_app.invoke(
            initial_state,
            config={"recursion_limit": 20}
        )

        print("\n========== WORKFLOW COMPLETED ==========")

    except Exception as exc:

        print("\n========== AGENTFLOW ERROR ==========")
        print(f"Error Type: {type(exc).__name__}")
        print(f"Error Message: {exc}")

        # Print the complete traceback in the backend terminal
        traceback.print_exc()

        print("======================================\n")

        raise HTTPException(
            status_code=500,
            detail=(
                f"AgentFlow execution failed: "
                f"{type(exc).__name__}: {exc}"
            )
        ) from exc

    # 4. Extract final response
    answer = (
        result.get("final_report")
        or "I couldn't generate a response for this request."
    )

    status = (
        result.get("completion_status")
        or result.get("workflow_status")
        or "completed"
    )

    status = str(status).upper()

    # 5. Extract task information safely
    plan = result.get("plan") or {}

    if not isinstance(plan, dict):
        print(
            "[API WARNING] Unexpected plan format:",
            type(plan).__name__
        )
        plan = {}

    task_type = plan.get("task_type") or ""

    if not isinstance(task_type, str):
        print(
            "[API WARNING] Unexpected task_type format:",
            type(task_type).__name__
        )
        task_type = str(task_type)

    # 6. Collect sources
    sources = []

    verification_records = result.get("verification_records") or []

    if isinstance(verification_records, list):

        for item in verification_records:

            if not isinstance(item, dict):
                continue

            url = item.get("source_url")

            if url and all(
                existing.get("url") != url
                for existing in sources
            ):
                sources.append({
                    "title": (
                        item.get("source_title")
                        or item.get("headline")
                        or "Source"
                    ),
                    "url": url,
                    "status": item.get("status", "VERIFIED"),
                })

    research_sources = result.get("research_sources") or []

    if isinstance(research_sources, list):

        for item in research_sources:

            if not isinstance(item, dict):
                continue

            url = item.get("url")

            if (
                url
                and (
                    item.get("fetch_status") == "Completed"
                    or url in answer
                )
                and all(
                    existing.get("url") != url
                    for existing in sources
                )
            ):
                sources.append({
                    "title": item.get("title") or "Source",
                    "url": url,
                    "status": item.get("fetch_status", "FETCHED"),
                })

    # 7. Save generated report
    try:

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_path = (
            REPORT_DIR /
            f"agentflow_report_{stamp}.md"
        )

        report_path.write_text(
            answer,
            encoding="utf-8"
        )

        print(f"[API] Report saved: {report_path}")

    except Exception as exc:

        print("[API WARNING] Report saving failed:")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # 8. Prepare completion information
    completion_reasons = result.get("completion_reasons") or []

    if not isinstance(completion_reasons, list):
        completion_reasons = [str(completion_reasons)]

    completed_agents = result.get("completed_agents") or []

    if not isinstance(completed_agents, list):
        completed_agents = []

    # 9. Return response to React frontend
    response = {
        "answer": answer,
        "status": status,
        "completion_note": "; ".join(
            str(reason) for reason in completion_reasons
        ),
        "task_type": task_type,
        "agents": completed_agents,
        "sources": sources[:10],
    }

    print("\n========== API RESPONSE ==========")
    print(f"Status: {status}")
    print(f"Task Type: {task_type}")
    print(f"Completed Agents: {completed_agents}")
    print(f"Sources Found: {len(sources)}")
    print("==================================\n")

    return response
