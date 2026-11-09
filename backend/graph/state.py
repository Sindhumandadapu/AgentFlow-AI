from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    user_request: str
    plan: Optional[dict]
    research: Optional[str]
    research_records: list[dict[str, Any]]
    research_sources: list[dict[str, Any]]
    research_error: Optional[str]
    analysis: Optional[str]
    verification: Optional[str]
    verification_records: list[dict[str, Any]]
    verification_error: Optional[str]
    final_report: Optional[str]
    completion_status: Optional[str]
    completion_reasons: list[str]
    completed_agents: list[str]
    workflow_status: Optional[str]
    workflow_warnings: list[str]
