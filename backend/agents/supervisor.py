"""LLM-led supervisor for a general-purpose, evidence-grounded assistant."""
from typing import Literal
from pydantic import BaseModel, Field
from agents.common import get_llm, GLOBAL_AGENT_PRINCIPLES


class AgentTask(BaseModel):
    agent: Literal["researcher", "analyst", "verifier", "reporter"]
    description: str


class TaskPlan(BaseModel):
    goal: str
    task_type: str = "general_research"
    needs_research: bool = Field(description="Whether external/current/source-specific information is needed to answer well")
    needs_analysis: bool = Field(description="Whether synthesis, comparison, calculation, interpretation or planning adds value")
    needs_verification: bool = Field(description="Whether important factual claims need evidence checks")
    response_depth: Literal["short", "normal", "detailed", "comprehensive"] = "normal"
    output_format: str = "direct answer"
    include_sources: bool = False
    source_policy: str = "Content first; show useful sources only when requested or materially helpful."
    deliverables: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    research_brief: str = ""
    tasks: list[AgentTask] = Field(default_factory=list)


def _fallback_plan(request: str) -> TaskPlan:
    """If planning is unavailable, preserve the base LLM experience.

    Do not infer domains from hardcoded keywords or force a fixed workflow.
    The Reporter calls the configured general-purpose model directly. This
    fallback intentionally avoids research because we have no reliable plan
    for what external information is needed.
    """
    return TaskPlan(
        goal=request,
        task_type="general_question",
        needs_research=False,
        needs_analysis=False,
        needs_verification=False,
        response_depth="normal",
        output_format="direct answer",
        include_sources=False,
        source_policy="Answer directly from the model's knowledge; use sources only when a planned research stage is selected.",
        deliverables=["Answer the user's request directly and helpfully"],
        completion_criteria=["Respond to the user's actual intent; ask a concise clarification if essential information is missing"],
        research_brief="",
        tasks=[AgentTask(agent="reporter", description="Answer the user directly using the configured general-purpose LLM. Ask for clarification only if essential information is missing.")],
    )


def _normalise_plan(plan: TaskPlan, request: str) -> TaskPlan:
    """Normalise the LLM's dynamic plan without imposing keyword routing."""
    selected = {
        task.agent: task.description.strip()
        for task in (plan.tasks or [])
        if task.description and task.description.strip()
    }

    # The planner's decisions are authoritative. Add a missing stage only
    # when its corresponding planning flag says that stage is needed.
    if plan.needs_research:
        selected.setdefault("researcher", "Research the user's request using relevant, dynamically generated queries and fetched pages.")
    if plan.needs_analysis:
        selected.setdefault("analyst", "Synthesize information into the deliverable requested by the user.")
    if plan.needs_verification:
        selected.setdefault("verifier", "Check material factual claims against the available evidence.")
    selected.setdefault("reporter", "Answer the user's exact request clearly and completely.")

    order = ("researcher", "analyst", "verifier", "reporter")
    tasks = [AgentTask(agent=name, description=selected[name]) for name in order if name in selected]
    return TaskPlan(
        goal=(plan.goal or "").strip() or request,
        task_type=(plan.task_type or "general_question").strip(),
        needs_research=bool(plan.needs_research),
        needs_analysis=bool(plan.needs_analysis),
        needs_verification=bool(plan.needs_verification),
        response_depth=plan.response_depth,
        output_format=plan.output_format or "direct answer",
        include_sources=bool(plan.include_sources),
        source_policy=plan.source_policy or "Answer content first; include useful sources when requested or materially helpful.",
        deliverables=plan.deliverables or ["Answer the user's request"],
        completion_criteria=plan.completion_criteria or ["Fulfil the user's actual request"],
        research_brief=plan.research_brief or "",
        tasks=tasks,
    )


def supervisor_node(state):
    request = (state.get("user_request") or "").strip()
    fallback = _fallback_plan(request)
    prompt = f"""You are the planning Supervisor for AgentFlow AI, a general-purpose assistant.
{GLOBAL_AGENT_PRINCIPLES}

Understand the user's intent from their natural-language request. Do not classify the user into a fixed set of domains or select a predefined domain workflow.
Decide independently:
- whether external/current/source-specific research is needed (stable, well-known facts may be answered directly)
- what information would make the answer complete
- whether analysis/synthesis and evidence verification add value
- the desired output and depth
- a concise research brief describing the information gaps to resolve

When research is needed, choose Researcher. The Researcher itself will use an LLM to create search queries from the full request, inspect results, fetch relevant pages, identify gaps and decide whether another search is needed. Never prescribe fixed domain templates.
Choose Analyst for synthesis, comparison, interpretation, calculations or plans when useful. Choose Verifier for material factual claims based on research. Reporter is always required.
Keep stages in this order: Researcher -> Analyst -> Verifier -> Reporter. Do not omit a stage that is clearly needed. Do not force research for a simple stable knowledge question.
Do not use hardcoded keyword lists or fixed domain templates to decide what the user means.
The configured LLM already has general knowledge: ordinary questions, explanations, definitions,
conversation and brainstorming should normally go directly to Reporter without web research.
When essential details are missing, ask a concise clarification rather than guessing or launching
an unfocused search. Research should augment the LLM's answer, never replace its general ability.

Return a structured plan for this exact request.

USER REQUEST:
{request}
"""
    try:
        planner = get_llm(temperature=0).with_structured_output(TaskPlan)
        proposed = planner.invoke(prompt)
        final_plan = _normalise_plan(proposed, request)
    except Exception as exc:
        print(f"[Supervisor] LLM planning failed; using generic fallback: {exc}")
        final_plan = fallback
    print("\n========== SUPERVISOR PLAN ==========")
    print("Goal:", final_plan.goal)
    print("Research:", final_plan.needs_research, "| Analysis:", final_plan.needs_analysis, "| Verification:", final_plan.needs_verification)
    print("Depth:", final_plan.response_depth, "| Output:", final_plan.output_format)
    for task in final_plan.tasks:
        print(f"\nAgent: {task.agent}\nTask: {task.description}")
    return {"plan": final_plan.model_dump()}
