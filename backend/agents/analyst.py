"""Analyst: synthesize retrieved evidence into the user's requested deliverable."""
from agents.common import get_llm, GLOBAL_AGENT_PRINCIPLES


def analyst_node(state):
    print("\n[Analyst] Synthesizing evidence for the requested outcome...")
    request = state.get("user_request", "")
    plan = state.get("plan") or {}
    research = state.get("research") or "No research evidence was collected."
    sources = state.get("research_sources") or []
    verification = state.get("verification") or "Verification has not yet run."
    source_context = "\n".join(
        f"[{i}] {item.get('title') or 'Source'} | {item.get('url')} | {item.get('published_date') or ''}"
        for i, item in enumerate(sources, 1) if item.get("fetch_status") == "Completed"
    )
    prompt = f"""You are the Analyst agent in AgentFlow AI.
{GLOBAL_AGENT_PRINCIPLES}

Produce the analysis or structured deliverable that best fulfils the user's request. Do not assume a
fixed domain or output template. If the user asks for a trip, produce a practical itinerary; if they
ask for a comparison, compare the requested dimensions; if they ask for an explanation, explain the
concept; if they ask for recommendations, give reasoned recommendations. These are examples, not
hardcoded categories or required headings.

Use the Researcher's evidence as the factual basis. You may make transparent calculations or clearly
labelled inferences, but do not invent external facts. Distinguish verified information, interpretation,
estimates and unknowns. Respect any caveats in verification. If the evidence is too thin, identify what
cannot be concluded instead of padding the answer.

USER REQUEST:
{request}

SUPERVISOR OUTPUT REQUIREMENT:
{plan.get('output_format', 'Answer the request')}

REQUESTED DEPTH:
{plan.get('response_depth', 'normal')}

RESEARCH EVIDENCE:
{research}

SOURCE INDEX:
{source_context or 'No successfully fetched source index.'}

VERIFICATION NOTES:
{verification}

Create a useful, substantive analysis that directly advances the requested deliverable. Avoid repeating
all source material and do not output a URL dump.
"""
    try:
        analysis = get_llm(temperature=0).invoke(prompt).content
    except Exception as exc:
        analysis = f"Analysis generation failed: {type(exc).__name__}: {exc}\n\nAvailable research:\n{research[:10000]}"
    return {"analysis": analysis}
