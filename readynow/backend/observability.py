"""
observability.py — full-lifecycle tracing for ADK agents.

Drop this file next to app.py. Then in app.py:

    from observability import configure_logging, attach_observability

    configure_logging()           # call BEFORE uvicorn.run / app startup
    attach_observability(root_agent)   # call AFTER root_agent is built

This wires before/after callbacks for every agent's:
  - agent lifecycle  (ENTER / EXIT  -> shows delegation between sub-agents)
  - model lifecycle  (prompt sent / response + tool calls the model decides on)
  - tool lifecycle   (tool name, args, and return value)

...and walks the whole sub_agent tree, so the search / critique / refine agents
inside your SequentialAgent all show up — not just the root. Existing callbacks
(your validation / boundary-policy logic) are preserved and chained, not replaced.
"""

import os
import sys
import logging
from typing import Any, Optional, Callable

logger = logging.getLogger("readynow.trace")


# --------------------------------------------------------------------------- #
# Logging setup (Docker-friendly: stdout, unbuffered-friendly, single handler)
# --------------------------------------------------------------------------- #
def configure_logging(level: int = logging.INFO, quiet_litellm: bool = True) -> None:
    """Send clean, structured logs to stdout so `docker logs` captures them.

    IMPORTANT — call this LATE: after your agents (and therefore the LiteLlm
    models) are constructed, right before attach_observability()/uvicorn.run().
    LiteLLM and some google libs add their OWN handler to the root logger when
    they're imported/built; if you configure logging before that happens you end
    up with two handlers and every line prints twice. Configuring last lets this
    function strip those extra handlers and leave exactly one.

    Also pass log_config=None to uvicorn.run() so uvicorn doesn't re-add handlers.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    # Remove EVERY existing handler — ours from a prior call AND any a third-party
    # library added via basicConfig()/addHandler(). This is what kills the
    # double-printing you saw in `docker logs`.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    if quiet_litellm:
        # LiteLLM ships its own handler (the "- LiteLLM:INFO:" lines) AND is very
        # chatty. Strip its private handler so it logs once through root, and
        # dial it down to warnings. Setting LITELLM_LOG belt-and-suspenders.
        os.environ.setdefault("LITELLM_LOG", "WARNING")
        for name in ("LiteLLM", "litellm"):
            lg = logging.getLogger(name)
            for h in list(lg.handlers):
                lg.removeHandler(h)
            lg.propagate = True
            lg.setLevel(logging.WARNING)

    # Keep uvicorn chatter at a sane level but still visible.
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    # Uncomment for EXTREMELY verbose internal ADK events (very noisy):
    # logging.getLogger("google_adk").setLevel(logging.DEBUG)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _short(value: Any, limit: int = 600) -> str:
    s = str(value).replace("\n", " ").strip()
    return s if len(s) <= limit else s[:limit] + " …[truncated]"


def _name(ctx: Any) -> str:
    # CallbackContext and ToolContext both expose agent_name.
    return getattr(ctx, "agent_name", None) or "agent"


def _inv(ctx: Any) -> str:
    inv = getattr(ctx, "invocation_id", "") or ""
    return inv[:8] if inv else "--------"


# --------------------------------------------------------------------------- #
# Agent lifecycle
# --------------------------------------------------------------------------- #
def trace_before_agent(callback_context) -> Optional[Any]:
    logger.info("[%s] ┌─ ENTER agent[%s]", _inv(callback_context), _name(callback_context))
    return None


def trace_after_agent(callback_context) -> Optional[Any]:
    logger.info("[%s] └─ EXIT  agent[%s]", _inv(callback_context), _name(callback_context))
    return None


# --------------------------------------------------------------------------- #
# Model (LLM) lifecycle
# --------------------------------------------------------------------------- #
def trace_before_model(callback_context, llm_request) -> Optional[Any]:
    inv, name = _inv(callback_context), _name(callback_context)
    try:
        contents = getattr(llm_request, "contents", None) or []
        if contents:
            parts = getattr(contents[-1], "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None) if not isinstance(part, dict) else part.get("text")
                if text:
                    logger.info("[%s] │  → LLM  [%s] prompt: %s", inv, name, _short(text))
    except Exception as e:
        logger.warning("[%s] │  (before_model trace error [%s]: %s)", inv, name, e)
    return None


def trace_after_model(callback_context, llm_response) -> Optional[Any]:
    inv, name = _inv(callback_context), _name(callback_context)
    # Skip streaming partials to avoid one log line per token.
    if getattr(llm_response, "partial", False):
        return None
    try:
        content = getattr(llm_response, "content", None)
        parts = (getattr(content, "parts", None) or []) if content else []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                logger.info("[%s] │  ← LLM  [%s] response: %s", inv, name, _short(text))
            fc = getattr(part, "function_call", None)
            if fc:
                args = dict(getattr(fc, "args", {}) or {})
                logger.info("[%s] │  ← LLM  [%s] wants tool: %s(%s)",
                            inv, name, getattr(fc, "name", "?"), _short(args))
    except Exception as e:
        logger.warning("[%s] │  (after_model trace error [%s]: %s)", inv, name, e)
    return None


# --------------------------------------------------------------------------- #
# Tool lifecycle
# --------------------------------------------------------------------------- #
def trace_before_tool(tool, args, tool_context) -> Optional[Any]:
    tool_name = getattr(tool, "name", None) or str(tool)
    logger.info("[%s] │  ⚙ TOOL  [%s] call %s args=%s",
                _inv(tool_context), _name(tool_context), tool_name, _short(args))
    return None


def trace_after_tool(tool, args, tool_context, tool_response) -> Optional[Any]:
    tool_name = getattr(tool, "name", None) or str(tool)
    logger.info("[%s] │  ⚙ TOOL  [%s] %s → %s",
                _inv(tool_context), _name(tool_context), tool_name, _short(tool_response))
    return None


# --------------------------------------------------------------------------- #
# Chaining + attachment
# --------------------------------------------------------------------------- #
def _chain(*callbacks: Optional[Callable]) -> Optional[Callable]:
    """Run callbacks in order. If any returns non-None, short-circuit with it
    (this is how ADK model callbacks block / override a request or response)."""
    cbs = [c for c in callbacks if c is not None]
    if not cbs:
        return None
    if len(cbs) == 1:
        return cbs[0]

    def chained(*args, **kwargs):
        for cb in cbs:
            result = cb(*args, **kwargs)
            if result is not None:
                return result
        return None

    return chained


def attach_observability(agent, _depth: int = 0) -> None:
    """Attach tracing callbacks to an agent and, recursively, all its sub_agents.

    - Tracing runs FIRST, so you log the real user prompt before any of your
      existing callbacks rewrite it.
    - Existing callbacks (validation, boundary policy) are preserved.
    """
    name = getattr(agent, "name", agent.__class__.__name__)
    logger.info("attaching observability to %s%s", "  " * _depth, name)

    # Agent-level callbacks exist on every agent type (incl. SequentialAgent).
    if hasattr(agent, "before_agent_callback"):
        agent.before_agent_callback = _chain(trace_before_agent, getattr(agent, "before_agent_callback", None))
    if hasattr(agent, "after_agent_callback"):
        agent.after_agent_callback = _chain(trace_after_agent, getattr(agent, "after_agent_callback", None))

    # Model callbacks only exist on LLM-backed agents (not workflow agents).
    if hasattr(agent, "before_model_callback"):
        agent.before_model_callback = _chain(trace_before_model, getattr(agent, "before_model_callback", None))
    if hasattr(agent, "after_model_callback"):
        agent.after_model_callback = _chain(trace_after_model, getattr(agent, "after_model_callback", None))

    # Tool callbacks only matter for agents that have tools.
    if hasattr(agent, "before_tool_callback"):
        agent.before_tool_callback = _chain(trace_before_tool, getattr(agent, "before_tool_callback", None))
    if hasattr(agent, "after_tool_callback"):
        agent.after_tool_callback = _chain(trace_after_tool, getattr(agent, "after_tool_callback", None))

    for sub in (getattr(agent, "sub_agents", None) or []):
        attach_observability(sub, _depth + 1)
