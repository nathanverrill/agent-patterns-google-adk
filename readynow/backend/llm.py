"""
llm.py — provider-agnostic model wiring for ReadyNow!

Every agent in the app talks to whatever endpoint you point it at: Gemini,
OpenAI, or *any* OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp,
Together, Groq, OpenRouter, Fireworks, an internal gateway...). ADK's `LiteLlm`
wrapper does the protocol work; this module just resolves the environment into
the right constructor arguments and fails loudly when something is missing.

The contract, in priority order:

    LLM_BASE_URL   OpenAI-compatible endpoint (…/v1). Setting this switches the
                   app to the OpenAI wire protocol.
    LLM_API_KEY    Key for that endpoint. Falls back to OPENAI_API_KEY, then
                   GEMINI_API_KEY. Optional for local servers that ignore auth.
    LLM_MODEL      Model id, e.g. `gpt-4o-mini`, `llama3.1:8b`,
                   `gemini/gemini-2.5-flash`. A value that already carries a
                   LiteLLM provider prefix (`anthropic/…`, `groq/…`) is passed
                   through untouched.

With no LLM_* variables set the app falls back to Gemini via GEMINI_API_KEY, so
existing setups keep working unchanged.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from google.adk.models.lite_llm import LiteLlm

logger = logging.getLogger("readynow.llm")

DEFAULT_GEMINI_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# LiteLLM routes on the part before the slash. If a model id already names a
# provider, we must not prepend one of our own.
_KNOWN_PREFIXES = (
    "openai/", "gemini/", "vertex_ai/", "anthropic/", "azure/", "bedrock/",
    "groq/", "mistral/", "together_ai/", "openrouter/", "ollama/",
    "ollama_chat/", "deepseek/", "xai/", "fireworks_ai/", "cohere/",
    "huggingface/", "hosted_vllm/", "custom_openai/",
)


class ModelConfigError(RuntimeError):
    """Raised when the environment does not describe a usable model."""


@dataclass(frozen=True)
class ModelConfig:
    """A resolved, ready-to-instantiate model configuration."""

    model: str
    api_base: Optional[str]
    api_key: Optional[str]
    provider: str

    def kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"model": self.model}
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
            # Community endpoints vary in which sampling params they accept;
            # let LiteLLM discard the ones a given server rejects.
            kw["drop_params"] = True
        return kw

    def describe(self) -> str:
        target = self.api_base or "provider default endpoint"
        if not self.api_key or self.api_key == "not-needed":
            key_state = "no key required"
        else:
            key_state = "key set"
        return f"{self.model} via {self.provider} → {target} ({key_state})"


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _is_local(base_url: str) -> bool:
    return any(host in base_url for host in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"))


def resolve_model_config() -> ModelConfig:
    """Read the environment and return the model configuration to run with.

    Raises ModelConfigError with an actionable message rather than letting a
    misconfigured container fail later, mid-request, inside LiteLLM.
    """
    base_url = _first_env("LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")
    api_key = _first_env("LLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
    model = _first_env("LLM_MODEL", "AGENT_MODEL_NAME")

    # An explicit provider prefix always wins — the caller knows what they want.
    if model and model.startswith(_KNOWN_PREFIXES):
        provider = model.split("/", 1)[0]
        return ModelConfig(model=model, api_base=base_url, api_key=api_key, provider=provider)

    # OpenAI-compatible mode: any endpoint speaking the OpenAI wire protocol.
    if base_url or _first_env("LLM_API_KEY", "OPENAI_API_KEY"):
        model = model or DEFAULT_OPENAI_MODEL
        if base_url and not api_key:
            if not _is_local(base_url):
                raise ModelConfigError(
                    f"LLM_BASE_URL points at {base_url} but no key was provided. "
                    "Set LLM_API_KEY (local servers that ignore auth are exempt)."
                )
            # Ollama / LM Studio / vLLM ignore auth, but the OpenAI client
            # still insists on *some* key being present.
            api_key = "not-needed"
        return ModelConfig(
            model=f"openai/{model}",
            api_base=base_url,
            api_key=api_key,
            provider="openai-compatible",
        )

    # Default: Gemini through LiteLLM.
    if not api_key:
        raise ModelConfigError(
            "No model configured. Set GEMINI_API_KEY for Gemini, or set "
            "LLM_API_KEY (+ optional LLM_BASE_URL and LLM_MODEL) for any "
            "OpenAI-compatible endpoint. See readynow/.env.example."
        )
    return ModelConfig(
        model=f"gemini/{model}" if model else DEFAULT_GEMINI_MODEL,
        api_base=None,
        api_key=api_key,
        provider="gemini",
    )


# Resolved once at import so a bad configuration fails at container start —
# with a readable banner rather than a traceback from deep inside LiteLLM.
try:
    MODEL_CONFIG = resolve_model_config()
except ModelConfigError as err:
    import sys

    print("\n" + "=" * 72, file=sys.stderr)
    print("❌ ReadyNow! cannot start — model configuration incomplete.", file=sys.stderr)
    print(f"   {err}", file=sys.stderr)
    print("=" * 72 + "\n", file=sys.stderr)
    raise SystemExit(1)


def build_model() -> LiteLlm:
    """Return a LiteLlm bound to the configured endpoint (one per agent)."""
    return LiteLlm(**MODEL_CONFIG.kwargs())
