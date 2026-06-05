"""
LLM abstraction layer.

Provides a single `LLM` base class plus two concrete backends:

  MockLLM                 — deterministic rule-based responses (no dependencies)
  OpenAICompatibleLLM     — any OpenAI-compatible /v1/chat/completions endpoint

Usage example
-------------
    from llm.llm import MockLLM, OpenAICompatibleLLM

    llm = MockLLM()                                              # default, zero setup
    llm = OpenAICompatibleLLM(
        model="deepseek-v4-pro",
        base_url="https://opencode.ai/zen/go/v1",               # OpenCode Go
        api_key="...",
    )
    llm = OpenAICompatibleLLM(
        model="qwen/qwen3-30b-a3b",
        base_url="http://localhost:1234/v1",                    # LM Studio (no key)
    )

    response = llm.generate("Summarise this text: ...")
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any


class LLM(ABC):
    """Abstract base class for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a text response for the given prompt.

        Parameters
        ----------
        prompt:
            The full prompt string to send to the model.
        **kwargs:
            Backend-specific overrides (temperature, max_tokens, etc.).

        Returns
        -------
        str
            The model's text response.
        """

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        """
        Convenience wrapper that formats a system + user message pair
        into a single prompt string and calls generate().
        """
        combined = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        return self.generate(combined, **kwargs)


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockLLM(LLM):
    """
    A deterministic, rule-based mock LLM.

    Produces structured but realistic-looking outputs so the full agent
    loop can be exercised without any external API keys or GPU.

    The mock recognises keywords in the prompt and returns templated
    responses that are well-formed enough for downstream parsers.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:  # noqa: ARG002
        p = prompt.lower()

        # ---- planning ----
        if "plan" in p or "break" in p or "tasks" in p:
            return json.dumps([
                {"step": 1, "action": "vector_search",   "query": "{goal}"},
                {"step": 2, "action": "extract_entities", "query": "from_document"},
                {"step": 3, "action": "vector_search",   "query": "key entities relationships"},
                {"step": 4, "action": "extract_entities", "query": "from_last_document"},
                {"step": 5, "action": "vector_search",   "query": "connections patterns"},
                {"step": 6, "action": "extract_entities", "query": "from_last_document"},
                {"step": 7, "action": "summarize",       "query": "from_last_document"},
            ])

        # ---- ontology / entity type discovery ----
        if "ontolog" in p or ("entity" in p and "type" in p):
            return json.dumps({
                "entity_types": ["concept", "entity", "topic", "event", "location", "person", "organisation"],
                "relation_triples": [
                    ["concept", "related_to", "concept"],
                    ["person", "works_at", "organisation"],
                    ["entity", "associated_with", "entity"],
                    ["event", "involves", "person"],
                ],
                "aliases": {},
            })

        # ---- entity extraction ----
        if "extract" in p and "entit" in p:
            return json.dumps([
                {"name": "Entity A", "type": "concept"},
                {"name": "Entity B", "type": "entity"},
                {"name": "Person X", "type": "person"},
                {"name": "Organisation Y", "type": "organisation"},
            ])

        # ---- relationship extraction ----
        if "relation" in p and ("extract" in p or "triple" in p or "find" in p):
            return json.dumps([
                {"source": "Entity A", "relation": "related_to", "target": "Entity B", "confidence": 0.7},
                {"source": "Person X", "relation": "works_at",   "target": "Organisation Y", "confidence": 0.8},
            ])

        # ---- summarisation ----
        if "summar" in p:
            return (
                "The document describes key concepts and entities and their "
                "relationships. Several important connections between entities "
                "were identified across the corpus."
            )

        # ---- hypothesis generation ----
        if "hypothesis" in p or "hypothes" in p:
            return json.dumps({
                "statement": (
                    "Entity A and Organisation Y are indirectly connected through "
                    "Person X, suggesting a latent structural relationship in the corpus."
                ),
                "entities": ["Entity A", "Person X", "Organisation Y"],
                "confidence": 0.65,
                "type": "structural_relationship",
            })

        # ---- validation ----
        if "valid" in p or "verif" in p or "confirm" in p:
            return json.dumps({
                "verdict": "SUPPORTED",
                "confidence_delta": 0.12,
                "reasoning": (
                    "Multiple document passages corroborate the proposed connection. "
                    "Co-occurrence patterns and explicit mentions both support the hypothesis."
                ),
                "new_evidence": [
                    "Document passage mentions both Entity A and Organisation Y in the same context.",
                    "Person X is referenced in relation to both entities.",
                ],
            })

        # ---- tool selection ----
        if "choose" in p or "select" in p or "tool" in p:
            return json.dumps({
                "tool": "vector_search",
                "input": "key entities and their relationships",
                "reasoning": "Start broad to surface all relevant document passages.",
            })

        # ---- graph path interpretation ----
        if "path" in p and ("graph" in p or "connect" in p or "hidden" in p):
            return (
                "The entities are indirectly connected through a shared intermediary, "
                "suggesting an implicit structural relationship in the document corpus."
            )

        # ---- default ----
        return (
            "I have analysed the available information. "
            "The evidence suggests meaningful structural relationships among "
            "the identified entities in the corpus."
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------

class OpenAICompatibleLLM(LLM):
    """
    Generic backend for any OpenAI-compatible /v1/chat/completions endpoint.

    Works with OpenAI, OpenRouter, Groq, Together, DeepSeek, Ollama,
    LM Studio, vLLM, OpenCode Go, and any other service that speaks the
    same protocol.

    Parameters
    ----------
    model:
        Model identifier, e.g. "gpt-4o", "deepseek-v4-pro", "llama3".
    base_url:
        Base URL of the endpoint, e.g. "https://api.openai.com/v1",
        "https://openrouter.ai/api/v1", "http://localhost:11434/v1".
        Default: "http://localhost:1234/v1" (LM Studio default).
    api_key:
        Optional API key.  Falls back to the LLM_API_KEY environment
        variable.  Omit for keyless local servers (LM Studio, Ollama).
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens to generate per response.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:1234/v1",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import requests  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "requests package is required for OpenAICompatibleLLM.  "
                "Install it with: pip install requests"
            ) from exc

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key: str | None = api_key or os.environ.get("LLM_API_KEY")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if resolved_key:
            self._headers["Authorization"] = f"Bearer {resolved_key}"

    def _post(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        import requests  # type: ignore[import]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        body = resp.json()

        choice = body.get("choices", [{}])[0]
        message: dict[str, str] = choice.get("message", {})

        # Some reasoning/thinking models return output in `reasoning_content`
        # and leave `content` as null.
        content: str = message.get("content") or ""
        if not content:
            content = message.get("reasoning_content", "") or ""
        return content

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        return self._post(
            [{"role": "user", "content": prompt}],
            temperature,
            max_tokens,
        )

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        return self._post(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature,
            max_tokens,
        )
