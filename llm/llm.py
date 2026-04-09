"""
LLM abstraction layer.

Provides a single `LLM` base class plus four concrete backends:

  MockLLM         — deterministic rule-based responses (no dependencies)
  OllamaLLM       — local Ollama REST API  (requires running Ollama server)
  OpenRouterLLM   — OpenRouter API; access 100+ models with one key
  LMStudioLLM     — local LM Studio server (OpenAI-compatible, no API key needed)

Usage example
-------------
    from llm.llm import MockLLM, OllamaLLM, OpenRouterLLM, LMStudioLLM

    llm = MockLLM()                                      # default, zero setup
    llm = OllamaLLM(model="llama3")                     # needs Ollama running locally
    llm = OpenRouterLLM(model="mistralai/mistral-7b-instruct")  # needs OPENROUTER_API_KEY
    llm = LMStudioLLM(model="qwen/qwen3-30b-a3b")       # needs LM Studio server on :1234

    response = llm.generate("Summarise this text: ...")
"""

from __future__ import annotations

import os
import json
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
# Mock backend — fully self-contained, no API keys required
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
# Ollama backend (local models via REST API)
# ---------------------------------------------------------------------------

class OllamaLLM(LLM):
    """
    Ollama local model backend.

    Requires Ollama running at http://localhost:11434 (default).
    Any model pulled via `ollama pull <model>` can be used.

    Parameters
    ----------
    model:
        Ollama model name, e.g. "llama3", "mistral", "phi3".
    base_url:
        Ollama API base URL.
    temperature:
        Sampling temperature.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
    ) -> None:
        try:
            import requests  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "requests package is required for OllamaLLM.  "
                "Install it with: pip install requests"
            ) from exc

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs: Any) -> str:
        import requests  # type: ignore[import]

        temperature = kwargs.get("temperature", self.temperature)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


# ---------------------------------------------------------------------------
# OpenRouter backend
# ---------------------------------------------------------------------------

class OpenRouterLLM(LLM):
    """
    OpenRouter backend — unified access to 100+ models via one API key.

    Uses the OpenRouter Chat Completions REST endpoint directly via
    `requests`; no `openai` SDK is required.

    API key
    -------
    Pass it directly as ``api_key``, or set the ``OPENROUTER_API_KEY``
    environment variable (the constructor checks both; the explicit argument
    takes precedence).

    Popular model strings
    ---------------------
    Free / low-cost:
      "mistralai/mistral-7b-instruct"
      "meta-llama/llama-3-8b-instruct"
      "google/gemma-3-27b-it:free"
      "microsoft/phi-3-mini-128k-instruct:free"

    Higher capability:
      "anthropic/claude-3.5-sonnet"
      "mistralai/mixtral-8x22b-instruct"
      "deepseek/deepseek-r1"

    See the full list at https://openrouter.ai/models

    Parameters
    ----------
    model:
        OpenRouter model identifier (provider/model-name).
    api_key:
        Your OpenRouter API key.  Falls back to OPENROUTER_API_KEY env var.
    temperature:
        Sampling temperature (default 0.3 for consistent reasoning).
    max_tokens:
        Maximum tokens to generate per response.
    site_url:
        Optional.  Sent as the HTTP-Referer header so OpenRouter can
        attribute usage to your application.
    site_name:
        Optional.  Sent as X-Title header (shown in OpenRouter dashboards).
    """

    _BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = "mistralai/mistral-7b-instruct",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        site_url: str = "",
        site_name: str = "KnowledgeDiscoveryAgent",
    ) -> None:
        try:
            import requests  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "requests package is required for OpenRouterLLM.  "
                "Install it with: pip install requests"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenRouter API key found.  "
                "Pass api_key= or set the OPENROUTER_API_KEY environment variable."
            )

        self._api_key = resolved_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }
        if site_url:
            self._headers["HTTP-Referer"] = site_url
        if site_name:
            self._headers["X-Title"] = site_name

    def _post(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        import requests  # type: ignore[import]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(self._BASE_URL, headers=self._headers, json=payload, timeout=120)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        # Some reasoning/thinking models (e.g. minimax-m2.5, deepseek-r1) return
        # their output in `reasoning_content` and leave `content` as null.
        # Fall back to `reasoning_content` so these models work transparently.
        content = message.get("content") or ""
        if not content:
            content = message.get("reasoning_content", "") or ""
        return content

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        return self._post([{"role": "user", "content": prompt}], temperature, max_tokens)

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._post(messages, temperature, max_tokens)


# ---------------------------------------------------------------------------
# HuggingFace local backend
# ---------------------------------------------------------------------------

class HuggingFaceLLM(LLM):
    """
    Local HuggingFace text-generation pipeline backend.

    Requires:
      - `transformers` and `torch` packages

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. "microsoft/phi-2".
    max_new_tokens:
        Maximum number of new tokens to generate.
    device:
        "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        max_new_tokens: int = 512,
        device: str = "cpu",
    ) -> None:
        try:
            from transformers import pipeline  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for HuggingFaceLLM.  "
                "Install with: pip install transformers torch"
            ) from exc

        self._pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        result = self._pipeline(prompt, return_full_text=False)
        return result[0]["generated_text"]


# ---------------------------------------------------------------------------
# LM Studio backend (local server, OpenAI-compatible, no API key required)
# ---------------------------------------------------------------------------

class LMStudioLLM(LLM):
    """
    LM Studio local server backend.

    LM Studio exposes an OpenAI-compatible REST API on ``http://localhost:1234/v1``
    by default.  No API key is required — the server accepts any value (or none).

    Start the server inside LM Studio: Developer tab → Start Server.

    Parameters
    ----------
    model:
        Model identifier exactly as shown in LM Studio's model list.
        Example: ``"qwen/qwen3-30b-a3b"``
    base_url:
        Base URL of the LM Studio server.  Change this only if you have
        configured a non-default port.
    temperature:
        Sampling temperature (default 0.3 for consistent reasoning).
    max_tokens:
        Maximum tokens to generate per response (default 4096).
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-30b-a3b",
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import requests  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "requests package is required for LMStudioLLM.  "
                "Install it with: pip install requests"
            ) from exc

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

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
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,  # LM Studio on large models can be slow — allow 5 min
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""

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
