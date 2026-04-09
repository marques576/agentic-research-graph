"""
Tool interface and all concrete tool implementations.

Every tool inherits from the abstract base class `Tool` which enforces
a single .run(input_data) method.  This makes tools interchangeable and
easy to register in a tool registry.

Tools implemented here:
  MultimodalVectorSearchTool  – semantic search using Qwen3-VL-Embedding
                                 (native multimodal: text, images, video in one
                                 shared vector space) + FAISS ANN index.
                                 Falls back to TF-IDF keyword search when the
                                 model is not installed.
  Qwen3VLRerankerTool         – re-ranks candidates with Qwen3-VL-Reranker
                                 (cross-encoder, multimodal query+document).
                                 Falls back to returning candidates unchanged.
  ReadDocumentTool            – retrieve full document text by id
  ExtractEntitiesTool         – NLP entity extraction from raw text
  SummarizeTool               – LLM-powered text summarisation
  GraphNeighborsTool          – returns all entities connected to a graph node
  GraphShortestPathTool       – finds shortest path between two graph nodes

Both Qwen3-VL models run fully locally via HuggingFace transformers.
No API key is required.  Weights are downloaded to ~/.cache/huggingface
on first use, or you can pass a local directory path as model_id.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from memory.memory import AgentMemory


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Tool(ABC):
    """
    Abstract base class for all tools.

    All tools expose a single entry point: run(input_data) -> Any.
    input_data can be a string, dict, or whatever the tool expects;
    the type is documented in each concrete class.
    """

    name: str = "base_tool"
    description: str = "Base tool — override in subclasses."

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the tool and return the result."""

    def __repr__(self) -> str:
        return f"Tool({self.name})"


# ---------------------------------------------------------------------------
# MultimodalVectorSearchTool  (replaces VectorSearchTool)
# ---------------------------------------------------------------------------

class MultimodalVectorSearchTool(Tool):
    """
    Semantic search using Qwen3-VL-Embedding — a native multimodal embedding
    model that encodes text, images, and video in a single shared vector space.

    Architecture
    ------------
    - **Index side** (build time): the ``MultimodalIngestionPipeline`` has
      already embedded every document chunk / image / video frame sequence
      with ``Qwen3VLEmbedder.process()``.  The resulting float32 numpy array
      plus a FAISS ``IndexFlatIP`` are passed in via the ``pipeline`` argument.
    - **Query side** (run time): a text query is embedded with the same
      ``Qwen3VLEmbedder`` (optionally with a task instruction) and the
      top-k nearest neighbours are retrieved from the FAISS index.
    - **Fallback**: if Qwen3-VL-Embedding or FAISS is not installed, a simple
      TF-IDF keyword overlap search is used instead.

    Models (local, no API key)
    --------------------------
    ``Qwen/Qwen3-VL-Embedding-2B``  —  ~5 GB VRAM, CPU-offloadable (default)
    ``Qwen/Qwen3-VL-Embedding-8B``  —  ~18 GB VRAM, best quality

    On first use, weights download to ``~/.cache/huggingface``.
    You can also pass an absolute local directory path as ``embedding_model_id``.

    Parameters
    ----------
    documents : dict[str, str]
        doc_id → extracted text (used for keyword fallback and snippets).
    pipeline : MultimodalIngestionPipeline | None
        Ingestion pipeline that already built embeddings and chunk metadata.
        Passing this avoids re-embedding at search time.
    top_k : int
        Number of results to return per query.
    embedding_model_id : str
        Local path or HuggingFace id for Qwen3-VL-Embedding.
    query_instruction : str | None
        Task instruction prepended to every query embedding.
        Improves recall by ~1–5 %.  Pass ``None`` to disable.
    embedding_kwargs : dict
        Extra kwargs forwarded to ``Qwen3VLEmbedder`` (e.g. ``torch_dtype``,
        ``attn_implementation``).
    """

    name = "vector_search"
    description = (
        "Multimodal semantic search over all ingested content "
        "(text, images, video, audio transcripts, …) using "
        "Qwen3-VL-Embedding running locally. Input: query string."
    )

    _DEFAULT_QUERY_INSTRUCTION = (
        "Retrieve relevant documents, images, or video segments that contain "
        "information about the entities, relationships, or facts in the query."
    )

    def __init__(
        self,
        documents: dict[str, str],
        pipeline: Any = None,
        top_k: int = 5,
        embedding_model_id: str = "Qwen/Qwen3-VL-Embedding-2B",
        query_instruction: str | None = None,
        embedding_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.documents        = documents
        self.top_k            = top_k
        self.embedding_model_id = embedding_model_id
        self.query_instruction  = query_instruction or self._DEFAULT_QUERY_INSTRUCTION
        self.embedding_kwargs   = embedding_kwargs or {}

        self._chunk_index: list[dict[str, Any]] = []
        self._faiss_index: Any = None

        # Reuse pre-built embeddings from the ingestion pipeline (fastest path)
        if pipeline is not None and getattr(pipeline, "chunk_index", None):
            self._chunk_index = pipeline.chunk_index
            self._faiss_index = pipeline.build_faiss_index()

        # If no pipeline was supplied but we have documents, embed them now
        if self._faiss_index is None and documents:
            self._build_index_from_documents()

    # ------------------------------------------------------------------
    # Cold-start index building (no pipeline available)
    # ------------------------------------------------------------------

    def _build_index_from_documents(self) -> None:
        """
        Chunk all documents, embed with Qwen3-VL-Embedding, and build FAISS.
        This is the slow path; prefer passing a pre-built pipeline.
        """
        try:
            from ingestion.multimodal_ingestion import (
                get_embedder,
                _chunk_text,
                _embed_input_for_text,
            )
            import faiss       # type: ignore[import]
            import numpy as np # type: ignore[import]

            embedder = get_embedder(self.embedding_model_id, **self.embedding_kwargs)

            embed_inputs: list[dict[str, Any]] = []
            for doc_id, text in self.documents.items():
                for i, chunk in enumerate(
                    _chunk_text(text, chunk_size=1500, overlap=200)
                ):
                    self._chunk_index.append({
                        "doc_id":       doc_id,
                        "chunk_id":     f"{doc_id}__chunk{i}",
                        "modality":     "text",
                        "text_snippet": chunk[:300],
                    })
                    embed_inputs.append(_embed_input_for_text(chunk))

            if not embed_inputs:
                return

            vecs = embedder.process(embed_inputs).cpu().float().numpy()
            vecs = vecs.astype(np.float32)
            faiss.normalize_L2(vecs)
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            self._faiss_index = index

        except ImportError:
            self._faiss_index = None
        except Exception:
            self._faiss_index = None

    # ------------------------------------------------------------------
    # Keyword fallback
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str) -> list[dict[str, Any]]:
        """TF-IDF keyword overlap — zero-dependency fallback."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for doc_id, text in self.documents.items():
            doc_terms = set(re.findall(r"\w+", text.lower()))
            score = len(query_terms & doc_terms) / max(len(query_terms), 1)
            scored.append((score, doc_id, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"doc_id": did, "chunk_id": f"{did}__kw", "score": s,
             "snippet": txt[:400], "modality": "text"}
            for s, did, txt in scored[: self.top_k]
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, input_data: str) -> list[dict[str, Any]]:
        """
        Parameters
        ----------
        input_data : str
            Natural-language search query (any language).

        Returns
        -------
        list of dicts — keys: doc_id, chunk_id, score, snippet, modality
        """
        query = str(input_data)

        if self._faiss_index is None or not self._chunk_index:
            return self._keyword_search(query)

        try:
            from ingestion.multimodal_ingestion import get_embedder
            import faiss       # type: ignore[import]
            import numpy as np # type: ignore[import]

            embedder = get_embedder(self.embedding_model_id, **self.embedding_kwargs)

            # Embed the text query with a task instruction
            query_input = [{
                "text":        query,
                "instruction": self.query_instruction,
            }]
            qvec = embedder.process(query_input).cpu().float().numpy().astype(np.float32)
            faiss.normalize_L2(qvec)

            k = min(self.top_k, len(self._chunk_index))
            distances, indices = self._faiss_index.search(qvec, k)

            results: list[dict[str, Any]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                entry = self._chunk_index[int(idx)]
                results.append({
                    "doc_id":     entry["doc_id"],
                    "chunk_id":   entry["chunk_id"],
                    "score":      float(dist),
                    "snippet":    entry.get("text_snippet", "")[:400],
                    "chunk_text": entry.get("text", ""),   # full chunk text
                    "modality":   entry.get("modality", "unknown"),
                })
            return results

        except Exception:
            return self._keyword_search(query)


# ---------------------------------------------------------------------------
# Qwen3VLRerankerTool  (optional re-ranking step)
# ---------------------------------------------------------------------------

class Qwen3VLRerankerTool(Tool):
    """
    Re-rank retrieval candidates using Qwen3-VL-Reranker locally.

    Qwen3-VL-Reranker is a cross-encoder that accepts (query, document) pairs
    where both sides can be text, images, or video.  It scores each pair and
    returns relevance probabilities, which are used to reorder the candidates
    from ``MultimodalVectorSearchTool``.

    Models (local, no API key)
    --------------------------
    ``Qwen/Qwen3-VL-Reranker-2B``  —  ~5 GB VRAM (default)
    ``Qwen/Qwen3-VL-Reranker-8B``  —  ~18 GB VRAM, best quality

    Parameters
    ----------
    model_id : str
        Local path or HuggingFace id for Qwen3-VL-Reranker.
    instruction : str | None
        Task instruction passed to the reranker.
    reranker_kwargs : dict
        Extra kwargs forwarded to ``Qwen3VLReranker`` (e.g. ``torch_dtype``).
    """

    name = "rerank"
    description = (
        "Re-ranks search results using Qwen3-VL-Reranker (local cross-encoder). "
        "Accepts text, image, or video documents. "
        "Input: dict with 'query' (str) and 'candidates' (list of dicts). "
        "Returns re-ranked list with 'reranker_score' added to each item."
    )

    _DEFAULT_INSTRUCTION = (
        "Given a research query, score how relevant the document is for "
        "finding information about the entities, relationships, or facts "
        "described in the query."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-Reranker-2B",
        instruction: str | None = None,
        reranker_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_id         = model_id
        self.instruction      = instruction or self._DEFAULT_INSTRUCTION
        self.reranker_kwargs  = reranker_kwargs or {}
        self._reranker: Any   = None   # lazy-loaded on first use

    def _load(self) -> bool:
        """Lazy-load Qwen3VLReranker. Returns True on success."""
        if self._reranker is not None:
            return True
        try:
            from ingestion.multimodal_ingestion import get_reranker
            self._reranker = get_reranker(self.model_id, **self.reranker_kwargs)
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def run(self, input_data: dict[str, Any] | str) -> list[dict[str, Any]]:
        """
        Parameters
        ----------
        input_data : dict
            ``query``      – str or dict with keys ``text`` / ``image`` / ``video``
            ``candidates`` – list of dicts, each with at minimum a ``snippet``
                             key (str).  Optionally ``image`` or ``video`` for
                             multimodal documents.

        Returns
        -------
        Same candidates list, sorted descending by ``reranker_score``.
        Falls back to original order if reranker cannot be loaded.
        """
        if isinstance(input_data, str):
            return []

        raw_query  = input_data.get("query", "")
        candidates: list[dict[str, Any]] = input_data.get("candidates", [])

        if not candidates:
            return []

        if not self._load():
            return candidates   # graceful fallback: return unsorted

        # Build query multimodal object
        if isinstance(raw_query, str):
            query_obj: dict[str, Any] = {"text": raw_query}
        else:
            query_obj = dict(raw_query)

        # Build document multimodal objects from candidates
        documents: list[dict[str, Any]] = []
        for c in candidates:
            doc: dict[str, Any] = {}
            if c.get("snippet"):
                doc["text"] = c["snippet"]
            if c.get("image"):
                doc["image"] = c["image"]
            if c.get("video"):
                doc["video"] = c["video"]
            if not doc:
                doc["text"] = str(c)
            documents.append(doc)

        reranker_input = {
            "instruction": self.instruction,
            "query":       query_obj,
            "documents":   documents,
        }

        try:
            scores: list[float] = self._reranker.process(reranker_input)
            for c, s in zip(candidates, scores):
                c["reranker_score"] = s
            return sorted(
                candidates,
                key=lambda x: x.get("reranker_score", 0.0),
                reverse=True,
            )
        except Exception:
            return candidates



# ---------------------------------------------------------------------------
# ReadDocumentTool
# ---------------------------------------------------------------------------

class ReadDocumentTool(Tool):
    """
    Retrieves the full text of a document by its id.

    Parameters
    ----------
    documents : dict[str, str]
        Mapping of doc_id -> full text.
    """

    name = "read_document"
    description = "Retrieves the full text of a document. Input: doc_id (string)."

    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents

    def run(self, input_data: str) -> dict[str, Any]:
        """
        Parameters
        ----------
        input_data : str
            The document id, or a natural-language description containing the
            document name.  Lookup order:
              1. Exact match on doc_id key.
              2. Case-insensitive exact match.
              3. Any key that is a substring of input_data (e.g. the LLM wrote
                 a sentence containing the filename).
              4. Any key whose words are all present in input_data.

        Returns
        -------
        dict with keys: doc_id, text, found (bool)
        """
        doc_id = str(input_data).strip()

        # 1. exact match
        text = self.documents.get(doc_id)
        if text is not None:
            return {"doc_id": doc_id, "found": True, "text": text}

        # 2. case-insensitive exact match
        lower = doc_id.lower()
        for key, val in self.documents.items():
            if key.lower() == lower:
                return {"doc_id": key, "found": True, "text": val}

        # 3. any stored key that appears as a substring in the input
        for key, val in self.documents.items():
            if key.lower() in lower:
                return {"doc_id": key, "found": True, "text": val}

        # 4. all words of a stored key appear somewhere in the input
        lower_words = set(lower.replace("_", " ").split())
        for key, val in self.documents.items():
            key_words = set(key.lower().replace("_", " ").split())
            if key_words and key_words.issubset(lower_words):
                return {"doc_id": key, "found": True, "text": val}

        return {"doc_id": doc_id, "found": False, "text": ""}


# ---------------------------------------------------------------------------
# ExtractEntitiesTool
# ---------------------------------------------------------------------------

import json as _json


class ExtractEntitiesTool(Tool):
    """
    Extracts named entities from text using the LLM, guided by the entity
    types learned in the domain ontology.

    If no LLM is provided, falls back to spaCy (if installed) for generic
    NER with standard types (ORG, PERSON, GPE, …).

    Parameters
    ----------
    llm : Any
        LLM backend (any object with a .generate(prompt) -> str method).
        Required for ontology-aware extraction; without it spaCy is used.
    ontology : DomainOntology | None
        Live ontology object.  Its entity_types are injected into the prompt
        so the LLM labels entities using the learned vocabulary.
    use_spacy : bool
        If True, attempt to load spaCy en_core_web_sm as a fallback when
        the LLM is unavailable.
    """

    name = "extract_entities"
    description = (
        "Extracts named entities from text using the LLM and the learned "
        "domain ontology types. Input: text string."
    )

    def __init__(
        self,
        llm: Any = None,
        ontology: Any = None,
        use_spacy: bool = True,
    ) -> None:
        self._llm = llm
        self._ontology = ontology
        self._nlp: Any = None
        if use_spacy:
            try:
                import spacy  # type: ignore
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                self._nlp = None

    def _llm_extract(self, text: str) -> list[dict[str, str]]:
        """Ask the LLM to extract entities, constrained to ontology types."""
        core_types: list[str] = []
        extra_types: list[str] = []
        if self._ontology is not None:
            core_types = sorted(self._ontology.core_entity_types)
            extra_types = sorted(
                self._ontology.entity_types - self._ontology.core_entity_types
            )

        # Build a two-tier type hint:
        #   - Core types (user-specified): LLM MUST tag all entities of these types.
        #   - Additional types (discovered): LLM MAY also use these.
        if core_types and extra_types:
            type_hint = (
                f"You MUST tag all entities of these required types: "
                f"{', '.join(core_types)}.\n"
                f"You may also use these additional types if present: "
                f"{', '.join(extra_types)}."
            )
        elif core_types:
            type_hint = (
                f"You MUST tag all entities of these required types: "
                f"{', '.join(core_types)}.\n"
                f"You may use other appropriate types if needed."
            )
        elif extra_types:
            type_hint = (
                f"Use ONLY these entity types: {', '.join(extra_types)}."
            )
        else:
            type_hint = "Use appropriate entity type labels (lowercase, singular)."

        prompt = (
            f"Extract all named entities from the text below.\n"
            f"{type_hint}\n\n"
            f"Return a JSON array of objects with keys 'name' and 'type'.\n"
            f"Only include entities explicitly named in the text.\n"
            f"Do not invent entities. Return valid JSON only.\n\n"
            f"TEXT:\n{text[:8000]}\n\n"
            f"JSON:"
        )
        try:
            raw = self._llm.generate(prompt)
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                parsed = _json.loads(match.group())
                if isinstance(parsed, list):
                    results = []
                    seen: set[str] = set()
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name", "")).strip()
                        etype = str(item.get("type", "entity")).strip().lower()
                        if name and name not in seen:
                            seen.add(name)
                            results.append({"name": name, "type": etype})
                    return results
        except Exception:
            pass
        return []

    def _spacy_extract(self, text: str) -> list[dict[str, str]]:
        """Generic spaCy NER — used when LLM is unavailable."""
        if self._nlp is None:
            return []
        doc = self._nlp(text)
        spacy_map = {
            "ORG": "company",
            "PERSON": "person",
            "GPE": "location",
            "FAC": "institution",
            "NORP": "org",
        }
        entities = []
        seen: set[str] = set()
        for ent in doc.ents:
            if ent.text not in seen:
                entities.append({
                    "name": ent.text,
                    "type": spacy_map.get(ent.label_, ent.label_.lower()),
                })
                seen.add(ent.text)
        return entities

    def run(self, input_data: str) -> list[dict[str, str]]:
        """
        Parameters
        ----------
        input_data : str
            Raw text to extract entities from.

        Returns
        -------
        list of dicts with keys: name, type
        """
        text = str(input_data)
        if self._llm is not None:
            return self._llm_extract(text)
        return self._spacy_extract(text)


# ---------------------------------------------------------------------------
# SummarizeTool
# ---------------------------------------------------------------------------

class SummarizeTool(Tool):
    """
    Summarises a piece of text using the LLM backend.

    Parameters
    ----------
    llm : LLM
        Any LLM instance (MockLLM, OpenAILLM, etc.).
    max_length : int
        Approximate target summary length in words.
    """

    name = "summarize"
    description = "Summarises a text passage. Input: text string."

    def __init__(self, llm: Any, max_length: int = 80) -> None:
        self._llm = llm
        self.max_length = max_length

    def run(self, input_data: str) -> str:
        """
        Parameters
        ----------
        input_data : str
            Text to summarise.

        Returns
        -------
        str
            Summary text.
        """
        text = str(input_data)
        prompt = (
            f"Summarise the following text in at most {self.max_length} words, "
            f"focusing on the key entities, relationships, and important facts.\n\n"
            f"TEXT:\n{text}\n\nSUMMARY:"
        )
        return self._llm.generate(prompt)


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------

#: Legacy name — points to the new multimodal tool.
VectorSearchTool = MultimodalVectorSearchTool

#: Short alias used by the controller.
Qwen3RerankerTool = Qwen3VLRerankerTool


# ---------------------------------------------------------------------------
# GraphNeighborsTool
# ---------------------------------------------------------------------------

class GraphNeighborsTool(Tool):
    """
    Returns all entities directly connected to a given node in the knowledge graph.

    Parameters
    ----------
    memory : AgentMemory
        Shared memory holding the KnowledgeGraph instance.
    """

    name = "graph_neighbors"
    description = (
        "Returns all entities directly connected to a given node in the knowledge graph. "
        "Input: node name (string) or dict with key 'node'."
    )

    def __init__(self, memory: AgentMemory) -> None:
        self._memory = memory

    def run(self, input_data: str | dict) -> dict:
        """
        Parameters
        ----------
        input_data : str | dict
            Node name as a string, or a dict with key ``node``.

        Returns
        -------
        dict with keys: node, neighbors (list of str), relationships (list of dicts).
        If node is not found an additional ``error`` key is included.
        """
        if isinstance(input_data, dict):
            node = str(input_data.get("node", ""))
        else:
            node = str(input_data).strip()

        graph = self._memory.graph
        entity = graph.get_entity(node)
        if entity is None:
            return {
                "node": node,
                "neighbors": [],
                "relationships": [],
                "error": "not found",
            }

        neighbors = graph.neighbors(node)
        relationships = graph.get_relationships(node)
        return {
            "node": node,
            "neighbors": neighbors,
            "relationships": relationships,
        }


# ---------------------------------------------------------------------------
# GraphShortestPathTool
# ---------------------------------------------------------------------------

class GraphShortestPathTool(Tool):
    """
    Finds the shortest path between two entities in the knowledge graph.

    Parameters
    ----------
    memory : AgentMemory
        Shared memory holding the KnowledgeGraph instance.
    """

    name = "graph_shortest_path"
    description = (
        "Finds the shortest path between two entities in the knowledge graph. "
        "Input: dict with keys 'source' and 'target'."
    )

    def __init__(self, memory: AgentMemory) -> None:
        self._memory = memory

    def run(self, input_data: str | dict) -> dict:
        """
        Parameters
        ----------
        input_data : dict
            Must have keys ``source`` and ``target``.

        Returns
        -------
        dict with keys: source, target, path (list of str), length (int).
        If no path exists: path=[], length=-1, and an ``error`` key is added.
        """
        if isinstance(input_data, str):
            # attempt to parse "source -> target"
            parts = [p.strip() for p in input_data.split("->")]
            source = parts[0] if len(parts) >= 1 else ""
            target = parts[1] if len(parts) >= 2 else ""
        else:
            source = str(input_data.get("source", ""))
            target = str(input_data.get("target", ""))

        graph = self._memory.graph
        path = graph.shortest_path(source, target)

        if path is None:
            return {
                "source": source,
                "target": target,
                "path": [],
                "length": -1,
                "error": "no path",
            }

        return {
            "source": source,
            "target": target,
            "path": path,
            "length": len(path) - 1,  # number of edges
        }


# ---------------------------------------------------------------------------
# Tool registry helper
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Simple registry that maps tool names to Tool instances.

    The agent controller uses this to look up tools by name.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())
