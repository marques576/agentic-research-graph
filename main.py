"""
main.py – runnable entry point for the knowledge graph builder.

Usage
-----
    # Run with mock LLM (no API key, smoke-test only):
    python main.py

    # Choose embedding model size (all run fully locally):
    python main.py --embedding-model Qwen/Qwen3-VL-Embedding-2B   # default
    python main.py --embedding-model Qwen/Qwen3-VL-Embedding-8B   # best quality

    # Disable the reranker (faster, lower precision):
    python main.py --no-reranker

    # Whisper model size for audio/video transcription:
    python main.py --whisper-size base   # tiny | base | small | medium | large

    # Use an LLM backend for reasoning:
    python main.py --llm openrouter --model mistralai/mistral-7b-instruct --api-key sk-or-...
    python main.py --llm ollama --model llama3
    python main.py --llm lmstudio --model qwen/qwen3-30b-a3b
    python main.py --llm lmstudio --model qwen/qwen3-30b-a3b --base-url http://localhost:1234/v1

    # Custom goal:
    python main.py --goal "Map the relationships between entities in my docs"

Data folder
-----------
  Drop any supported file type into  project/data/  (sub-folders are scanned automatically):
    Text   : .txt  .md  .rst  .csv  .json  .xml  .html
    PDF    : .pdf
    Images : .jpg  .jpeg  .png  .gif  .bmp  .tiff  .webp
    Audio  : .mp3  .wav  .ogg  .flac  .m4a
    Video  : .mp4  .avi  .mov  .mkv  .webm
    Office : .docx  .pptx  .xlsx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Suppress duplicate OpenMP runtime warning on macOS (PyTorch + faiss/scipy
# each bundle their own libomp.dylib).  Must be set before any native libs load.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Make sure the project root is on the Python path when running directly
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from controller.research_controller import ResearchController
from llm.llm import LLM, MockLLM, OllamaLLM, OpenRouterLLM, LMStudioLLM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Knowledge Discovery Agent – multimodal drop-folder runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Research goal ----
    parser.add_argument(
        "--goal",
        default="Analyse this document corpus and discover hidden relationships.",
        help="Research goal / question for the agent.",
    )

    # ---- Embedding model ----
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-VL-Embedding-2B",
        help=(
            "Qwen3-VL-Embedding model for local multimodal embeddings.\n"
            "  Qwen/Qwen3-VL-Embedding-2B  – ~5 GB VRAM (default)\n"
            "  Qwen/Qwen3-VL-Embedding-8B  – ~18 GB VRAM, best quality"
        ),
    )

    # ---- Reranker model ----
    parser.add_argument(
        "--reranker-model",
        default="Qwen/Qwen3-VL-Reranker-2B",
        help=(
            "Qwen3-VL-Reranker model for optional re-ranking.\n"
            "  Qwen/Qwen3-VL-Reranker-2B  – ~5 GB VRAM (default)\n"
            "  Qwen/Qwen3-VL-Reranker-8B  – ~18 GB VRAM, best quality"
        ),
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        default=False,
        help="Disable the reranker (faster, lower precision).",
    )

    # ---- Audio / video transcription ----
    parser.add_argument(
        "--whisper-size",
        default="turbo",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model size for audio/video transcription (default: turbo).",
    )

    # ---- LLM backend ----
    parser.add_argument(
        "--llm",
        choices=["mock", "ollama", "openrouter", "lmstudio"],
        default="mock",
        help="LLM backend (default: mock).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "LLM model name.\n"
            "  openrouter: mistralai/mistral-7b-instruct\n"
            "  ollama:     llama3\n"
            "  lmstudio:   qwen/qwen3-30b-a3b"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM backend (or set OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "(lmstudio / ollama) Override the server base URL.\n"
            "  lmstudio default: http://localhost:1234/v1\n"
            "  ollama default:   http://localhost:11434"
        ),
    )
    parser.add_argument(
        "--site-url",
        default="",
        help="(OpenRouter only) HTTP-Referer header.",
    )

    # ---- Loop control ----
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Number of agent loop iterations (default: 3).",
    )

    # ---- Ontology ----
    parser.add_argument(
        "--ontology-path",
        default=None,
        metavar="PATH",
        help="Path to save / load the learned ontology JSON (default: ontology.json in CWD).",
    )

    return parser


def build_llm(args: argparse.Namespace) -> LLM:
    """Instantiate the selected LLM backend."""
    if args.llm == "lmstudio":
        model = args.model or "qwen/qwen3-30b-a3b"
        base_url = getattr(args, "base_url", None) or "http://localhost:1234/v1"
        print(f"Using LM Studio backend  (model: {model}, server: {base_url})")
        return LMStudioLLM(model=model, base_url=base_url)

    if args.llm == "openrouter":
        model = args.model or "mistralai/mistral-7b-instruct"
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        print(f"Using OpenRouter backend  (model: {model})")
        return OpenRouterLLM(
            model=model,
            api_key=api_key,
            site_url=getattr(args, "site_url", ""),
        )

    if args.llm == "ollama":
        model = args.model or "llama3"
        base_url = getattr(args, "base_url", None) or "http://localhost:11434"
        print(f"Using Ollama backend  (model: {model}, server: {base_url})")
        return OllamaLLM(model=model, base_url=base_url)

    print("Using MockLLM backend (no API key required)")
    return MockLLM()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    llm = build_llm(args)

    # --- Full run: ingest → ontology → agent loop ---
    data_dir = _ROOT / "data"

    controller = ResearchController(
        llm=llm,
        data_dir=data_dir,
        max_iterations=args.max_iterations,
        embedding_model_id=args.embedding_model,
        reranker_model_id=None if args.no_reranker else args.reranker_model,
        ingestion_kwargs={
            "whisper_model_size": args.whisper_size,
        },
        helper_prompt=args.goal,
        ontology_path=getattr(args, "ontology_path", None),
    )

    report = controller.run(goal=args.goal)

    # Always write graph.html (live snapshots update it during the run too)
    html_out = Path.cwd() / "graph.html"
    try:
        ResearchController.export_html_graph(report, html_out)
        print(f"\nKnowledge graph: {html_out.resolve()}\n")
    except Exception as exc:
        print(f"Warning: could not write graph.html: {exc}\n")

    # Always save the full JSON report
    json_out = Path.cwd() / "graph.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {json_out.resolve()}\n")




if __name__ == "__main__":
    main()
